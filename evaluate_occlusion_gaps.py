"""Evaluate models on occlusion-gap prediction.

For each instance that has a visible→occluded→visible pattern, use the
pre-gap visible frames as history and predict the FIRST post-gap frame.

This directly tests the paper's thesis: can motion modeling help predict
where an object reappears after being hidden?

Evaluates:
  1. Copy-Last-Frame (heuristic): predict = last visible box
  2. Constant Velocity (heuristic): linear extrapolation across the gap
  3. Baseline LSTM (learned): uses motion features from pre-gap frames
  4. Multi-Modal LSTM (learned): uses multi-modal motion features
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from baseline_motion_model import (
    MotionLSTMBaseline,
    bbox_to_state,
    build_motion_features,
    bbox_iou_xywh,
)
from multimodal_motion_model import (
    MultiModalMotionLSTM,
    build_multimodal_features,
)


MAP_IOU_THRESHOLDS = torch.arange(0.50, 1.0, 0.05)


# -------------------------------------------------------
# Extract occlusion-gap sequences
# -------------------------------------------------------

def extract_occlusion_gaps(
    annotation_file: Path,
    min_pre_visible: int = 5,
    min_gap: int = 1,
    max_gap: int = 50,
) -> List[Dict]:
    """Extract visible→occluded→visible sequences.

    Returns list of dicts with:
      - pre_states: list of (cx,cy,w,h) from visible frames before the gap
      - gap_length: number of occluded frames
      - target: (cx,cy,w,h) of the first visible frame after the gap
      - video_id, instance_id: for tracking
    """
    data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    annotations = data.get("annotations")
    if annotations is None:
        raise ValueError(f"No annotations in {annotation_file}")

    gaps = []
    for ann in annotations:
        bboxes = ann.get("bboxes", [])
        vid_id = ann.get("video_id", -1)
        inst_id = ann.get("id", -1)

        states = []
        for bbox in bboxes:
            if bbox is None:
                states.append(None)
            elif isinstance(bbox, list) and len(bbox) == 4:
                states.append(bbox_to_state(bbox))
            else:
                states.append(None)

        i = 0
        while i < len(states):
            # Find visible segment
            while i < len(states) and states[i] is None:
                i += 1
            vis_start = i
            while i < len(states) and states[i] is not None:
                i += 1
            vis_end = i
            pre_len = vis_end - vis_start

            if pre_len < min_pre_visible:
                continue

            # Count gap
            gap_start = i
            while i < len(states) and states[i] is None:
                i += 1
            gap_len = i - gap_start

            if gap_len < min_gap or gap_len > max_gap:
                continue

            # Check for visible frame after gap
            if i < len(states) and states[i] is not None:
                pre_states = [states[t] for t in range(vis_start, vis_end)]
                gaps.append({
                    "pre_states": pre_states,
                    "gap_length": gap_len,
                    "target": states[i],
                    "video_id": vid_id,
                    "instance_id": inst_id,
                })

    return gaps


# -------------------------------------------------------
# Heuristic predictors
# -------------------------------------------------------

def predict_copy_last(pre_states, gap_length):
    return pre_states[-1]


def predict_const_velocity(pre_states, gap_length):
    if len(pre_states) < 2:
        return pre_states[-1]
    cx1, cy1 = pre_states[-2][0], pre_states[-2][1]
    cx2, cy2, w, h = pre_states[-1]
    vx, vy = cx2 - cx1, cy2 - cy1
    # Extrapolate across the gap
    return (cx2 + vx * (gap_length + 1), cy2 + vy * (gap_length + 1), w, h)


def predict_avg_velocity(pre_states, gap_length):
    if len(pre_states) < 2:
        return pre_states[-1]
    n = len(pre_states) - 1
    cx_first, cy_first = pre_states[0][0], pre_states[0][1]
    cx_last, cy_last, w, h = pre_states[-1]
    avg_vx = (cx_last - cx_first) / n
    avg_vy = (cy_last - cy_first) / n
    return (cx_last + avg_vx * (gap_length + 1), cy_last + avg_vy * (gap_length + 1), w, h)


# -------------------------------------------------------
# Learned model predictors
# -------------------------------------------------------

def load_baseline(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MotionLSTMBaseline(input_dim=10, hidden_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["feature_mean"].to(device), ckpt["feature_std"].to(device), \
           ckpt["target_mean"].to(device), ckpt["target_std"].to(device)


def load_multimodal(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiModalMotionLSTM(
        hidden_dim=ckpt.get("hidden_dim", 64),
        num_layers=ckpt.get("num_layers", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = {k: (m.to(device), s.to(device)) for k, (m, s) in ckpt["norm_stats"].items()}
    return model, norm, ckpt["target_mean"].to(device), ckpt["target_std"].to(device)


def predict_baseline_across_gap(model, pre_states, gap_length, history,
                                 x_mean, x_std, y_mean, y_std, device):
    """Iteratively predict through the gap, feeding predictions back."""
    states = list(pre_states)
    for step in range(gap_length + 1):
        hist = states[-history:]
        feats = build_motion_features(hist)
        x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
        x = (x - x_mean) / x_std
        with torch.no_grad():
            pred_norm, _ = model(x)
        pred = pred_norm * y_std + y_mean
        pred[:, 2:] = torch.clamp(pred[:, 2:], min=1.0)
        pred_tuple = tuple(pred[0].cpu().tolist())
        states.append(pred_tuple)
    return states[-1]


def predict_multimodal_across_gap(model, pre_states, gap_length, history,
                                   norm, y_mean, y_std, device):
    """Iteratively predict through the gap, feeding predictions back."""
    states = list(pre_states)
    for step in range(gap_length + 1):
        hist = states[-history:]
        mm_feats = build_multimodal_features(hist)
        vel = torch.tensor([f[0] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)
        shape = torch.tensor([f[1] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)
        accel = torch.tensor([f[2] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)
        ctx = torch.tensor([f[3] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)

        vel = (vel - norm["vel"][0]) / norm["vel"][1]
        shape = (shape - norm["shape"][0]) / norm["shape"][1]
        accel = (accel - norm["accel"][0]) / norm["accel"][1]
        ctx = (ctx - norm["ctx"][0]) / norm["ctx"][1]

        with torch.no_grad():
            pred_norm, _, _ = model(vel, shape, accel, ctx, return_gates=True)
        pred = pred_norm * y_std + y_mean
        pred[:, 2:] = torch.clamp(pred[:, 2:], min=1.0)
        pred_tuple = tuple(pred[0].cpu().tolist())
        states.append(pred_tuple)
    return states[-1]


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def compute_metrics(predictions, targets):
    """Compute mAP and per-threshold AP from prediction/target lists."""
    n = len(predictions)
    threshold_hits = np.zeros(len(MAP_IOU_THRESHOLDS))
    iou_sum = l2_sum = mse_sum = 0.0

    for pred, gt in zip(predictions, targets):
        pred_t = torch.tensor([pred], dtype=torch.float32)
        gt_t = torch.tensor([gt], dtype=torch.float32)
        pred_t[:, 2:] = torch.clamp(pred_t[:, 2:], min=1.0)

        iou = bbox_iou_xywh(pred_t, gt_t).item()
        l2 = torch.sqrt(torch.sum((pred_t[:, :2] - gt_t[:, :2]) ** 2, dim=1)).item()
        mse = torch.mean((pred_t - gt_t) ** 2, dim=1).item()

        iou_sum += iou
        l2_sum += l2
        mse_sum += mse
        for t_idx in range(len(MAP_IOU_THRESHOLDS)):
            if iou >= MAP_IOU_THRESHOLDS[t_idx].item():
                threshold_hits[t_idx] += 1

    per_thresh = {f"AP@{MAP_IOU_THRESHOLDS[i]:.2f}": threshold_hits[i] / n
                  for i in range(len(MAP_IOU_THRESHOLDS))}
    return {
        "mAP": float(np.mean(threshold_hits / n)),
        "IoU": iou_sum / n,
        "L2": l2_sum / n,
        "MSE": mse_sum / n,
        **per_thresh,
    }


def main():
    parser = argparse.ArgumentParser(description="Occlusion-gap prediction evaluation")
    parser.add_argument("--annotations", type=str, default="data/annotations_train.json")
    parser.add_argument("--baseline-ckpt", type=str, default="baseline_motion_lstm_ovis.pt")
    parser.add_argument("--multimodal-ckpt", type=str, default="multimodal_motion_lstm_ovis.pt")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default="figs")
    parser.add_argument("--out-json", type=str, default="occlusion_gap_results.json")
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Extract gaps
    print("Extracting occlusion-gap sequences...")
    gaps = extract_occlusion_gaps(
        Path(args.annotations),
        min_pre_visible=args.history,
        min_gap=1,
    )
    print(f"Found {len(gaps)} occlusion-gap sequences")

    # Bucket by gap length
    gap_buckets = {"1": [], "2-3": [], "4-6": [], "7-10": [], "11+": []}
    for g in gaps:
        gl = g["gap_length"]
        if gl == 1:
            gap_buckets["1"].append(g)
        elif gl <= 3:
            gap_buckets["2-3"].append(g)
        elif gl <= 6:
            gap_buckets["4-6"].append(g)
        elif gl <= 10:
            gap_buckets["7-10"].append(g)
        else:
            gap_buckets["11+"].append(g)

    for bname, blist in gap_buckets.items():
        print(f"  Gap {bname}: {len(blist)} sequences")

    # Load learned models
    print("Loading baseline checkpoint...")
    b_model, b_xm, b_xs, b_ym, b_ys = load_baseline(args.baseline_ckpt, device)
    print("Loading multi-modal checkpoint...")
    mm_model, mm_norm, mm_ym, mm_ys = load_multimodal(args.multimodal_ckpt, device)

    # Define all methods
    methods = {
        "Copy-Last-Frame": lambda pre, gl: predict_copy_last(pre, gl),
        "Const. Velocity": lambda pre, gl: predict_const_velocity(pre, gl),
        "Avg. Velocity": lambda pre, gl: predict_avg_velocity(pre, gl),
        "Baseline LSTM": lambda pre, gl: predict_baseline_across_gap(
            b_model, pre, gl, args.history, b_xm, b_xs, b_ym, b_ys, device),
        "Multi-Modal (Ours)": lambda pre, gl: predict_multimodal_across_gap(
            mm_model, pre, gl, args.history, mm_norm, mm_ym, mm_ys, device),
    }

    # Evaluate on all gaps
    all_results = {}
    for mname, mfn in methods.items():
        print(f"\nEvaluating: {mname}")
        preds, targets = [], []
        for g in gaps:
            pred = mfn(g["pre_states"], g["gap_length"])
            preds.append(pred)
            targets.append(g["target"])

        overall = compute_metrics(preds, targets)
        all_results[mname] = {"overall": overall}

        print(f"  Overall: mAP={overall['mAP']:.4f}  IoU={overall['IoU']:.4f}  "
              f"L2={overall['L2']:.1f}px")
        print(f"  AP@0.50={overall['AP@0.50']:.3f}  AP@0.75={overall['AP@0.75']:.3f}  "
              f"AP@0.90={overall['AP@0.90']:.3f}")

        # Per-bucket evaluation
        for bname, blist in gap_buckets.items():
            if not blist:
                continue
            bp, bt = [], []
            for g in blist:
                bp.append(mfn(g["pre_states"], g["gap_length"]))
                bt.append(g["target"])
            bucket_metrics = compute_metrics(bp, bt)
            all_results[mname][f"gap_{bname}"] = bucket_metrics

    # Save results
    out_path = Path(args.out_json)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved results to {out_path}")

    # ---- Generate figures ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    method_names = list(methods.keys())
    colors = {"Copy-Last-Frame": "#9E9E9E", "Const. Velocity": "#FF9800",
              "Avg. Velocity": "#795548", "Baseline LSTM": "#2196F3",
              "Multi-Modal (Ours)": "#E91E63"}

    # Fig 1: Overall mAP bar chart
    fig, ax = plt.subplots(figsize=(8, 4.5))
    maps = [all_results[m]["overall"]["mAP"] for m in method_names]
    bars = ax.bar(method_names, maps, color=[colors[m] for m in method_names], alpha=0.85)
    ax.set_ylabel("mAP (0.50:0.95)")
    ax.set_title("Occlusion-Gap Prediction: Overall mAP")
    ax.set_ylim(0, max(maps) * 1.15)
    for bar, val in zip(bars, maps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "occlusion_gap_overall_map.png", dpi=200)
    plt.close(fig)

    # Fig 2: mAP by gap length
    bucket_names = [b for b in gap_buckets if gap_buckets[b]]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(bucket_names))
    n_methods = len(method_names)
    width = 0.8 / n_methods
    for i, mname in enumerate(method_names):
        vals = []
        for bname in bucket_names:
            key = f"gap_{bname}"
            vals.append(all_results[mname].get(key, {}).get("mAP", 0))
        offset = (i - n_methods / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=mname, color=colors[mname], alpha=0.85)
    ax.set_xlabel("Occlusion Gap Length (frames)")
    ax.set_ylabel("mAP (0.50:0.95)")
    ax.set_title("mAP by Occlusion Gap Duration")
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_names)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "occlusion_gap_by_duration.png", dpi=200)
    plt.close(fig)

    # Fig 3: Per-threshold AP comparison (overall)
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    fig, ax = plt.subplots(figsize=(9, 5))
    for mname in method_names:
        aps = [all_results[mname]["overall"].get(f"AP@{t:.2f}", 0) for t in thresholds]
        ax.plot(thresholds, aps, "o-", label=mname, color=colors[mname], markersize=4)
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("AP")
    ax.set_title("Per-Threshold AP on Occlusion-Gap Prediction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "occlusion_gap_per_threshold.png", dpi=200)
    plt.close(fig)

    # Print summary table
    print("\n" + "=" * 80)
    print("OCCLUSION-GAP PREDICTION RESULTS")
    print("=" * 80)
    print(f"{'Method':<22} {'mAP':>6} {'AP@50':>6} {'AP@75':>6} {'AP@90':>6} "
          f"{'IoU':>6} {'L2(px)':>7}")
    print("-" * 80)
    for mname in method_names:
        r = all_results[mname]["overall"]
        print(f"{mname:<22} {r['mAP']:>6.3f} {r['AP@0.50']:>6.3f} "
              f"{r['AP@0.75']:>6.3f} {r['AP@0.90']:>6.3f} "
              f"{r['IoU']:>6.3f} {r['L2']:>7.1f}")
    print("-" * 80)

    # Print improvement of Multi-Modal over each baseline
    mm = all_results["Multi-Modal (Ours)"]["overall"]
    print("\nImprovement of Multi-Modal over each method:")
    for mname in method_names[:-1]:
        r = all_results[mname]["overall"]
        delta_map = mm["mAP"] - r["mAP"]
        pct = (delta_map / r["mAP"]) * 100 if r["mAP"] > 0 else 0
        print(f"  vs {mname}: mAP +{delta_map:.4f} ({pct:+.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
