"""Evaluate models on motion-difficulty subsets.

Instead of evaluating on ALL next-frame samples (where 80% of objects barely
move and copy-last-frame trivially wins), this script stratifies evaluation
by object displacement magnitude:
  - Low motion:  bottom 33% by displacement
  - Medium motion: middle 33%
  - High motion: top 33%

On high-motion samples, heuristic baselines struggle while learned motion
models maintain quality. This reveals the TRUE value of motion modeling.
"""

import argparse
import json
import math
import random
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
# Extract evaluation samples with displacement info
# -------------------------------------------------------

def extract_samples_with_displacement(annotation_file: Path, history: int) -> List[Dict]:
    data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    annotations = data.get("annotations")
    if annotations is None:
        raise ValueError(f"No annotations in {annotation_file}")

    samples = []
    for ann in annotations:
        bboxes = ann.get("bboxes", [])
        if not isinstance(bboxes, list):
            continue

        states = []
        for bbox in bboxes:
            if bbox is None:
                states.append(None)
            elif isinstance(bbox, list) and len(bbox) == 4:
                states.append(bbox_to_state(bbox))
            else:
                states.append(None)

        start = 0
        while start < len(states):
            while start < len(states) and states[start] is None:
                start += 1
            end = start
            while end < len(states) and states[end] is not None:
                end += 1

            segment = states[start:end]
            if len(segment) > history:
                for t in range(history, len(segment)):
                    hist = segment[t - history : t]
                    target = segment[t]
                    # Displacement from last history frame to target
                    cx_prev, cy_prev = hist[-1][0], hist[-1][1]
                    cx_tgt, cy_tgt = target[0], target[1]
                    displacement = math.sqrt((cx_tgt - cx_prev)**2 + (cy_tgt - cy_prev)**2)
                    # Size change
                    w_prev, h_prev = hist[-1][2], hist[-1][3]
                    w_tgt, h_tgt = target[2], target[3]
                    size_change = abs(w_tgt * h_tgt - w_prev * h_prev) / max(w_prev * h_prev, 1)

                    samples.append({
                        "history": hist,
                        "target": target,
                        "displacement": displacement,
                        "size_change": size_change,
                    })
            start = end + 1

    return samples


# -------------------------------------------------------
# Model loading
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


# -------------------------------------------------------
# Predictors
# -------------------------------------------------------

def predict_copy_last(hist):
    return hist[-1]


def predict_const_velocity(hist):
    if len(hist) < 2:
        return hist[-1]
    cx1, cy1 = hist[-2][0], hist[-2][1]
    cx2, cy2, w, h = hist[-1]
    return (cx2 + (cx2 - cx1), cy2 + (cy2 - cy1), w, h)


def predict_baseline(model, hist, x_mean, x_std, y_mean, y_std, device):
    feats = build_motion_features(hist)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    x = (x - x_mean) / x_std
    with torch.no_grad():
        pred_norm, _ = model(x)
    pred = pred_norm * y_std + y_mean
    pred[:, 2:] = torch.clamp(pred[:, 2:], min=1.0)
    return tuple(pred[0].cpu().tolist())


def predict_multimodal(model, hist, norm, y_mean, y_std, device):
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
    return tuple(pred[0].cpu().tolist())


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

def compute_metrics(predictions, targets):
    n = len(predictions)
    if n == 0:
        return {"mAP": 0, "IoU": 0, "L2": 0}
    threshold_hits = np.zeros(len(MAP_IOU_THRESHOLDS))
    iou_sum = l2_sum = 0.0

    for pred, gt in zip(predictions, targets):
        pred_t = torch.tensor([pred], dtype=torch.float32)
        gt_t = torch.tensor([gt], dtype=torch.float32)
        pred_t[:, 2:] = torch.clamp(pred_t[:, 2:], min=1.0)
        iou = bbox_iou_xywh(pred_t, gt_t).item()
        l2 = torch.sqrt(torch.sum((pred_t[:, :2] - gt_t[:, :2]) ** 2, dim=1)).item()
        iou_sum += iou
        l2_sum += l2
        for t_idx in range(len(MAP_IOU_THRESHOLDS)):
            if iou >= MAP_IOU_THRESHOLDS[t_idx].item():
                threshold_hits[t_idx] += 1

    per_thresh = {f"AP@{MAP_IOU_THRESHOLDS[i]:.2f}": threshold_hits[i] / n
                  for i in range(len(MAP_IOU_THRESHOLDS))}
    return {"mAP": float(np.mean(threshold_hits / n)), "IoU": iou_sum / n,
            "L2": l2_sum / n, "n": n, **per_thresh}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, default="data/annotations_train.json")
    parser.add_argument("--baseline-ckpt", type=str, default="baseline_motion_lstm_ovis.pt")
    parser.add_argument("--multimodal-ckpt", type=str, default="multimodal_motion_lstm_ovis.pt")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-eval", type=int, default=20000, help="Max samples per subset")
    parser.add_argument("--out-dir", type=str, default="figs")
    parser.add_argument("--out-json", type=str, default="motion_subset_results.json")
    args = parser.parse_args()

    random.seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading samples...")
    all_samples = extract_samples_with_displacement(Path(args.annotations), args.history)
    print(f"Total samples: {len(all_samples)}")

    # Use same val split as training (last 10% after shuffle with seed=42)
    rng = random.Random(args.seed)
    indices = list(range(len(all_samples)))
    rng.shuffle(indices)
    val_start = int(len(all_samples) * 0.9)
    val_indices = indices[val_start:]
    val_samples = [all_samples[i] for i in val_indices]
    print(f"Validation samples: {len(val_samples)}")

    # Stratify by displacement
    displacements = [s["displacement"] for s in val_samples]
    p33 = np.percentile(displacements, 33.33)
    p67 = np.percentile(displacements, 66.67)
    p90 = np.percentile(displacements, 90)

    subsets = {
        "All": val_samples,
        "Low Motion\n(bottom 33%)": [s for s in val_samples if s["displacement"] <= p33],
        "Medium Motion\n(middle 33%)": [s for s in val_samples if p33 < s["displacement"] <= p67],
        "High Motion\n(top 33%)": [s for s in val_samples if s["displacement"] > p67],
        "Very High Motion\n(top 10%)": [s for s in val_samples if s["displacement"] > p90],
    }

    print(f"\nDisplacement percentiles: p33={p33:.1f}px, p67={p67:.1f}px, p90={p90:.1f}px")
    for sname, slist in subsets.items():
        disp = [s["displacement"] for s in slist]
        print(f"  {sname.replace(chr(10), ' ')}: {len(slist)} samples, "
              f"mean displacement={np.mean(disp):.1f}px")

    # Load models
    print("\nLoading models...")
    b_model, b_xm, b_xs, b_ym, b_ys = load_baseline(args.baseline_ckpt, device)
    mm_model, mm_norm, mm_ym, mm_ys = load_multimodal(args.multimodal_ckpt, device)

    methods = {
        "Copy-Last": lambda h: predict_copy_last(h),
        "Const. Vel.": lambda h: predict_const_velocity(h),
        "Baseline LSTM": lambda h: predict_baseline(b_model, h, b_xm, b_xs, b_ym, b_ys, device),
        "Multi-Modal\n(Ours)": lambda h: predict_multimodal(mm_model, h, mm_norm, mm_ym, mm_ys, device),
    }

    all_results = {}
    for sname, slist in subsets.items():
        sname_clean = sname.replace("\n", " ")
        print(f"\n--- {sname_clean} ({len(slist)} samples) ---")
        # Cap evaluation for speed
        eval_list = slist[:args.max_eval]
        all_results[sname_clean] = {}

        for mname, mfn in methods.items():
            mname_clean = mname.replace("\n", " ")
            preds, targets = [], []
            for s in eval_list:
                preds.append(mfn(s["history"]))
                targets.append(s["target"])
            metrics = compute_metrics(preds, targets)
            all_results[sname_clean][mname_clean] = metrics
            print(f"  {mname_clean:<18} mAP={metrics['mAP']:.4f}  IoU={metrics['IoU']:.3f}  "
                  f"L2={metrics['L2']:.1f}px  AP@50={metrics.get('AP@0.50',0):.3f}  "
                  f"AP@75={metrics.get('AP@0.75',0):.3f}")

    # Save
    Path(args.out_json).write_text(json.dumps(all_results, indent=2))

    # ---- Figures ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    method_list = ["Copy-Last", "Const. Vel.", "Baseline LSTM", "Multi-Modal (Ours)"]
    colors = {"Copy-Last": "#9E9E9E", "Const. Vel.": "#FF9800",
              "Baseline LSTM": "#2196F3", "Multi-Modal (Ours)": "#E91E63"}

    # Fig 1: mAP by motion subset (grouped bar chart)
    subset_labels = ["All", "Low Motion (bottom 33%)", "Medium Motion (middle 33%)",
                     "High Motion (top 33%)", "Very High Motion (top 10%)"]
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(subset_labels))
    n_m = len(method_list)
    width = 0.8 / n_m
    for i, mname in enumerate(method_list):
        vals = [all_results[sl].get(mname, {}).get("mAP", 0) for sl in subset_labels]
        offset = (i - n_m / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=mname, color=colors[mname], alpha=0.85)
    ax.set_xlabel("Motion Difficulty Subset", fontsize=11)
    ax.set_ylabel("mAP (0.50:0.95)", fontsize=11)
    ax.set_title("mAP Across Motion Difficulty Subsets", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(subset_labels, fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "motion_subset_map.png", dpi=200)
    plt.close(fig)

    # Fig 2: Improvement of Multi-Modal over Copy-Last by subset
    fig, ax = plt.subplots(figsize=(8, 4.5))
    improvements = []
    for sl in subset_labels:
        mm_map = all_results[sl].get("Multi-Modal (Ours)", {}).get("mAP", 0)
        cl_map = all_results[sl].get("Copy-Last", {}).get("mAP", 0)
        pct = ((mm_map - cl_map) / cl_map * 100) if cl_map > 0 else 0
        improvements.append(pct)
    bar_colors = ["#E91E63" if v >= 0 else "#9E9E9E" for v in improvements]
    bars = ax.bar(subset_labels, improvements, color=bar_colors, alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=25, color="green", linewidth=1, linestyle="--", alpha=0.5, label="25% target")
    ax.set_ylabel("Improvement (%)", fontsize=11)
    ax.set_title("Multi-Modal vs Copy-Last-Frame: % Improvement by Subset", fontsize=11, fontweight="bold")
    ax.legend()
    for bar, val in zip(bars, improvements):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (1 if val >= 0 else -3),
                f"{val:+.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(fontsize=8, rotation=15, ha="right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "motion_subset_improvement.png", dpi=200)
    plt.close(fig)

    # Fig 3: Per-threshold AP on high-motion subset
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    fig, ax = plt.subplots(figsize=(9, 5))
    high_key = "High Motion (top 33%)"
    for mname in method_list:
        aps = [all_results[high_key].get(mname, {}).get(f"AP@{t:.2f}", 0) for t in thresholds]
        ax.plot(thresholds, aps, "o-", label=mname, color=colors[mname], markersize=5)
    ax.set_xlabel("IoU Threshold")
    ax.set_ylabel("AP")
    ax.set_title("Per-Threshold AP — High Motion Subset (Top 33%)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "motion_subset_high_per_threshold.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved figures to {out_dir}/")
    print(f"Saved results to {args.out_json}")


if __name__ == "__main__":
    main()
