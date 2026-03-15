"""Visualize model predictions on OVIS video sequences.

Loads baseline and multi-modal checkpoints, runs predictions on sample
sequences from the training set (which has ground truth), and renders
side-by-side comparisons showing:
  - Actual video frames with GT masks overlaid
  - Predicted bounding boxes vs ground truth bounding boxes
  - Motion trajectory trails
"""

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from PIL import Image
from pycocotools import mask as mask_util

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

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def decode_rle_mask(rle: dict, h: int, w: int) -> np.ndarray:
    """Decode OVIS RLE segmentation to binary mask."""
    if isinstance(rle.get("counts"), list):
        rle = mask_util.frPyObjects(rle, h, w)
    return mask_util.decode(rle).astype(np.uint8)


def load_baseline_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MotionLSTMBaseline(input_dim=10, hidden_dim=128).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x_mean = ckpt["feature_mean"].to(device)
    x_std = ckpt["feature_std"].to(device)
    y_mean = ckpt["target_mean"].to(device)
    y_std = ckpt["target_std"].to(device)
    return model, x_mean, x_std, y_mean, y_std


def load_multimodal_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiModalMotionLSTM(
        hidden_dim=ckpt.get("hidden_dim", 64),
        num_layers=ckpt.get("num_layers", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    norm = {}
    for k, (m, s) in ckpt["norm_stats"].items():
        norm[k] = (m.to(device), s.to(device))
    y_mean = ckpt["target_mean"].to(device)
    y_std = ckpt["target_std"].to(device)
    return model, norm, y_mean, y_std


def predict_baseline(model, states_history, x_mean, x_std, y_mean, y_std, device):
    """Run baseline model on a list of (cx,cy,w,h) states."""
    feats = build_motion_features(states_history)
    x = torch.tensor(feats, dtype=torch.float32).unsqueeze(0).to(device)
    x = (x - x_mean) / x_std
    with torch.no_grad():
        pred_norm, _ = model(x)
    pred = pred_norm * y_std + y_mean
    pred[:, 2:] = torch.clamp(pred[:, 2:], min=1.0)
    return pred[0].cpu().numpy()


def predict_multimodal(model, states_history, norm, y_mean, y_std, device):
    """Run multi-modal model on a list of (cx,cy,w,h) states."""
    mm_feats = build_multimodal_features(states_history)
    vel = torch.tensor([f[0] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)
    shape = torch.tensor([f[1] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)
    accel = torch.tensor([f[2] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)
    ctx = torch.tensor([f[3] for f in mm_feats], dtype=torch.float32).unsqueeze(0).to(device)

    vel = (vel - norm["vel"][0]) / norm["vel"][1]
    shape = (shape - norm["shape"][0]) / norm["shape"][1]
    accel = (accel - norm["accel"][0]) / norm["accel"][1]
    ctx = (ctx - norm["ctx"][0]) / norm["ctx"][1]

    with torch.no_grad():
        pred_norm, _, gates = model(vel, shape, accel, ctx, return_gates=True)
    pred = pred_norm * y_std + y_mean
    pred[:, 2:] = torch.clamp(pred[:, 2:], min=1.0)
    return pred[0].cpu().numpy(), gates[0].cpu().numpy()


def cxywh_to_xyxy(box):
    """Convert (cx, cy, w, h) -> (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2


def draw_box(ax, box_cxywh, color, label=None, linewidth=2, linestyle="-"):
    x1, y1, x2, y2 = cxywh_to_xyxy(box_cxywh)
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=linewidth, edgecolor=color, facecolor="none", linestyle=linestyle,
    )
    ax.add_patch(rect)
    if label:
        ax.text(x1, y1 - 4, label, fontsize=7, color=color,
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.6))


def overlay_mask(ax, mask, color, alpha=0.35):
    """Overlay a binary mask with a semi-transparent color."""
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[mask > 0] = [*color, alpha]
    ax.imshow(overlay)


# -------------------------------------------------------
# Main visualization
# -------------------------------------------------------

INSTANCE_COLORS = [
    (0.12, 0.56, 1.0),   # blue
    (1.0, 0.27, 0.0),    # red-orange
    (0.0, 0.8, 0.4),     # green
    (1.0, 0.84, 0.0),    # gold
    (0.58, 0.0, 0.83),   # purple
    (0.0, 0.81, 0.82),   # cyan
    (1.0, 0.41, 0.71),   # pink
    (0.55, 0.27, 0.07),  # brown
]


def visualize_sequence(
    video_meta: dict,
    annotations: list,
    data_root: Path,
    history: int,
    baseline_model, b_xmean, b_xstd, b_ymean, b_ystd,
    mm_model, mm_norm, mm_ymean, mm_ystd,
    device: torch.device,
    out_path: Path,
    max_frames: int = 12,
):
    """Visualize predictions for one video sequence."""
    file_names = video_meta["file_names"]
    vid_h, vid_w = video_meta["height"], video_meta["width"]
    num_frames = min(len(file_names), max_frames + history)

    # Filter annotations that have enough visible frames
    good_anns = []
    for ann in annotations:
        bboxes = ann["bboxes"]
        visible = [i for i in range(num_frames) if i < len(bboxes) and bboxes[i] is not None]
        if len(visible) > history + 1:
            good_anns.append(ann)
    if not good_anns:
        return False

    # Pick frames to visualize: we need history frames + prediction frames
    # Find a contiguous segment where at least one instance is visible
    start_frame = 0
    vis_frames = list(range(start_frame, min(start_frame + history + max_frames, num_frames)))

    # Create the figure: 2 rows x N cols
    # Row 1: GT masks + GT boxes on actual frames
    # Row 2: GT boxes (green) + baseline pred (blue dashed) + multimodal pred (red dashed)
    pred_frames = [f for f in vis_frames if f >= history]
    n_cols = min(len(pred_frames), 8)
    if n_cols < 1:
        return False
    pred_frames = pred_frames[:n_cols]

    fig, axes = plt.subplots(2, n_cols, figsize=(n_cols * 3.5, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    for col_idx, frame_idx in enumerate(pred_frames):
        # Load the actual image
        img_path = data_root / file_names[frame_idx]
        if img_path.exists():
            img = np.array(Image.open(img_path))
        else:
            img = np.zeros((vid_h, vid_w, 3), dtype=np.uint8)

        # --- Row 1: Ground truth masks + boxes ---
        ax1 = axes[0, col_idx]
        ax1.imshow(img)
        ax1.set_title(f"Frame {frame_idx}", fontsize=8)
        ax1.axis("off")

        # --- Row 2: Predictions vs GT ---
        ax2 = axes[1, col_idx]
        ax2.imshow(img)
        ax2.axis("off")

        for inst_idx, ann in enumerate(good_anns[:6]):  # limit instances
            color = INSTANCE_COLORS[inst_idx % len(INSTANCE_COLORS)]
            bboxes = ann["bboxes"]
            segs = ann.get("segmentations", [])
            occlusions = ann.get("occlusion", [])

            # Draw GT mask on row 1
            if frame_idx < len(segs) and segs[frame_idx] is not None:
                mask = decode_rle_mask(segs[frame_idx], vid_h, vid_w)
                overlay_mask(ax1, mask, color, alpha=0.4)

            # Draw GT box on row 1
            if frame_idx < len(bboxes) and bboxes[frame_idx] is not None:
                gt_state = bbox_to_state(bboxes[frame_idx])
                occ_label = occlusions[frame_idx] if frame_idx < len(occlusions) else ""
                occ_short = {"no_occlusion": "", "slight_occlusion": "S",
                             "severe_occlusion": "X"}.get(occ_label, "?")
                label = f"#{inst_idx}" + (f" [{occ_short}]" if occ_short else "")
                draw_box(ax1, gt_state, color, label=label, linewidth=2)

            # Build history for prediction
            hist_states = []
            for t in range(frame_idx - history, frame_idx):
                if 0 <= t < len(bboxes) and bboxes[t] is not None:
                    hist_states.append(bbox_to_state(bboxes[t]))

            if len(hist_states) < history:
                continue  # not enough history

            gt_state = bbox_to_state(bboxes[frame_idx]) if frame_idx < len(bboxes) and bboxes[frame_idx] is not None else None

            # Baseline prediction
            b_pred = predict_baseline(baseline_model, hist_states, b_xmean, b_xstd, b_ymean, b_ystd, device)
            # Multi-modal prediction
            mm_pred, gates = predict_multimodal(mm_model, hist_states, mm_norm, mm_ymean, mm_ystd, device)

            # Draw on row 2
            if gt_state is not None:
                draw_box(ax2, gt_state, (0.0, 1.0, 0.0), label="GT", linewidth=2)

                # Compute IoU for labels
                gt_t = torch.tensor([gt_state], dtype=torch.float32)
                b_iou = bbox_iou_xywh(torch.tensor([b_pred], dtype=torch.float32), gt_t).item()
                mm_iou = bbox_iou_xywh(torch.tensor([mm_pred], dtype=torch.float32), gt_t).item()
                draw_box(ax2, b_pred, (0.3, 0.5, 1.0), label=f"BL:{b_iou:.2f}", linewidth=1.5, linestyle="--")
                draw_box(ax2, mm_pred, (1.0, 0.2, 0.4), label=f"MM:{mm_iou:.2f}", linewidth=1.5, linestyle="-.")
            else:
                draw_box(ax2, b_pred, (0.3, 0.5, 1.0), label="BL", linewidth=1.5, linestyle="--")
                draw_box(ax2, mm_pred, (1.0, 0.2, 0.4), label="MM", linewidth=1.5, linestyle="-.")

            # Draw trajectory trail (centroid path) on row 2
            trail = [s[:2] for s in hist_states]
            if len(trail) > 1:
                xs, ys = zip(*trail)
                ax2.plot(xs, ys, "-", color=color, linewidth=1, alpha=0.6)
                ax2.plot(xs[-1], ys[-1], "o", color=color, markersize=3)

    # Row labels
    axes[0, 0].set_ylabel("GT Masks", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Predictions", fontsize=10, fontweight="bold")

    vid_id = video_meta["id"]
    fig.suptitle(
        f"Video {vid_id} — Green: GT | Blue dashed: Baseline | Red dash-dot: Multi-Modal",
        fontsize=10, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions on OVIS sequences")
    parser.add_argument("--annotations", type=str, default="data/annotations_train.json")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--baseline-ckpt", type=str, default="baseline_motion_lstm_ovis.pt")
    parser.add_argument("--multimodal-ckpt", type=str, default="multimodal_motion_lstm_ovis.pt")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--num-videos", type=int, default=5, help="Number of videos to visualize")
    parser.add_argument("--max-frames", type=int, default=8, help="Max prediction frames per video")
    parser.add_argument("--out-dir", type=str, default="figs/predictions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load annotations
    ann_path = Path(args.annotations)
    data = json.loads(ann_path.read_text(encoding="utf-8-sig"))
    videos = {v["id"]: v for v in data["videos"]}
    categories = {c["id"]: c["name"] for c in data["categories"]}

    # Group annotations by video
    anns_by_video: Dict[int, list] = {}
    for ann in data["annotations"]:
        vid = ann["video_id"]
        anns_by_video.setdefault(vid, []).append(ann)

    # Load models
    print("Loading baseline checkpoint...")
    b_model, b_xmean, b_xstd, b_ymean, b_ystd = load_baseline_checkpoint(args.baseline_ckpt, device)

    print("Loading multi-modal checkpoint...")
    mm_model, mm_norm, mm_ymean, mm_ystd = load_multimodal_checkpoint(args.multimodal_ckpt, device)

    # Pick videos with multiple instances and some occlusion
    candidates = []
    for vid_id, anns in anns_by_video.items():
        if len(anns) < 2:
            continue
        # Prefer videos with occlusion
        has_occ = any(
            "severe_occlusion" in ann.get("occlusion", [])
            for ann in anns
        )
        candidates.append((vid_id, has_occ, len(anns)))

    # Sort: occluded first, then by instance count
    candidates.sort(key=lambda x: (-int(x[1]), -x[2]))
    selected = candidates[:args.num_videos * 3]
    random.shuffle(selected)
    selected = selected[:args.num_videos]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for vid_id, _, _ in selected:
        video_meta = videos[vid_id]
        anns = anns_by_video[vid_id]
        out_path = out_dir / f"video_{vid_id:04d}_predictions.png"
        print(f"Visualizing video {vid_id} ({len(anns)} instances)...")
        ok = visualize_sequence(
            video_meta, anns, Path(args.data_root),
            args.history,
            b_model, b_xmean, b_xstd, b_ymean, b_ystd,
            mm_model, mm_norm, mm_ymean, mm_ystd,
            device, out_path, args.max_frames,
        )
        if ok:
            print(f"  Saved to {out_path}")
            count += 1

    print(f"\nDone. Generated {count} visualizations in {out_dir}/")


if __name__ == "__main__":
    main()
