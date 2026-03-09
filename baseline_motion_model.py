import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split


# -------------------------------
# Feature extraction
# -------------------------------

def bbox_to_state(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Convert [x, y, w, h] to (cx, cy, w, h)."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0, max(w, 1e-6), max(h, 1e-6)


def build_motion_features(states: List[Tuple[float, float, float, float]]) -> List[List[float]]:
    """Build per-frame motion features from center/size history."""
    feats: List[List[float]] = []
    prev_vx, prev_vy = 0.0, 0.0
    prev_cx, prev_cy = None, None

    for cx, cy, w, h in states:
        if prev_cx is None:
            vx, vy = 0.0, 0.0
            ax, ay = 0.0, 0.0
        else:
            vx = cx - prev_cx
            vy = cy - prev_cy
            ax = vx - prev_vx
            ay = vy - prev_vy

        area = w * h
        log_aspect = math.log(w / h)
        feats.append([cx, cy, w, h, vx, vy, ax, ay, area, log_aspect])

        prev_cx, prev_cy = cx, cy
        prev_vx, prev_vy = vx, vy

    return feats


def bbox_iou_xywh(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU for [cx, cy, w, h] tensors of shape [N, 4]."""
    ax1 = a[:, 0] - a[:, 2] / 2.0
    ay1 = a[:, 1] - a[:, 3] / 2.0
    ax2 = a[:, 0] + a[:, 2] / 2.0
    ay2 = a[:, 1] + a[:, 3] / 2.0

    bx1 = b[:, 0] - b[:, 2] / 2.0
    by1 = b[:, 1] - b[:, 3] / 2.0
    bx2 = b[:, 0] + b[:, 2] / 2.0
    by2 = b[:, 1] + b[:, 3] / 2.0

    inter_x1 = torch.maximum(ax1, bx1)
    inter_y1 = torch.maximum(ay1, by1)
    inter_x2 = torch.minimum(ax2, bx2)
    inter_y2 = torch.minimum(ay2, by2)

    inter_w = torch.clamp(inter_x2 - inter_x1, min=0.0)
    inter_h = torch.clamp(inter_y2 - inter_y1, min=0.0)
    inter = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = torch.clamp(area_a + area_b - inter, min=1e-6)
    return inter / union


# -------------------------------
# Dataset
# -------------------------------

@dataclass
class MotionSample:
    x: torch.Tensor
    y_box: torch.Tensor
    y_vis: torch.Tensor


class MotionSequenceDataset(Dataset):
    def __init__(self, samples: List[MotionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MotionSample:
        return self.samples[idx]


def parse_ovis_json(annotation_file: Path, history: int) -> List[MotionSample]:
    # Accept both UTF-8 and UTF-8-with-BOM files (common on Windows).
    data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    annotations = data.get("annotations", [])
    samples: List[MotionSample] = []

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

        valid_idx = [i for i, s in enumerate(states) if s is not None]
        if len(valid_idx) <= history:
            continue

        # Build features on contiguous valid segments only.
        start = 0
        while start < len(states):
            while start < len(states) and states[start] is None:
                start += 1
            end = start
            while end < len(states) and states[end] is not None:
                end += 1

            segment = states[start:end]
            if len(segment) > history:
                feats = build_motion_features(segment)
                for t in range(history, len(feats)):
                    seq = feats[t - history : t]
                    target = segment[t]

                    samples.append(
                        MotionSample(
                            x=torch.tensor(seq, dtype=torch.float32),
                            y_box=torch.tensor(target, dtype=torch.float32),
                            y_vis=torch.tensor([1.0], dtype=torch.float32),
                        )
                    )
            start = end + 1

    if not samples:
        raise ValueError("No valid training samples found in annotation file.")

    return samples


def make_synthetic_samples(num_sequences: int, history: int) -> List[MotionSample]:
    samples: List[MotionSample] = []

    for _ in range(num_sequences):
        cx, cy = random.uniform(100, 500), random.uniform(100, 400)
        vx, vy = random.uniform(-5, 5), random.uniform(-4, 4)
        w, h = random.uniform(30, 120), random.uniform(30, 120)

        states = []
        length = history + 1
        for _ in range(length):
            ax, ay = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            vx += ax
            vy += ay
            cx += vx
            cy += vy
            w = max(10.0, w + random.uniform(-1.2, 1.2))
            h = max(10.0, h + random.uniform(-1.2, 1.2))
            states.append((cx, cy, w, h))

        feats = build_motion_features(states)
        seq = feats[:history]
        target = states[history]
        samples.append(
            MotionSample(
                x=torch.tensor(seq, dtype=torch.float32),
                y_box=torch.tensor(target, dtype=torch.float32),
                y_vis=torch.tensor([1.0], dtype=torch.float32),
            )
        )

    return samples


# -------------------------------
# Model
# -------------------------------

class MotionLSTMBaseline(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.box_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.vis_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        pred_box = self.box_head(h)
        pred_vis_logit = self.vis_head(h)
        return pred_box, pred_vis_logit


# -------------------------------
# Train / eval
# -------------------------------

def collate(batch: List[MotionSample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.stack([s.x for s in batch], dim=0)
    y_box = torch.stack([s.y_box for s in batch], dim=0)
    y_vis = torch.stack([s.y_vis for s in batch], dim=0)
    return x, y_box, y_vis


def compute_input_norm_stats(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.cat([dataset[i].x for i in range(len(dataset))], dim=0)
    mean = xs.mean(dim=0)
    std = xs.std(dim=0).clamp(min=1e-6)
    return mean, std


def compute_target_norm_stats(dataset: Dataset) -> Tuple[torch.Tensor, torch.Tensor]:
    ys = torch.stack([dataset[i].y_box for i in range(len(dataset))], dim=0)
    mean = ys.mean(dim=0)
    std = ys.std(dim=0).clamp(min=1e-6)
    return mean, std


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()

    for x, y_box, y_vis in loader:
        x = x.to(device)
        y_box = y_box.to(device)
        y_vis = y_vis.to(device)

        x = (x - x_mean) / x_std
        y_box_norm = (y_box - y_mean) / y_std

        pred_box_norm, pred_vis_logit = model(x)
        loss_box = mse(pred_box_norm, y_box_norm)
        loss_vis = bce(pred_vis_logit, y_vis)
        loss = loss_box + 0.1 * loss_vis

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    mse_sum = 0.0
    l2_sum = 0.0
    iou_sum = 0.0
    n = 0

    for x, y_box, _y_vis in loader:
        x = x.to(device)
        y_box = y_box.to(device)
        x = (x - x_mean) / x_std

        pred_box_norm, _ = model(x)
        pred_box = pred_box_norm * y_std + y_mean
        pred_box[:, 2:] = torch.clamp(pred_box[:, 2:], min=1.0)

        mse = torch.mean((pred_box - y_box) ** 2, dim=1)
        l2 = torch.sqrt(torch.sum((pred_box[:, :2] - y_box[:, :2]) ** 2, dim=1))
        iou = bbox_iou_xywh(pred_box, y_box)

        bs = x.size(0)
        mse_sum += mse.sum().item()
        l2_sum += l2.sum().item()
        iou_sum += iou.sum().item()
        n += bs

    return mse_sum / n, l2_sum / n, iou_sum / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Rudimentary motion baseline for OVIS-style VIS")
    parser.add_argument("--annotations", type=str, default="", help="Path to OVIS-style JSON annotations")
    parser.add_argument("--history", type=int, default=5, help="Number of past frames used")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-samples", type=int, default=5000)
    parser.add_argument("--model-out", type=str, default="baseline_motion_lstm.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.annotations:
        ann_path = Path(args.annotations)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")
        samples = parse_ovis_json(ann_path, history=args.history)
        print(f"Loaded {len(samples)} samples from {ann_path}")
    else:
        samples = make_synthetic_samples(args.synthetic_samples, history=args.history)
        print(f"Using synthetic data with {len(samples)} samples")

    dataset = MotionSequenceDataset(samples)
    n_samples = len(dataset)
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for train/val split.")

    if n_samples < 50:
        print(
            "Warning: dataset is very small. Loss/metrics will be noisy and often large in pixel scale. "
            "Generate a larger rudimentary JSON for more meaningful results."
        )

    train_len = max(1, int(0.9 * n_samples))
    val_len = n_samples - train_len
    if val_len == 0:
        train_len = n_samples - 1
        val_len = 1

    train_set, val_set = random_split(dataset, [train_len, val_len])

    x_mean, x_std = compute_input_norm_stats(train_set)
    y_mean, y_std = compute_target_norm_stats(train_set)

    x_mean = x_mean.view(1, 1, -1)
    x_std = x_std.view(1, 1, -1)
    y_mean = y_mean.view(1, -1)
    y_std = y_std.view(1, -1)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MotionLSTMBaseline(input_dim=10, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    x_mean = x_mean.to(device)
    x_std = x_std.to(device)
    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, x_mean, x_std, y_mean, y_std, device)
        val_mse, val_l2, val_iou = evaluate(model, val_loader, x_mean, x_std, y_mean, y_std, device)
        print(
            f"Epoch {epoch:02d} | train_loss_norm={train_loss:.4f} | "
            f"val_mse_px={val_mse:.4f} | val_center_l2_px={val_l2:.3f} | val_iou={val_iou:.3f}"
        )

    out_path = Path(args.model_out)
    payload = {
        "model_state": model.state_dict(),
        "feature_mean": x_mean.cpu(),
        "feature_std": x_std.cpu(),
        "target_mean": y_mean.cpu(),
        "target_std": y_std.cpu(),
        "history": args.history,
    }
    torch.save(payload, out_path)
    print(f"Saved baseline model to {out_path.resolve()}")


if __name__ == "__main__":
    main()
