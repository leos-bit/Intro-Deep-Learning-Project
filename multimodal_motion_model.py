import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from baseline_motion_model import (
    bbox_iou_xywh,
    bbox_to_state,
    compute_target_norm_stats,
)


# -------------------------------
# Multi-modal feature extraction
# -------------------------------

VEL_DIM = 8
SHAPE_DIM = 8
ACCEL_DIM = 8
CTX_DIM = 4
EMA_ALPHA = 0.3

# Clamp extreme derived features to prevent normalization pollution
_FEAT_CLAMP = 50.0


def build_multimodal_features(
    states: List[Tuple[float, float, float, float]],
) -> List[Tuple[List[float], List[float], List[float], List[float]]]:
    """Build per-frame multi-modal features from center/size history.

    Returns per-frame tuples of:
        (velocity_feats[8], shape_feats[8], accel_feats[8], context_feats[4])
    """
    results: List[Tuple[List[float], List[float], List[float], List[float]]] = []

    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    prev_w: Optional[float] = None
    prev_h: Optional[float] = None
    prev_vx, prev_vy = 0.0, 0.0
    prev_ax, prev_ay = 0.0, 0.0
    vx_ema, vy_ema = 0.0, 0.0
    prev_log_area: Optional[float] = None
    prev_log_aspect: Optional[float] = None

    def _clamp(v: float) -> float:
        return max(-_FEAT_CLAMP, min(_FEAT_CLAMP, v))

    for cx, cy, w, h in states:
        log_area = math.log(w * h)
        log_aspect = math.log(w / h)

        if prev_cx is None:
            vx, vy = 0.0, 0.0
            speed = 0.0
            direction = 0.0
            vx_ema, vy_ema = 0.0, 0.0
            vx_rel, vy_rel = 0.0, 0.0

            dw, dh = 0.0, 0.0
            d_log_area = 0.0
            d_log_aspect = 0.0
            scale_x, scale_y = 1.0, 1.0

            ax, ay = 0.0, 0.0
            a_mag = 0.0
            jerk_x, jerk_y = 0.0, 0.0
            curvature = 0.0
            tangential_a, normal_a = 0.0, 0.0
        else:
            # Velocity
            vx = cx - prev_cx
            vy = cy - prev_cy
            speed = math.sqrt(vx * vx + vy * vy)
            direction = math.atan2(vy, vx) / math.pi
            vx_ema = EMA_ALPHA * vx + (1.0 - EMA_ALPHA) * vx_ema
            vy_ema = EMA_ALPHA * vy + (1.0 - EMA_ALPHA) * vy_ema
            vx_rel = vx / max(w, 1e-6)
            vy_rel = vy / max(h, 1e-6)

            # Shape change
            dw = w - prev_w
            dh = h - prev_h
            d_log_area = log_area - prev_log_area
            d_log_aspect = log_aspect - prev_log_aspect
            scale_x = w / max(prev_w, 1e-6)
            scale_y = h / max(prev_h, 1e-6)

            # Acceleration
            ax = vx - prev_vx
            ay = vy - prev_vy
            a_mag = math.sqrt(ax * ax + ay * ay)
            jerk_x = ax - prev_ax
            jerk_y = ay - prev_ay
            # Use a safe minimum to prevent extreme values when speed is near zero
            speed_safe = max(speed, 1.0)
            speed_cubed = speed_safe ** 3
            curvature = _clamp(abs(vx * ay - vy * ax) / speed_cubed)
            tangential_a = _clamp((vx * ax + vy * ay) / speed_safe)
            normal_a = _clamp((vx * ay - vy * ax) / speed_safe)

        vel_feats = [vx, vy, speed, direction, vx_ema, vy_ema, vx_rel, vy_rel]
        shape_feats = [dw, dh, log_area, d_log_area, log_aspect, d_log_aspect, scale_x, scale_y]
        accel_feats = [ax, ay, a_mag, jerk_x, jerk_y, curvature, tangential_a, normal_a]
        ctx_feats = [cx, cy, w, h]

        results.append((vel_feats, shape_feats, accel_feats, ctx_feats))

        prev_cx, prev_cy = cx, cy
        prev_w, prev_h = w, h
        prev_vx, prev_vy = vx, vy
        prev_ax, prev_ay = ax, ay
        prev_log_area = log_area
        prev_log_aspect = log_aspect

    return results


# -------------------------------
# Dataset
# -------------------------------

@dataclass
class MultiModalMotionSample:
    x_velocity: torch.Tensor   # [history, 8]
    x_shape: torch.Tensor      # [history, 8]
    x_accel: torch.Tensor      # [history, 8]
    x_context: torch.Tensor    # [history, 4]
    y_box: torch.Tensor        # [4]
    y_vis: torch.Tensor        # [1]


class MultiModalMotionDataset(Dataset):
    def __init__(self, samples: List[MultiModalMotionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> MultiModalMotionSample:
        return self.samples[idx]


def parse_ovis_json_multimodal(annotation_file: Path, history: int) -> List[MultiModalMotionSample]:
    data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    annotations = data.get("annotations")
    if annotations is None:
        raise ValueError(
            f"Annotation file has no usable instance labels: {annotation_file}. "
            "This split may contain hidden ground truth."
        )
    samples: List[MultiModalMotionSample] = []

    for ann in annotations:
        bboxes = ann.get("bboxes", [])
        if not isinstance(bboxes, list):
            continue

        states: List[Optional[Tuple[float, float, float, float]]] = []
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

        start = 0
        while start < len(states):
            while start < len(states) and states[start] is None:
                start += 1
            end = start
            while end < len(states) and states[end] is not None:
                end += 1

            segment = states[start:end]
            if len(segment) > history:
                feats = build_multimodal_features(segment)
                for t in range(history, len(feats)):
                    vel_seq = [feats[i][0] for i in range(t - history, t)]
                    shape_seq = [feats[i][1] for i in range(t - history, t)]
                    accel_seq = [feats[i][2] for i in range(t - history, t)]
                    ctx_seq = [feats[i][3] for i in range(t - history, t)]
                    target = segment[t]

                    samples.append(
                        MultiModalMotionSample(
                            x_velocity=torch.tensor(vel_seq, dtype=torch.float32),
                            x_shape=torch.tensor(shape_seq, dtype=torch.float32),
                            x_accel=torch.tensor(accel_seq, dtype=torch.float32),
                            x_context=torch.tensor(ctx_seq, dtype=torch.float32),
                            y_box=torch.tensor(target, dtype=torch.float32),
                            y_vis=torch.tensor([1.0], dtype=torch.float32),
                        )
                    )
            start = end + 1

    if not samples:
        raise ValueError("No valid training samples found in annotation file.")

    return samples


def make_synthetic_samples_multimodal(num_sequences: int, history: int) -> List[MultiModalMotionSample]:
    samples: List[MultiModalMotionSample] = []

    for _ in range(num_sequences):
        cx, cy = random.uniform(100, 500), random.uniform(100, 400)
        vx, vy = random.uniform(-5, 5), random.uniform(-4, 4)
        w, h = random.uniform(30, 120), random.uniform(30, 120)

        raw_states: List[Tuple[float, float, float, float]] = []
        length = history + 1
        for _ in range(length):
            ax_r, ay_r = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)
            vx += ax_r
            vy += ay_r
            cx += vx
            cy += vy
            w = max(10.0, w + random.uniform(-1.2, 1.2))
            h = max(10.0, h + random.uniform(-1.2, 1.2))
            raw_states.append((cx, cy, w, h))

        feats = build_multimodal_features(raw_states)
        vel_seq = [feats[i][0] for i in range(history)]
        shape_seq = [feats[i][1] for i in range(history)]
        accel_seq = [feats[i][2] for i in range(history)]
        ctx_seq = [feats[i][3] for i in range(history)]
        target = raw_states[history]

        samples.append(
            MultiModalMotionSample(
                x_velocity=torch.tensor(vel_seq, dtype=torch.float32),
                x_shape=torch.tensor(shape_seq, dtype=torch.float32),
                x_accel=torch.tensor(accel_seq, dtype=torch.float32),
                x_context=torch.tensor(ctx_seq, dtype=torch.float32),
                y_box=torch.tensor(target, dtype=torch.float32),
                y_vis=torch.tensor([1.0], dtype=torch.float32),
            )
        )

    return samples


def collate_multimodal(
    batch: List[MultiModalMotionSample],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x_vel = torch.stack([s.x_velocity for s in batch], dim=0)
    x_shape = torch.stack([s.x_shape for s in batch], dim=0)
    x_accel = torch.stack([s.x_accel for s in batch], dim=0)
    x_ctx = torch.stack([s.x_context for s in batch], dim=0)
    y_box = torch.stack([s.y_box for s in batch], dim=0)
    y_vis = torch.stack([s.y_vis for s in batch], dim=0)
    return x_vel, x_shape, x_accel, x_ctx, y_box, y_vis


def compute_multimodal_norm_stats(
    dataset: Dataset,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    all_vel, all_shape, all_accel, all_ctx = [], [], [], []
    for i in range(len(dataset)):
        s = dataset[i]
        all_vel.append(s.x_velocity)
        all_shape.append(s.x_shape)
        all_accel.append(s.x_accel)
        all_ctx.append(s.x_context)

    def stats(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        cat = torch.cat(tensors, dim=0)
        return cat.mean(dim=0), cat.std(dim=0).clamp(min=1e-6)

    return {
        "vel": stats(all_vel),
        "shape": stats(all_shape),
        "accel": stats(all_accel),
        "ctx": stats(all_ctx),
    }


# -------------------------------
# Model
# -------------------------------

class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.proj(x))
        out, _ = self.lstm(x)
        return out[:, -1, :]


class GatedFusion(nn.Module):
    """Gate only the motion modalities (velocity, shape, acceleration)."""

    def __init__(self, num_modalities: int, modality_dim: int):
        super().__init__()
        total_dim = num_modalities * modality_dim
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, total_dim),
            nn.ReLU(),
            nn.Linear(total_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, modality_outputs: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat(modality_outputs, dim=-1)
        gates = self.gate_net(concat)                       # [B, num_mod]
        stacked = torch.stack(modality_outputs, dim=1)      # [B, num_mod, dim]
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        return fused, gates


class MultiModalMotionLSTM(nn.Module):
    def __init__(
        self,
        vel_dim: int = VEL_DIM,
        shape_dim: int = SHAPE_DIM,
        accel_dim: int = ACCEL_DIM,
        context_dim: int = CTX_DIM,
        hidden_dim: int = 64,
        num_layers: int = 1,
    ):
        super().__init__()
        self.vel_encoder = ModalityEncoder(vel_dim, hidden_dim, num_layers)
        self.shape_encoder = ModalityEncoder(shape_dim, hidden_dim, num_layers)
        self.accel_encoder = ModalityEncoder(accel_dim, hidden_dim, num_layers)

        # Context gets its own encoder — NOT gated with motion modalities
        self.context_encoder = ModalityEncoder(context_dim, hidden_dim, num_layers)

        # Gate only the 3 motion modalities
        self.fusion = GatedFusion(num_modalities=3, modality_dim=hidden_dim)

        # Combine fused motion (hidden_dim) + context (hidden_dim) -> prediction
        combined_dim = hidden_dim * 2
        self.box_head = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.ReLU(),
            nn.Linear(combined_dim, 4),
        )
        self.vis_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Residual skip: raw last-frame position directly aids prediction
        self.residual_proj = nn.Linear(context_dim, 4)

    def forward(
        self,
        x_vel: torch.Tensor,
        x_shape: torch.Tensor,
        x_accel: torch.Tensor,
        x_context: torch.Tensor,
        return_gates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h_vel = self.vel_encoder(x_vel)
        h_shape = self.shape_encoder(x_shape)
        h_accel = self.accel_encoder(x_accel)
        h_ctx = self.context_encoder(x_context)

        # Gate only motion modalities
        fused_motion, gates = self.fusion([h_vel, h_shape, h_accel])

        # Concatenate fused motion with context — both are always used
        combined = torch.cat([fused_motion, h_ctx], dim=-1)

        pred_box = self.box_head(combined)
        # Residual: add a projection of the raw last-frame state
        pred_box = pred_box + self.residual_proj(x_context[:, -1, :])

        pred_vis_logit = self.vis_head(combined)

        if return_gates:
            return pred_box, pred_vis_logit, gates
        return pred_box, pred_vis_logit, None


# -------------------------------
# Train / eval
# -------------------------------

def train_epoch(
    model: MultiModalMotionLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    gate_entropy_weight: float,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()

    vel_mean, vel_std = norm["vel"]
    shape_mean, shape_std = norm["shape"]
    accel_mean, accel_std = norm["accel"]
    ctx_mean, ctx_std = norm["ctx"]

    for x_vel, x_shape, x_accel, x_ctx, y_box, y_vis in loader:
        x_vel = (x_vel.to(device) - vel_mean) / vel_std
        x_shape = (x_shape.to(device) - shape_mean) / shape_std
        x_accel = (x_accel.to(device) - accel_mean) / accel_std
        x_ctx = (x_ctx.to(device) - ctx_mean) / ctx_std
        y_box = y_box.to(device)
        y_vis = y_vis.to(device)

        y_box_norm = (y_box - y_mean) / y_std

        pred_box_norm, pred_vis_logit, gates = model(
            x_vel, x_shape, x_accel, x_ctx, return_gates=True,
        )
        loss_box = mse_fn(pred_box_norm, y_box_norm)
        loss_vis = bce_fn(pred_vis_logit, y_vis)

        loss = loss_box + 0.1 * loss_vis

        # Light entropy regularization: prevent complete gate collapse
        # but keep weight small so prediction loss dominates
        if gate_entropy_weight > 0.0 and gates is not None:
            gate_entropy = -torch.sum(gates * torch.log(gates + 1e-8), dim=-1).mean()
            loss = loss - gate_entropy_weight * gate_entropy

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * x_vel.size(0)

    return total_loss / len(loader.dataset)


# COCO-style IoU thresholds for mAP: 0.50, 0.55, ..., 0.95
MAP_IOU_THRESHOLDS = torch.arange(0.50, 1.0, 0.05)


@torch.no_grad()
def evaluate(
    model: MultiModalMotionLSTM,
    loader: DataLoader,
    norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> Tuple[float, float, float, float, Dict[str, float], torch.Tensor]:
    """Returns (mse, l2, iou, mAP, per-threshold APs, avg_gates)."""
    model.eval()
    mse_sum = 0.0
    l2_sum = 0.0
    iou_sum = 0.0
    # Count of predictions exceeding each IoU threshold
    threshold_hits = torch.zeros(len(MAP_IOU_THRESHOLDS))
    gate_sum = torch.zeros(3)
    n = 0

    vel_mean, vel_std = norm["vel"]
    shape_mean, shape_std = norm["shape"]
    accel_mean, accel_std = norm["accel"]
    ctx_mean, ctx_std = norm["ctx"]

    for x_vel, x_shape, x_accel, x_ctx, y_box, _y_vis in loader:
        x_vel = (x_vel.to(device) - vel_mean) / vel_std
        x_shape = (x_shape.to(device) - shape_mean) / shape_std
        x_accel = (x_accel.to(device) - accel_mean) / accel_std
        x_ctx = (x_ctx.to(device) - ctx_mean) / ctx_std
        y_box = y_box.to(device)

        pred_box_norm, _, gates = model(
            x_vel, x_shape, x_accel, x_ctx, return_gates=True,
        )
        pred_box = pred_box_norm * y_std + y_mean
        pred_box[:, 2:] = torch.clamp(pred_box[:, 2:], min=1.0)

        mse = torch.mean((pred_box - y_box) ** 2, dim=1)
        l2 = torch.sqrt(torch.sum((pred_box[:, :2] - y_box[:, :2]) ** 2, dim=1))
        iou = bbox_iou_xywh(pred_box, y_box)

        # mAP: count predictions exceeding each IoU threshold
        iou_cpu = iou.cpu()
        for t_idx, thresh in enumerate(MAP_IOU_THRESHOLDS):
            threshold_hits[t_idx] += (iou_cpu >= thresh).sum().item()

        bs = x_vel.size(0)
        mse_sum += mse.sum().item()
        l2_sum += l2.sum().item()
        iou_sum += iou.sum().item()
        gate_sum += gates.sum(dim=0).cpu()
        n += bs

    avg_gates = gate_sum / n
    per_thresh_ap = {f"AP@{MAP_IOU_THRESHOLDS[i]:.2f}": threshold_hits[i].item() / n
                     for i in range(len(MAP_IOU_THRESHOLDS))}
    mAP = (threshold_hits / n).mean().item()

    return mse_sum / n, l2_sum / n, iou_sum / n, mAP, per_thresh_ap, avg_gates


# -------------------------------
# Data loading helpers
# -------------------------------

def load_split_samples_multimodal(
    single_annotations: str,
    train_annotations: str,
    val_annotations: str,
    history: int,
    synthetic_samples: int,
) -> Tuple[List[MultiModalMotionSample], Optional[List[MultiModalMotionSample]]]:
    if train_annotations and val_annotations:
        train_path = Path(train_annotations)
        val_path = Path(val_annotations)
        if not train_path.exists():
            raise FileNotFoundError(f"Train annotations file not found: {train_path}")
        if not val_path.exists():
            raise FileNotFoundError(f"Validation annotations file not found: {val_path}")

        train_samples = parse_ovis_json_multimodal(train_path, history=history)
        val_samples = parse_ovis_json_multimodal(val_path, history=history)
        print(f"Loaded {len(train_samples)} train samples from {train_path}")
        print(f"Loaded {len(val_samples)} val samples from {val_path}")
        return train_samples, val_samples

    if train_annotations or val_annotations:
        raise ValueError("Provide both --train-annotations and --val-annotations together.")

    if single_annotations:
        ann_path = Path(single_annotations)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {ann_path}")
        samples = parse_ovis_json_multimodal(ann_path, history=history)
        print(f"Loaded {len(samples)} samples from {ann_path}")
        return samples, None

    samples = make_synthetic_samples_multimodal(synthetic_samples, history=history)
    print(f"Using synthetic data with {len(samples)} samples")
    return samples, None


# -------------------------------
# Main
# -------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-modal motion model for OVIS-style VIS",
    )
    parser.add_argument("--annotations", type=str, default="", help="Path to OVIS-style JSON annotations")
    parser.add_argument("--train-annotations", type=str, default="", help="Path to training annotations JSON")
    parser.add_argument("--val-annotations", type=str, default="", help="Path to validation annotations JSON")
    parser.add_argument("--history", type=int, default=5, help="Number of past frames used")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthetic-samples", type=int, default=5000)
    parser.add_argument("--hidden-dim", type=int, default=64, help="Per-modality LSTM hidden dim")
    parser.add_argument("--num-layers", type=int, default=1, help="LSTM layers per modality encoder")
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01, help="Gate entropy regularization weight")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm (0 to disable)")
    parser.add_argument("--model-out", type=str, default="multimodal_motion_lstm.pt")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples, val_samples = load_split_samples_multimodal(
        single_annotations=args.annotations,
        train_annotations=args.train_annotations,
        val_annotations=args.val_annotations,
        history=args.history,
        synthetic_samples=args.synthetic_samples,
    )

    if val_samples is not None:
        train_set = MultiModalMotionDataset(samples)
        val_set = MultiModalMotionDataset(val_samples)
        if len(train_set) < 1 or len(val_set) < 1:
            raise ValueError("Need at least 1 sample in both train and validation splits.")
    else:
        dataset = MultiModalMotionDataset(samples)
        n_samples = len(dataset)
        if n_samples < 2:
            raise ValueError("Need at least 2 samples for train/val split.")

        if n_samples < 50:
            print(
                "Warning: dataset is very small. Loss/metrics will be noisy and often large in pixel scale."
            )

        train_len = max(1, int(0.9 * n_samples))
        val_len = n_samples - train_len
        if val_len == 0:
            train_len = n_samples - 1
            val_len = 1

        train_set, val_set = random_split(dataset, [train_len, val_len])

    if len(train_set) < 50:
        print(
            "Warning: training split is very small. Loss/metrics will be noisy and often large in pixel scale."
        )

    # Per-modality normalization stats
    norm_stats = compute_multimodal_norm_stats(train_set)

    # Target normalization (reuse baseline utility via duck-typed dataset access)
    y_mean, y_std = compute_target_norm_stats(train_set)
    y_mean = y_mean.view(1, -1)
    y_std = y_std.view(1, -1)

    # Reshape norm stats for broadcasting: [1, 1, D]
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    norm_on_device: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, (mean, std) in norm_stats.items():
        norm_on_device[key] = (
            mean.view(1, 1, -1).to(device),
            std.view(1, 1, -1).to(device),
        )

    y_mean = y_mean.to(device)
    y_std = y_std.to(device)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multimodal)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multimodal)

    model = MultiModalMotionLSTM(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    best_val_iou = float("-inf")
    best_epoch = None
    out_path = Path(args.model_out)
    last_out_path = out_path.with_name(f"{out_path.stem}_last{out_path.suffix}")

    # Collect metrics for logging
    history_log: List[Dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer,
            norm_on_device, y_mean, y_std, device,
            args.gate_entropy_weight, args.grad_clip,
        )
        val_mse, val_l2, val_iou, val_map, per_thresh_ap, avg_gates = evaluate(
            model, val_loader, norm_on_device, y_mean, y_std, device,
        )

        history_log.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_l2": val_l2,
            "val_iou": val_iou,
            "val_mAP": val_map,
            **per_thresh_ap,
            "gate_vel": avg_gates[0].item(),
            "gate_shape": avg_gates[1].item(),
            "gate_accel": avg_gates[2].item(),
        })

        print(
            f"Epoch {epoch:02d} | train_loss_norm={train_loss:.4f} | "
            f"val_mse_px={val_mse:.4f} | val_center_l2_px={val_l2:.3f} | "
            f"val_iou={val_iou:.3f} | val_mAP={val_map:.3f}"
        )
        print(
            f"  Gates: vel={avg_gates[0]:.3f} shape={avg_gates[1]:.3f} "
            f"accel={avg_gates[2]:.3f}"
        )
        print(
            f"  AP@0.50={per_thresh_ap['AP@0.50']:.3f} "
            f"AP@0.75={per_thresh_ap['AP@0.75']:.3f} "
            f"AP@0.90={per_thresh_ap['AP@0.90']:.3f}"
        )

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_epoch = epoch
            payload = {
                "model_state": model.state_dict(),
                "norm_stats": {
                    k: (m.cpu(), s.cpu()) for k, (m, s) in norm_on_device.items()
                },
                "target_mean": y_mean.cpu(),
                "target_std": y_std.cpu(),
                "history": args.history,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "best_val_iou": best_val_iou,
                "best_epoch": best_epoch,
                "model_type": "multimodal",
            }
            torch.save(payload, out_path)
            print(f"  Saved new best checkpoint at epoch {epoch:02d} to {out_path.resolve()}")

    # Save last-epoch checkpoint
    payload = {
        "model_state": model.state_dict(),
        "norm_stats": {
            k: (m.cpu(), s.cpu()) for k, (m, s) in norm_on_device.items()
        },
        "target_mean": y_mean.cpu(),
        "target_std": y_std.cpu(),
        "history": args.history,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
        "model_type": "multimodal",
    }
    torch.save(payload, last_out_path)
    print(f"Saved last-epoch checkpoint to {last_out_path.resolve()}")

    # Save training log as JSON for plotting
    log_path = out_path.with_suffix(".json")
    log_path.write_text(json.dumps(history_log, indent=2))
    print(f"Saved training log to {log_path.resolve()}")


if __name__ == "__main__":
    main()
