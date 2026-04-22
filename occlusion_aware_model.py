"""Occlusion-Aware Multi-Modal Motion LSTM for OVIS.

Phase 2 model that extends the multi-modal gated-fusion LSTM with:
  1. Occlusion-aware temporal attention (re-weights frames by occlusion confidence)
  2. Per-instance adaptive gating (conditions gate weights on motion regime)
  3. Occlusion-aware memory bank for gap prediction
  4. Occlusion classification head

Builds on top of multimodal_motion_model.py (Phase 1).
"""

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from baseline_motion_model import (
    bbox_iou_xywh,
    bbox_to_state,
    compute_target_norm_stats,
)
from multimodal_motion_model import (
    ModalityEncoder,
    build_multimodal_features,
    VEL_DIM,
    SHAPE_DIM,
    ACCEL_DIM,
    CTX_DIM,
    compute_multimodal_norm_stats,
)


# -------------------------------------------------------
# Constants
# -------------------------------------------------------

OCC_DIM = 3  # one-hot: no_occlusion, slight_occlusion, severe_occlusion
AREA_RATIO_DIM = 1
OCC_INPUT_DIM = OCC_DIM + AREA_RATIO_DIM  # 4

MAX_GAP_LENGTH = 50
MAP_IOU_THRESHOLDS = torch.arange(0.50, 1.0, 0.05)

OCC_LABEL_MAP = {
    "no_occlusion": [1.0, 0.0, 0.0],
    "slight_occlusion": [0.0, 1.0, 0.0],
    "severe_occlusion": [0.0, 0.0, 1.0],
}
OCC_DEFAULT = [0.0, 1.0, 0.0]  # default to slight for missing labels


# -------------------------------------------------------
# Data structures
# -------------------------------------------------------

@dataclass
class OcclusionAwareMotionSample:
    x_velocity: torch.Tensor     # [history, 8]
    x_shape: torch.Tensor        # [history, 8]
    x_accel: torch.Tensor        # [history, 8]
    x_context: torch.Tensor      # [history, 4]
    x_occlusion: torch.Tensor    # [history, 3]
    x_area_ratio: torch.Tensor   # [history, 1]
    y_box: torch.Tensor          # [4]
    y_vis: torch.Tensor          # [1]
    y_occlusion: torch.Tensor    # [3]
    gap_length: int              # 0 for standard, 1-50 for gap samples
    memory_repr: Optional[torch.Tensor] = None  # [128] for gap samples


class OcclusionAwareMotionDataset(Dataset):
    def __init__(self, samples: List[OcclusionAwareMotionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> OcclusionAwareMotionSample:
        return self.samples[idx]


# -------------------------------------------------------
# Data parsing
# -------------------------------------------------------

def _encode_occlusion(label: Optional[str]) -> List[float]:
    if label is None:
        return OCC_DEFAULT
    return OCC_LABEL_MAP.get(label, OCC_DEFAULT)


def parse_ovis_json_occlusion_aware(
    annotation_file: Path, history: int,
) -> List[OcclusionAwareMotionSample]:
    """Parse OVIS annotations including occlusion labels and area ratios."""
    data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    annotations = data.get("annotations")
    if annotations is None:
        raise ValueError(f"No annotations in {annotation_file}")

    samples: List[OcclusionAwareMotionSample] = []

    for ann in annotations:
        bboxes = ann.get("bboxes", [])
        occ_labels = ann.get("occlusion", [None] * len(bboxes))
        areas_raw = ann.get("areas", [None] * len(bboxes))
        if not isinstance(bboxes, list):
            continue

        # Build states and metadata
        states: List[Optional[Tuple[float, float, float, float]]] = []
        occ_per_frame: List[Optional[str]] = []
        area_per_frame: List[Optional[float]] = []

        for i, bbox in enumerate(bboxes):
            occ_label = occ_labels[i] if i < len(occ_labels) else None
            area = areas_raw[i] if i < len(areas_raw) else None

            if bbox is None:
                states.append(None)
                occ_per_frame.append(None)
                area_per_frame.append(None)
            elif isinstance(bbox, list) and len(bbox) == 4:
                states.append(bbox_to_state(bbox))
                occ_per_frame.append(occ_label)
                if area is not None:
                    area_per_frame.append(float(area))
                else:
                    x, y, w, h = bbox
                    area_per_frame.append(float(w * h))
            else:
                states.append(None)
                occ_per_frame.append(None)
                area_per_frame.append(None)

        # Extract samples from contiguous visible segments
        start = 0
        while start < len(states):
            while start < len(states) and states[start] is None:
                start += 1
            end = start
            while end < len(states) and states[end] is not None:
                end += 1

            segment_states = states[start:end]
            segment_occ = occ_per_frame[start:end]
            segment_area = area_per_frame[start:end]

            if len(segment_states) > history:
                feats = build_multimodal_features(segment_states)

                # Compute area ratios for segment
                area_ratios = [1.0]  # first frame has ratio 1.0
                for j in range(1, len(segment_area)):
                    prev_a = segment_area[j - 1]
                    cur_a = segment_area[j]
                    if prev_a and cur_a and prev_a > 0:
                        ratio = max(0.5, min(2.0, cur_a / prev_a))
                    else:
                        ratio = 1.0
                    area_ratios.append(ratio)

                for t in range(history, len(feats)):
                    vel_seq = [feats[i][0] for i in range(t - history, t)]
                    shape_seq = [feats[i][1] for i in range(t - history, t)]
                    accel_seq = [feats[i][2] for i in range(t - history, t)]
                    ctx_seq = [feats[i][3] for i in range(t - history, t)]

                    occ_seq = [_encode_occlusion(segment_occ[i]) for i in range(t - history, t)]
                    ar_seq = [[area_ratios[i]] for i in range(t - history, t)]

                    target = segment_states[t]
                    target_occ = _encode_occlusion(segment_occ[t])

                    samples.append(
                        OcclusionAwareMotionSample(
                            x_velocity=torch.tensor(vel_seq, dtype=torch.float32),
                            x_shape=torch.tensor(shape_seq, dtype=torch.float32),
                            x_accel=torch.tensor(accel_seq, dtype=torch.float32),
                            x_context=torch.tensor(ctx_seq, dtype=torch.float32),
                            x_occlusion=torch.tensor(occ_seq, dtype=torch.float32),
                            x_area_ratio=torch.tensor(ar_seq, dtype=torch.float32),
                            y_box=torch.tensor(target, dtype=torch.float32),
                            y_vis=torch.tensor([1.0], dtype=torch.float32),
                            y_occlusion=torch.tensor(target_occ, dtype=torch.float32),
                            gap_length=0,
                        )
                    )

            start = end + 1

    if not samples:
        raise ValueError("No valid training samples found.")
    return samples


def parse_ovis_json_gap_samples(
    annotation_file: Path,
    history: int,
    min_gap: int = 1,
    max_gap: int = MAX_GAP_LENGTH,
) -> List[OcclusionAwareMotionSample]:
    """Extract gap training samples from visible->occluded->visible patterns."""
    data = json.loads(annotation_file.read_text(encoding="utf-8-sig"))
    annotations = data.get("annotations")
    if annotations is None:
        raise ValueError(f"No annotations in {annotation_file}")

    samples: List[OcclusionAwareMotionSample] = []

    for ann in annotations:
        bboxes = ann.get("bboxes", [])
        occ_labels = ann.get("occlusion", [None] * len(bboxes))
        areas_raw = ann.get("areas", [None] * len(bboxes))
        if not isinstance(bboxes, list):
            continue

        states: List[Optional[Tuple[float, float, float, float]]] = []
        occ_per_frame: List[Optional[str]] = []
        area_per_frame: List[Optional[float]] = []

        for i, bbox in enumerate(bboxes):
            occ_label = occ_labels[i] if i < len(occ_labels) else None
            area = areas_raw[i] if i < len(areas_raw) else None

            if bbox is None:
                states.append(None)
                occ_per_frame.append(None)
                area_per_frame.append(None)
            elif isinstance(bbox, list) and len(bbox) == 4:
                states.append(bbox_to_state(bbox))
                occ_per_frame.append(occ_label)
                if area is not None:
                    area_per_frame.append(float(area))
                else:
                    x, y, w, h = bbox
                    area_per_frame.append(float(w * h))
            else:
                states.append(None)
                occ_per_frame.append(None)
                area_per_frame.append(None)

        # Find visible -> gap -> visible patterns
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

            if pre_len < history:
                continue

            # Count gap
            gap_start = i
            while i < len(states) and states[i] is None:
                i += 1
            gap_len = i - gap_start

            if gap_len < min_gap or gap_len > max_gap:
                continue

            # Check for visible frame after gap
            if i >= len(states) or states[i] is None:
                continue

            # Build sample from the last `history` visible frames before the gap
            seg_start = max(vis_start, vis_end - history)
            segment_states = [states[t] for t in range(seg_start, vis_end)]
            segment_occ = [occ_per_frame[t] for t in range(seg_start, vis_end)]
            segment_area = [area_per_frame[t] for t in range(seg_start, vis_end)]

            if len(segment_states) < history:
                continue

            feats = build_multimodal_features(segment_states)
            if len(feats) < history:
                continue

            # Use the last `history` frames
            h = history
            vel_seq = [feats[j][0] for j in range(len(feats) - h, len(feats))]
            shape_seq = [feats[j][1] for j in range(len(feats) - h, len(feats))]
            accel_seq = [feats[j][2] for j in range(len(feats) - h, len(feats))]
            ctx_seq = [feats[j][3] for j in range(len(feats) - h, len(feats))]

            occ_seq = [_encode_occlusion(segment_occ[j]) for j in range(len(feats) - h, len(feats))]

            area_ratios = [1.0]
            for j in range(1, len(segment_area)):
                prev_a = segment_area[j - 1]
                cur_a = segment_area[j]
                if prev_a and cur_a and prev_a > 0:
                    ratio = max(0.5, min(2.0, cur_a / prev_a))
                else:
                    ratio = 1.0
                area_ratios.append(ratio)
            ar_seq = [[area_ratios[j]] for j in range(len(feats) - h, len(feats))]

            target = states[i]
            target_occ = _encode_occlusion(occ_per_frame[i] if i < len(occ_per_frame) else None)

            samples.append(
                OcclusionAwareMotionSample(
                    x_velocity=torch.tensor(vel_seq, dtype=torch.float32),
                    x_shape=torch.tensor(shape_seq, dtype=torch.float32),
                    x_accel=torch.tensor(accel_seq, dtype=torch.float32),
                    x_context=torch.tensor(ctx_seq, dtype=torch.float32),
                    x_occlusion=torch.tensor(occ_seq, dtype=torch.float32),
                    x_area_ratio=torch.tensor(ar_seq, dtype=torch.float32),
                    y_box=torch.tensor(target, dtype=torch.float32),
                    y_vis=torch.tensor([1.0], dtype=torch.float32),
                    y_occlusion=torch.tensor(target_occ, dtype=torch.float32),
                    gap_length=gap_len,
                )
            )

    return samples


# -------------------------------------------------------
# Collate
# -------------------------------------------------------

def collate_occlusion_aware(
    batch: List[OcclusionAwareMotionSample],
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor,
    torch.Tensor, torch.Tensor, torch.Tensor,
    torch.Tensor,
]:
    x_vel = torch.stack([s.x_velocity for s in batch])
    x_shape = torch.stack([s.x_shape for s in batch])
    x_accel = torch.stack([s.x_accel for s in batch])
    x_ctx = torch.stack([s.x_context for s in batch])
    x_occ = torch.stack([s.x_occlusion for s in batch])
    x_ar = torch.stack([s.x_area_ratio for s in batch])
    y_box = torch.stack([s.y_box for s in batch])
    y_vis = torch.stack([s.y_vis for s in batch])
    y_occ = torch.stack([s.y_occlusion for s in batch])
    gap_lengths = torch.tensor([s.gap_length for s in batch], dtype=torch.long)
    return x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar, y_box, y_vis, y_occ, gap_lengths


# -------------------------------------------------------
# New modules
# -------------------------------------------------------

class OcclusionConfidenceModule(nn.Module):
    """Maps per-frame occlusion signals to a scalar confidence in [0, 1]."""

    def __init__(self, input_dim: int = OCC_INPUT_DIM, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, input_dim] -> [B, T, 1]"""
        return self.net(x)


class OcclusionAwareTemporalAttention(nn.Module):
    """Scaled dot-product attention with occlusion confidence modulation."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(
        self,
        h_seq: torch.Tensor,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_seq: [B, T, hidden_dim]
        confidence: [B, T, 1]
        Returns: attended [B, hidden_dim], attn_weights [B, T]
        """
        query = self.query_proj(h_seq[:, -1, :]).unsqueeze(1)  # [B, 1, D]
        keys = self.key_proj(h_seq)                             # [B, T, D]
        values = self.value_proj(h_seq)                         # [B, T, D]

        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # [B, 1, T]

        # Occlusion modulation: low-confidence frames get suppressed
        log_conf = torch.log(confidence.transpose(1, 2) + 1e-8)  # [B, 1, T]
        scores = scores + log_conf

        attn_weights = F.softmax(scores, dim=-1)  # [B, 1, T]
        attended = torch.bmm(attn_weights, values).squeeze(1)  # [B, D]

        return attended, attn_weights.squeeze(1)  # [B, D], [B, T]


class AdaptiveGatedFusion(nn.Module):
    """Gated fusion conditioned on motion regime summary."""

    def __init__(self, num_modalities: int, modality_dim: int, summary_dim: int = 3):
        super().__init__()
        total_dim = num_modalities * modality_dim + summary_dim
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, num_modalities * modality_dim),
            nn.ReLU(),
            nn.Linear(num_modalities * modality_dim, num_modalities),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        modality_outputs: List[torch.Tensor],
        motion_summary: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        concat = torch.cat(modality_outputs + [motion_summary], dim=-1)
        gates = self.gate_net(concat)  # [B, num_mod]
        stacked = torch.stack(modality_outputs, dim=1)  # [B, num_mod, dim]
        fused = (stacked * gates.unsqueeze(-1)).sum(dim=1)  # [B, dim]
        return fused, gates


class MemoryReadout(nn.Module):
    """Differentiable memory readout with gap-length embedding."""

    def __init__(self, repr_dim: int, gap_embed_dim: int = 16, max_gap: int = MAX_GAP_LENGTH + 1):
        super().__init__()
        self.gap_embedding = nn.Embedding(max_gap, gap_embed_dim)
        combined_dim = repr_dim * 2 + gap_embed_dim
        self.gate = nn.Sequential(
            nn.Linear(combined_dim, repr_dim),
            nn.ReLU(),
            nn.Linear(repr_dim, 1),
            nn.Sigmoid(),
        )
        self.memory_proj = nn.Linear(repr_dim, repr_dim)

    def forward(
        self,
        current_repr: torch.Tensor,
        memory_repr: torch.Tensor,
        gap_length: torch.Tensor,
    ) -> torch.Tensor:
        """
        current_repr: [B, repr_dim]
        memory_repr: [B, repr_dim]
        gap_length: [B] (long tensor)
        Returns: [B, repr_dim]
        """
        gap_emb = self.gap_embedding(gap_length.clamp(0, MAX_GAP_LENGTH))  # [B, gap_embed_dim]
        combined = torch.cat([current_repr, memory_repr, gap_emb], dim=-1)
        g = self.gate(combined)  # [B, 1]
        mem_proj = self.memory_proj(memory_repr)
        return g * mem_proj + (1 - g) * current_repr


# -------------------------------------------------------
# Memory Bank (inference-time data structure)
# -------------------------------------------------------

@dataclass
class InstanceMemory:
    instance_id: int
    motion_repr: torch.Tensor
    last_visible_box: Tuple[float, float, float, float]
    last_velocity: Tuple[float, float]
    frames_since_visible: int
    confidence: float
    history_buffer: List[Tuple[float, float, float, float]]


class MemoryBank:
    def __init__(self, history_capacity: int = 10):
        self.entries: Dict[int, InstanceMemory] = {}
        self.history_capacity = history_capacity

    def update(
        self,
        instance_id: int,
        motion_repr: torch.Tensor,
        box: Tuple[float, float, float, float],
        velocity: Tuple[float, float],
        confidence: float,
    ):
        if instance_id in self.entries:
            entry = self.entries[instance_id]
            entry.motion_repr = motion_repr.detach()
            entry.last_visible_box = box
            entry.last_velocity = velocity
            entry.frames_since_visible = 0
            entry.confidence = confidence
            entry.history_buffer.append(box)
            if len(entry.history_buffer) > self.history_capacity:
                entry.history_buffer.pop(0)
        else:
            self.entries[instance_id] = InstanceMemory(
                instance_id=instance_id,
                motion_repr=motion_repr.detach(),
                last_visible_box=box,
                last_velocity=velocity,
                frames_since_visible=0,
                confidence=confidence,
                history_buffer=[box],
            )

    def mark_occluded(self, instance_id: int):
        if instance_id in self.entries:
            self.entries[instance_id].frames_since_visible += 1

    def query(self, instance_id: int) -> Optional[InstanceMemory]:
        return self.entries.get(instance_id)

    def clear(self):
        self.entries.clear()


# -------------------------------------------------------
# Integrated model
# -------------------------------------------------------

class OcclusionAwareMultiModalLSTM(nn.Module):
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
        self.hidden_dim = hidden_dim

        # Modality encoders (can be initialized from Phase 1 checkpoint)
        self.vel_encoder = ModalityEncoder(vel_dim, hidden_dim, num_layers)
        self.shape_encoder = ModalityEncoder(shape_dim, hidden_dim, num_layers)
        self.accel_encoder = ModalityEncoder(accel_dim, hidden_dim, num_layers)
        self.context_encoder = ModalityEncoder(context_dim, hidden_dim, num_layers)

        # Occlusion confidence
        self.occlusion_confidence = OcclusionConfidenceModule()

        # Temporal attention per modality
        self.temporal_attn_vel = OcclusionAwareTemporalAttention(hidden_dim)
        self.temporal_attn_shape = OcclusionAwareTemporalAttention(hidden_dim)
        self.temporal_attn_accel = OcclusionAwareTemporalAttention(hidden_dim)
        self.temporal_attn_ctx = OcclusionAwareTemporalAttention(hidden_dim)

        # Adaptive gated fusion
        self.fusion = AdaptiveGatedFusion(num_modalities=3, modality_dim=hidden_dim, summary_dim=3)

        # Memory readout
        combined_dim = hidden_dim * 2
        self.memory_readout = MemoryReadout(repr_dim=combined_dim)

        # Output heads
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
        self.occlusion_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

        # Residual skip
        self.residual_proj = nn.Linear(context_dim, 4)

    def forward(
        self,
        x_vel: torch.Tensor,
        x_shape: torch.Tensor,
        x_accel: torch.Tensor,
        x_context: torch.Tensor,
        x_occlusion: torch.Tensor,
        x_area_ratio: torch.Tensor,
        gap_lengths: Optional[torch.Tensor] = None,
        memory_repr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # 1. Encode modalities (all timesteps)
        h_vel_seq = self.vel_encoder(x_vel, return_all_timesteps=True)
        h_shape_seq = self.shape_encoder(x_shape, return_all_timesteps=True)
        h_accel_seq = self.accel_encoder(x_accel, return_all_timesteps=True)
        h_ctx_seq = self.context_encoder(x_context, return_all_timesteps=True)

        # 2. Compute occlusion confidence
        occ_input = torch.cat([x_occlusion, x_area_ratio], dim=-1)  # [B, T, 4]
        confidence = self.occlusion_confidence(occ_input)  # [B, T, 1]

        # 3. Temporal attention per modality
        h_vel, attn_vel = self.temporal_attn_vel(h_vel_seq, confidence)
        h_shape, attn_shape = self.temporal_attn_shape(h_shape_seq, confidence)
        h_accel, attn_accel = self.temporal_attn_accel(h_accel_seq, confidence)
        h_ctx, attn_ctx = self.temporal_attn_ctx(h_ctx_seq, confidence)

        # 4. Motion regime summary for adaptive gating
        mean_speed = x_vel[:, :, 2].mean(dim=1, keepdim=True)  # speed feature
        max_accel = x_accel[:, :, 2].max(dim=1, keepdim=True).values  # accel magnitude
        mean_conf = confidence.squeeze(-1).mean(dim=1, keepdim=True)
        motion_summary = torch.cat([mean_speed, max_accel, mean_conf], dim=1)  # [B, 3]

        # 5. Adaptive gated fusion
        fused_motion, gates = self.fusion([h_vel, h_shape, h_accel], motion_summary)

        # 6. Combine with context
        combined = torch.cat([fused_motion, h_ctx], dim=-1)  # [B, 2*hidden_dim]

        # 7. Memory readout for gap samples
        if gap_lengths is not None and memory_repr is not None:
            has_gap = gap_lengths > 0
            if has_gap.any():
                combined = torch.where(
                    has_gap.unsqueeze(-1),
                    self.memory_readout(combined, memory_repr, gap_lengths),
                    combined,
                )

        # 8. Predict
        pred_box = self.box_head(combined) + self.residual_proj(x_context[:, -1, :])
        pred_vis = self.vis_head(combined)
        pred_occ = self.occlusion_head(combined)

        return {
            "pred_box": pred_box,
            "pred_vis": pred_vis,
            "pred_occ": pred_occ,
            "gates": gates,
            "attn_vel": attn_vel,
            "attn_shape": attn_shape,
            "attn_accel": attn_accel,
            "attn_ctx": attn_ctx,
            "confidence": confidence,
            "combined": combined.detach(),
        }

    def get_combined_repr(
        self,
        x_vel: torch.Tensor,
        x_shape: torch.Tensor,
        x_accel: torch.Tensor,
        x_context: torch.Tensor,
        x_occlusion: torch.Tensor,
        x_area_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """Get the fused representation without prediction heads (for memory storage)."""
        with torch.no_grad():
            h_vel_seq = self.vel_encoder(x_vel, return_all_timesteps=True)
            h_shape_seq = self.shape_encoder(x_shape, return_all_timesteps=True)
            h_accel_seq = self.accel_encoder(x_accel, return_all_timesteps=True)
            h_ctx_seq = self.context_encoder(x_context, return_all_timesteps=True)

            occ_input = torch.cat([x_occlusion, x_area_ratio], dim=-1)
            confidence = self.occlusion_confidence(occ_input)

            h_vel, _ = self.temporal_attn_vel(h_vel_seq, confidence)
            h_shape, _ = self.temporal_attn_shape(h_shape_seq, confidence)
            h_accel, _ = self.temporal_attn_accel(h_accel_seq, confidence)
            h_ctx, _ = self.temporal_attn_ctx(h_ctx_seq, confidence)

            mean_speed = x_vel[:, :, 2].mean(dim=1, keepdim=True)
            max_accel = x_accel[:, :, 2].max(dim=1, keepdim=True).values
            mean_conf = confidence.squeeze(-1).mean(dim=1, keepdim=True)
            motion_summary = torch.cat([mean_speed, max_accel, mean_conf], dim=1)

            fused_motion, _ = self.fusion([h_vel, h_shape, h_accel], motion_summary)
            combined = torch.cat([fused_motion, h_ctx], dim=-1)
            return combined


def load_phase1_weights(model: OcclusionAwareMultiModalLSTM, checkpoint_path: Path, device: torch.device):
    """Load encoder weights from Phase 1 multimodal checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    phase1_state = ckpt["model_state"]

    encoder_prefixes = ["vel_encoder.", "shape_encoder.", "accel_encoder.", "context_encoder."]
    loaded = 0
    for key, value in phase1_state.items():
        for prefix in encoder_prefixes:
            if key.startswith(prefix):
                if key in model.state_dict():
                    model.state_dict()[key].copy_(value)
                    loaded += 1
                break

    # Also load residual_proj if available
    if "residual_proj.weight" in phase1_state:
        model.residual_proj.weight.data.copy_(phase1_state["residual_proj.weight"])
        model.residual_proj.bias.data.copy_(phase1_state["residual_proj.bias"])
        loaded += 2

    print(f"Loaded {loaded} parameter tensors from Phase 1 checkpoint")
    return ckpt.get("norm_stats"), ckpt.get("target_mean"), ckpt.get("target_std")


# -------------------------------------------------------
# Training
# -------------------------------------------------------

def train_epoch_phase2(
    model: OcclusionAwareMultiModalLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    gate_entropy_weight: float = 0.01,
    attn_entropy_weight: float = 0.001,
    occ_loss_weight: float = 0.05,
    grad_clip: float = 1.0,
    gap_loader: Optional[DataLoader] = None,
    max_gap_curriculum: int = MAX_GAP_LENGTH,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_box_loss = 0.0
    total_gap_loss = 0.0
    n_samples = 0
    n_gap_samples = 0

    mse_fn = nn.MSELoss()
    bce_fn = nn.BCEWithLogitsLoss()
    ce_fn = nn.CrossEntropyLoss()

    vel_mean, vel_std = norm["vel"]
    shape_mean, shape_std = norm["shape"]
    accel_mean, accel_std = norm["accel"]
    ctx_mean, ctx_std = norm["ctx"]

    # Standard samples
    for batch in loader:
        x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar, y_box, y_vis, y_occ, gap_lengths = batch

        x_vel = (x_vel.to(device) - vel_mean) / vel_std
        x_shape = (x_shape.to(device) - shape_mean) / shape_std
        x_accel = (x_accel.to(device) - accel_mean) / accel_std
        x_ctx = (x_ctx.to(device) - ctx_mean) / ctx_std
        x_occ = x_occ.to(device)
        x_ar = x_ar.to(device)
        y_box = y_box.to(device)
        y_vis = y_vis.to(device)
        y_occ = y_occ.to(device)

        y_box_norm = (y_box - y_mean) / y_std

        out = model(x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar)

        loss_box = mse_fn(out["pred_box"], y_box_norm)
        loss_vis = bce_fn(out["pred_vis"], y_vis)
        loss_occ = ce_fn(out["pred_occ"], y_occ.argmax(dim=-1))

        loss = loss_box + 0.1 * loss_vis + occ_loss_weight * loss_occ

        # Gate entropy regularization
        if gate_entropy_weight > 0.0:
            gate_entropy = -torch.sum(out["gates"] * torch.log(out["gates"] + 1e-8), dim=-1).mean()
            loss = loss - gate_entropy_weight * gate_entropy

        # Attention entropy regularization
        if attn_entropy_weight > 0.0:
            attn_entropy = 0.0
            for attn_key in ["attn_vel", "attn_shape", "attn_accel", "attn_ctx"]:
                w = out[attn_key]
                attn_entropy += -torch.sum(w * torch.log(w + 1e-8), dim=-1).mean()
            attn_entropy /= 4.0
            loss = loss - attn_entropy_weight * attn_entropy

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        bs = x_vel.size(0)
        total_loss += loss.item() * bs
        total_box_loss += loss_box.item() * bs
        n_samples += bs

    # Gap samples (Phase 2b)
    if gap_loader is not None:
        for batch in gap_loader:
            x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar, y_box, y_vis, y_occ, gap_lengths = batch

            # Filter by curriculum
            mask = gap_lengths <= max_gap_curriculum
            if not mask.any():
                continue

            x_vel = (x_vel[mask].to(device) - vel_mean) / vel_std
            x_shape = (x_shape[mask].to(device) - shape_mean) / shape_std
            x_accel = (x_accel[mask].to(device) - accel_mean) / accel_std
            x_ctx = (x_ctx[mask].to(device) - ctx_mean) / ctx_std
            x_occ = x_occ[mask].to(device)
            x_ar = x_ar[mask].to(device)
            y_box = y_box[mask].to(device)
            gap_lengths_filtered = gap_lengths[mask].to(device)

            y_box_norm = (y_box - y_mean) / y_std

            # Compute memory representation from pre-gap frames
            memory_repr = model.get_combined_repr(x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar)

            out = model(
                x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar,
                gap_lengths=gap_lengths_filtered,
                memory_repr=memory_repr,
            )

            loss_gap = mse_fn(out["pred_box"], y_box_norm)
            # Upweight longer gaps
            gap_weight = torch.log(gap_lengths_filtered.float() + 1).mean()
            loss = loss_gap * (1.0 + 0.5 * gap_weight)

            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            bs = x_vel.size(0)
            total_gap_loss += loss.item() * bs
            n_gap_samples += bs

    metrics = {
        "train_loss": total_loss / max(n_samples, 1),
        "train_box_loss": total_box_loss / max(n_samples, 1),
    }
    if n_gap_samples > 0:
        metrics["train_gap_loss"] = total_gap_loss / n_gap_samples
        metrics["n_gap_samples"] = n_gap_samples
    return metrics


# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------

@torch.no_grad()
def evaluate_phase2(
    model: OcclusionAwareMultiModalLSTM,
    loader: DataLoader,
    norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    mse_sum = 0.0
    l2_sum = 0.0
    iou_sum = 0.0
    threshold_hits = torch.zeros(len(MAP_IOU_THRESHOLDS))
    gate_sum = torch.zeros(3)
    attn_entropy_sum = 0.0
    n = 0

    vel_mean, vel_std = norm["vel"]
    shape_mean, shape_std = norm["shape"]
    accel_mean, accel_std = norm["accel"]
    ctx_mean, ctx_std = norm["ctx"]

    for batch in loader:
        x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar, y_box, y_vis, y_occ, gap_lengths = batch

        x_vel = (x_vel.to(device) - vel_mean) / vel_std
        x_shape = (x_shape.to(device) - shape_mean) / shape_std
        x_accel = (x_accel.to(device) - accel_mean) / accel_std
        x_ctx = (x_ctx.to(device) - ctx_mean) / ctx_std
        x_occ = x_occ.to(device)
        x_ar = x_ar.to(device)
        y_box = y_box.to(device)

        out = model(x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar)

        pred_box = out["pred_box"] * y_std + y_mean
        pred_box[:, 2:] = torch.clamp(pred_box[:, 2:], min=1.0)

        mse = torch.mean((pred_box - y_box) ** 2, dim=1)
        l2 = torch.sqrt(torch.sum((pred_box[:, :2] - y_box[:, :2]) ** 2, dim=1))
        iou = bbox_iou_xywh(pred_box, y_box)

        iou_cpu = iou.cpu()
        for t_idx, thresh in enumerate(MAP_IOU_THRESHOLDS):
            threshold_hits[t_idx] += (iou_cpu >= thresh).sum().item()

        bs = x_vel.size(0)
        mse_sum += mse.sum().item()
        l2_sum += l2.sum().item()
        iou_sum += iou.sum().item()
        gate_sum += out["gates"].sum(dim=0).cpu()

        # Track attention entropy
        for attn_key in ["attn_vel", "attn_shape", "attn_accel", "attn_ctx"]:
            w = out[attn_key]
            attn_entropy_sum += -torch.sum(w * torch.log(w + 1e-8), dim=-1).sum().item()
        n += bs

    per_thresh_ap = {
        f"AP@{MAP_IOU_THRESHOLDS[i]:.2f}": threshold_hits[i].item() / n
        for i in range(len(MAP_IOU_THRESHOLDS))
    }
    mAP = (threshold_hits / n).mean().item()
    avg_gates = gate_sum / n

    return {
        "val_mse": mse_sum / n,
        "val_l2": l2_sum / n,
        "val_iou": iou_sum / n,
        "val_mAP": mAP,
        **per_thresh_ap,
        "gate_vel": avg_gates[0].item(),
        "gate_shape": avg_gates[1].item(),
        "gate_accel": avg_gates[2].item(),
        "attn_entropy": attn_entropy_sum / (4 * n),
    }


@torch.no_grad()
def evaluate_gap_prediction(
    model: OcclusionAwareMultiModalLSTM,
    gap_samples: List[OcclusionAwareMotionSample],
    norm: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate gap prediction performance grouped by gap duration."""
    model.eval()

    vel_mean_t, vel_std_t = norm["vel"]
    shape_mean_t, shape_std_t = norm["shape"]
    accel_mean_t, accel_std_t = norm["accel"]
    ctx_mean_t, ctx_std_t = norm["ctx"]

    results_by_bucket = {}
    buckets = [(1, 3), (4, 6), (7, 10), (11, MAX_GAP_LENGTH)]

    for bmin, bmax in buckets:
        bucket_samples = [s for s in gap_samples if bmin <= s.gap_length <= bmax]
        if not bucket_samples:
            results_by_bucket[f"gap_{bmin}_{bmax}"] = {"n": 0, "iou": 0.0, "mAP": 0.0}
            continue

        iou_sum = 0.0
        threshold_hits = torch.zeros(len(MAP_IOU_THRESHOLDS))
        n = 0

        # Process in batches
        batch_size = 128
        for start in range(0, len(bucket_samples), batch_size):
            batch = bucket_samples[start:start + batch_size]
            collated = collate_occlusion_aware(batch)
            x_vel, x_shape, x_accel, x_ctx, x_occ, x_ar, y_box, y_vis, y_occ, gap_lengths = collated

            x_vel_n = (x_vel.to(device) - vel_mean_t) / vel_std_t
            x_shape_n = (x_shape.to(device) - shape_mean_t) / shape_std_t
            x_accel_n = (x_accel.to(device) - accel_mean_t) / accel_std_t
            x_ctx_n = (x_ctx.to(device) - ctx_mean_t) / ctx_std_t
            x_occ = x_occ.to(device)
            x_ar = x_ar.to(device)
            y_box = y_box.to(device)
            gap_lengths = gap_lengths.to(device)

            memory_repr = model.get_combined_repr(x_vel_n, x_shape_n, x_accel_n, x_ctx_n, x_occ, x_ar)
            out = model(
                x_vel_n, x_shape_n, x_accel_n, x_ctx_n, x_occ, x_ar,
                gap_lengths=gap_lengths,
                memory_repr=memory_repr,
            )

            pred_box = out["pred_box"] * y_std + y_mean
            pred_box[:, 2:] = torch.clamp(pred_box[:, 2:], min=1.0)

            iou = bbox_iou_xywh(pred_box, y_box)
            iou_cpu = iou.cpu()
            iou_sum += iou.sum().item()
            for t_idx, thresh in enumerate(MAP_IOU_THRESHOLDS):
                threshold_hits[t_idx] += (iou_cpu >= thresh).sum().item()
            n += len(batch)

        mAP = (threshold_hits / n).mean().item() if n > 0 else 0.0
        results_by_bucket[f"gap_{bmin}_{bmax}"] = {
            "n": n,
            "iou": iou_sum / n if n > 0 else 0.0,
            "mAP": mAP,
        }

    return results_by_bucket


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Occlusion-Aware Multi-Modal Motion LSTM")
    parser.add_argument("--annotations", type=str, default="",
                        help="Single annotation file (auto-split 90/10)")
    parser.add_argument("--train-annotations", type=str, default="")
    parser.add_argument("--val-annotations", type=str, default="")
    parser.add_argument("--phase1-checkpoint", type=str, default="multimodal_motion_lstm_ovis.pt",
                        help="Path to Phase 1 multimodal checkpoint for weight initialization")
    parser.add_argument("--history", type=int, default=5)
    parser.add_argument("--epochs-2a", type=int, default=20, help="Epochs for Phase 2a (temporal attention)")
    parser.add_argument("--epochs-2b", type=int, default=15, help="Epochs for Phase 2b (memory bank)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr-2a", type=float, default=5e-4)
    parser.add_argument("--lr-2b", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gate-entropy-weight", type=float, default=0.01)
    parser.add_argument("--attn-entropy-weight", type=float, default=0.001)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--freeze-epochs", type=int, default=3, help="Epochs to freeze encoders in Phase 2a")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-out", type=str, default="occlusion_aware_motion_lstm_ovis.pt")
    parser.add_argument("--skip-2a", action="store_true", help="Skip Phase 2a (load from --phase2a-checkpoint)")
    parser.add_argument("--phase2a-checkpoint", type=str, default="")
    parser.add_argument("--skip-2b", action="store_true", help="Skip Phase 2b")
    # Ablation flags
    parser.add_argument("--no-attention", action="store_true", help="Ablation: disable temporal attention")
    parser.add_argument("--no-memory", action="store_true", help="Ablation: disable memory readout")
    parser.add_argument("--no-adaptive-gate", action="store_true", help="Ablation: use fixed gating")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data — support both separate files and single file with auto-split
    if args.train_annotations and args.val_annotations:
        train_ann_path = Path(args.train_annotations)
        val_ann_path = Path(args.val_annotations)
        print("Loading standard training samples...")
        train_samples = parse_ovis_json_occlusion_aware(train_ann_path, history=args.history)
        val_samples = parse_ovis_json_occlusion_aware(val_ann_path, history=args.history)
        print(f"Standard samples: {len(train_samples)} train, {len(val_samples)} val")
        print("Loading gap training samples...")
        gap_train_samples = parse_ovis_json_gap_samples(train_ann_path, history=args.history)
        gap_val_samples = parse_ovis_json_gap_samples(val_ann_path, history=args.history)
    else:
        ann_path = Path(args.annotations or args.train_annotations)
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        print(f"Loading samples from {ann_path} (will auto-split 90/10)...")
        all_samples = parse_ovis_json_occlusion_aware(ann_path, history=args.history)
        random.shuffle(all_samples)
        split_idx = int(0.9 * len(all_samples))
        train_samples = all_samples[:split_idx]
        val_samples = all_samples[split_idx:]
        print(f"Standard samples: {len(train_samples)} train, {len(val_samples)} val")
        print("Loading gap samples...")
        all_gap_samples = parse_ovis_json_gap_samples(ann_path, history=args.history)
        random.shuffle(all_gap_samples)
        gap_split = int(0.9 * len(all_gap_samples))
        gap_train_samples = all_gap_samples[:gap_split]
        gap_val_samples = all_gap_samples[gap_split:]

    print(f"Gap samples: {len(gap_train_samples)} train, {len(gap_val_samples)} val")

    train_set = OcclusionAwareMotionDataset(train_samples)
    val_set = OcclusionAwareMotionDataset(val_samples)

    # Compute normalization stats (reuse multimodal utility)
    norm_stats = compute_multimodal_norm_stats(train_set)
    y_mean, y_std = compute_target_norm_stats(train_set)
    y_mean = y_mean.view(1, -1).to(device)
    y_std = y_std.view(1, -1).to(device)

    norm_on_device: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for key, (mean, std) in norm_stats.items():
        norm_on_device[key] = (mean.view(1, 1, -1).to(device), std.view(1, 1, -1).to(device))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_occlusion_aware)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_occlusion_aware)

    # Create model
    model = OcclusionAwareMultiModalLSTM(hidden_dim=args.hidden_dim).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Load Phase 1 weights
    phase1_path = Path(args.phase1_checkpoint)
    if phase1_path.exists():
        print(f"Loading Phase 1 weights from {phase1_path}...")
        load_phase1_weights(model, phase1_path, device)
    else:
        print(f"Phase 1 checkpoint not found at {phase1_path}, training from scratch")

    out_path = Path(args.model_out)
    history_log: List[Dict] = []
    best_val_iou = float("-inf")
    best_epoch = 0

    # ========================
    # Phase 2a: Temporal Attention
    # ========================
    if not args.skip_2a:
        print("\n" + "=" * 60)
        print("PHASE 2a: Training Temporal Attention")
        print("=" * 60)

        # Freeze encoders for warm-up
        encoder_params = []
        other_params = []
        for name, param in model.named_parameters():
            is_encoder = any(name.startswith(p) for p in [
                "vel_encoder.", "shape_encoder.", "accel_encoder.", "context_encoder.",
            ])
            if is_encoder:
                encoder_params.append(param)
            else:
                other_params.append(param)

        optimizer = torch.optim.Adam([
            {"params": other_params, "lr": args.lr_2a},
            {"params": encoder_params, "lr": 0.0},  # frozen initially
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_2a, eta_min=1e-5)

        for epoch in range(1, args.epochs_2a + 1):
            # Unfreeze encoders after warm-up
            if epoch == args.freeze_epochs + 1:
                print(f"  Unfreezing encoders at epoch {epoch}")
                optimizer.param_groups[1]["lr"] = args.lr_2a * 0.5

            metrics = train_epoch_phase2(
                model, train_loader, optimizer,
                norm_on_device, y_mean, y_std, device,
                gate_entropy_weight=args.gate_entropy_weight,
                attn_entropy_weight=args.attn_entropy_weight,
                grad_clip=args.grad_clip,
            )
            val_metrics = evaluate_phase2(model, val_loader, norm_on_device, y_mean, y_std, device)

            scheduler.step()

            log_entry = {"phase": "2a", "epoch": epoch, **metrics, **val_metrics}
            history_log.append(log_entry)

            print(
                f"Epoch {epoch:02d} | loss={metrics['train_loss']:.4f} | "
                f"val_iou={val_metrics['val_iou']:.3f} | val_mAP={val_metrics['val_mAP']:.3f} | "
                f"attn_H={val_metrics['attn_entropy']:.3f}"
            )
            print(
                f"  Gates: vel={val_metrics['gate_vel']:.3f} "
                f"shape={val_metrics['gate_shape']:.3f} "
                f"accel={val_metrics['gate_accel']:.3f}"
            )
            print(
                f"  AP@0.50={val_metrics['AP@0.50']:.3f} "
                f"AP@0.75={val_metrics['AP@0.75']:.3f} "
                f"AP@0.90={val_metrics['AP@0.90']:.3f}"
            )

            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                best_epoch = epoch
                _save_checkpoint(model, norm_on_device, y_mean, y_std, args, best_val_iou, best_epoch, "2a", out_path)
                print(f"  New best checkpoint at epoch {epoch}")

        # Save Phase 2a final
        phase2a_path = out_path.with_name(f"{out_path.stem}_phase2a{out_path.suffix}")
        _save_checkpoint(model, norm_on_device, y_mean, y_std, args, best_val_iou, best_epoch, "2a", phase2a_path)
        print(f"Saved Phase 2a checkpoint to {phase2a_path}")

    elif args.phase2a_checkpoint:
        print(f"Loading Phase 2a checkpoint from {args.phase2a_checkpoint}...")
        ckpt = torch.load(args.phase2a_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        best_val_iou = ckpt.get("best_val_iou", 0.0)

    # ========================
    # Phase 2b: Memory Bank + Gap Training
    # ========================
    if not args.skip_2b and gap_train_samples:
        print("\n" + "=" * 60)
        print("PHASE 2b: Training Memory Bank with Gap Samples")
        print("=" * 60)

        gap_train_set = OcclusionAwareMotionDataset(gap_train_samples)
        gap_loader = DataLoader(
            gap_train_set, batch_size=max(1, args.batch_size // 4),
            shuffle=True, collate_fn=collate_occlusion_aware,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_2b)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_2b, eta_min=1e-5)

        for epoch in range(1, args.epochs_2b + 1):
            # Gap curriculum
            if epoch <= 5:
                max_gap_curr = 3
            elif epoch <= 10:
                max_gap_curr = 6
            else:
                max_gap_curr = MAX_GAP_LENGTH

            metrics = train_epoch_phase2(
                model, train_loader, optimizer,
                norm_on_device, y_mean, y_std, device,
                gate_entropy_weight=args.gate_entropy_weight,
                attn_entropy_weight=args.attn_entropy_weight,
                grad_clip=args.grad_clip,
                gap_loader=gap_loader,
                max_gap_curriculum=max_gap_curr,
            )
            val_metrics = evaluate_phase2(model, val_loader, norm_on_device, y_mean, y_std, device)

            # Evaluate gap prediction
            gap_metrics = evaluate_gap_prediction(
                model, gap_val_samples, norm_on_device, y_mean, y_std, device,
            )

            scheduler.step()

            log_entry = {"phase": "2b", "epoch": epoch, "gap_curriculum": max_gap_curr, **metrics, **val_metrics}
            for bname, bmetrics in gap_metrics.items():
                for mk, mv in bmetrics.items():
                    log_entry[f"{bname}_{mk}"] = mv
            history_log.append(log_entry)

            gap_summary = " | ".join(
                f"{k}: iou={v['iou']:.3f} mAP={v['mAP']:.3f} (n={v['n']})"
                for k, v in gap_metrics.items() if v["n"] > 0
            )

            print(
                f"Epoch {epoch:02d} | loss={metrics['train_loss']:.4f} | "
                f"val_iou={val_metrics['val_iou']:.3f} | val_mAP={val_metrics['val_mAP']:.3f} | "
                f"gap_curriculum<={max_gap_curr}"
            )
            if "train_gap_loss" in metrics:
                print(f"  Gap loss={metrics['train_gap_loss']:.4f} (n={metrics.get('n_gap_samples', 0)})")
            print(f"  Gap eval: {gap_summary}")

            if val_metrics["val_iou"] > best_val_iou:
                best_val_iou = val_metrics["val_iou"]
                best_epoch = epoch
                _save_checkpoint(model, norm_on_device, y_mean, y_std, args, best_val_iou, best_epoch, "2b", out_path)
                print(f"  New best checkpoint at epoch {epoch}")

    # Save final
    final_path = out_path.with_name(f"{out_path.stem}_last{out_path.suffix}")
    _save_checkpoint(model, norm_on_device, y_mean, y_std, args, best_val_iou, best_epoch, "final", final_path)
    print(f"\nSaved final checkpoint to {final_path}")
    print(f"Best val IoU: {best_val_iou:.4f} at epoch {best_epoch}")

    # Save training log
    log_path = out_path.with_suffix(".json")
    log_path.write_text(json.dumps(history_log, indent=2))
    print(f"Saved training log to {log_path}")


def _save_checkpoint(
    model, norm_on_device, y_mean, y_std, args, best_val_iou, best_epoch, phase, path,
):
    torch.save({
        "model_state": model.state_dict(),
        "norm_stats": {k: (m.cpu(), s.cpu()) for k, (m, s) in norm_on_device.items()},
        "target_mean": y_mean.cpu(),
        "target_std": y_std.cpu(),
        "history": args.history,
        "hidden_dim": args.hidden_dim,
        "model_type": "occlusion_aware",
        "phase": phase,
        "best_val_iou": best_val_iou,
        "best_epoch": best_epoch,
    }, path)


if __name__ == "__main__":
    main()
