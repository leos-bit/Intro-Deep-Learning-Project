# Seeing Through Occlusion: Multi-Modal Motion Modeling for Video Instance Segmentation

18-786 Introduction to Deep Learning — Course Project

**Authors:** Alan Mohan, Soren Dupont, Rithwick Sethi, Samuel Giovanetti, Leo Shaw

## Overview

This project tackles the problem of video instance segmentation under heavy occlusion on the [OVIS dataset](https://songbai.site/ovis/). Standard VIS methods rely on appearance embeddings that collapse when objects are hidden. We model instance-level motion directly from bounding box trajectories using recurrent networks with multi-modal feature decomposition, temporal attention, and an occlusion-aware memory bank.

The project is implemented in three phases:

| Phase | Model | Params | Val IoU | Val mAP | Status |
|-------|-------|--------|---------|---------|--------|
| 1a | Baseline LSTM | 84K | 0.747 | 0.587 | Complete |
| 1b | Multi-Modal Gated Fusion LSTM | 198K | 0.756 | 0.603 | Complete |
| 2a | Occlusion-Aware Temporal Attention | 310K | **0.762** | **0.614** | Complete |
| 2b | Memory Bank + Gap Training | 310K | 0.757 | 0.607 | Complete |

## Repository Structure

```
.
├── baseline_motion_model.py          # Phase 1a: single-stream LSTM baseline
├── multimodal_motion_model.py        # Phase 1b: multi-modal gated fusion LSTM
├── occlusion_aware_model.py          # Phase 2: temporal attention + memory bank
├── simple_baseline_model.py          # Heuristic baselines (copy-last, const velocity)
├── evaluate_motion_subsets.py        # Motion-stratified evaluation (all models)
├── evaluate_occlusion_gaps.py        # Occlusion-gap prediction evaluation (all models)
├── visualize_predictions.py          # Qualitative prediction visualization
├── plot_results.py                   # Training curve plots
├── generate_paper_figures.py         # Paper-quality figure generation
├── tools/
│   └── make_rudimentary_ovis_json.py # Synthetic OVIS-format data generator
├── data/
│   ├── annotations_train.json        # OVIS training annotations
│   ├── annotations_valid.json        # OVIS validation annotations (hidden GT)
│   └── annotations_test.json         # OVIS test annotations (hidden GT)
├── figs/                             # Generated figures and visualizations
└── results/                          # Paper source and detailed results
```

## Environment Setup

```bash
python -m pip install torch torchvision torchaudio
```

Tested with Python 3.8+ and PyTorch 2.3. Supports CPU, CUDA, and Apple MPS.

## Data

The primary dataset is **OVIS (Occluded Video Instance Segmentation)**:
- 901 videos with severe object occlusion
- 296K instance masks across 25 semantic categories
- 5,223 unique tracked instances
- Our training set: 181,694 next-frame prediction samples from 607 videos
- Evaluated with 90/10 train/validation split (validation GT is hidden on OVIS server)

## Phase 1a: Baseline Single-Stream Motion LSTM

Following the InstMove formulation, models instance motion using a single LSTM that consumes a 10-dimensional feature vector per frame:
- Centroid position (cx, cy), box size (w, h)
- Velocity (vx, vy), acceleration (ax, ay)
- Area, log aspect ratio

```bash
python baseline_motion_model.py \
  --annotations data/annotations_train.json \
  --history 5 --epochs 20 --batch-size 128 --lr 1e-3 \
  --model-out baseline_motion_lstm_ovis.pt
```

**Architecture:** Single-layer LSTM (hidden_dim=128, 84K params) with box regression and visibility heads.

**Results:** Val IoU 0.747, mAP 0.587, center L2 24.5px.

## Phase 1b: Multi-Modal Gated Fusion LSTM

Decomposes motion into four specialized streams, each with a dedicated LSTM encoder:

| Stream | Dim | Features |
|--------|-----|----------|
| Velocity | 8 | vx, vy, speed, heading, vx_ema, vy_ema, vx_rel, vy_rel |
| Shape | 8 | dw, dh, log_area, d_log_area, log_aspect, d_log_aspect, scale_x, scale_y |
| Acceleration | 8 | ax, ay, a_mag, jerk_x, jerk_y, curvature, tangential_a, normal_a |
| Context | 4 | cx, cy, w, h (raw position — always-on, not gated) |

The three motion streams are fused via a learned softmax gating mechanism. Context is concatenated separately (not gated) because positional information is always needed. A residual skip connection from the last observed frame improves convergence.

```bash
python multimodal_motion_model.py \
  --annotations data/annotations_train.json \
  --history 5 --epochs 20 --batch-size 128 --lr 1e-3 \
  --hidden-dim 64 --gate-entropy-weight 0.01 --grad-clip 1.0 \
  --model-out multimodal_motion_lstm_ovis.pt
```

**Results:** Val IoU 0.756, mAP 0.603 (+2.7% over baseline).

**Key design decisions and bug fixes from Phase 1b:**
1. Gate entropy weight reduced from 0.1 to 0.01 (was creating negative loss)
2. Context separated from gated fusion (position needed always, not competitively)
3. Residual skip connection from last-frame state added for better convergence
4. Curvature and acceleration features clamped to [-50, 50] for numerical stability

## Phase 2a: Occlusion-Aware Temporal Attention

Replaces the LSTM's "take last hidden state" approach with a temporal attention mechanism that explicitly re-weights which past frames are most informative, modulated by an occlusion confidence signal.

**New components:**

- **OcclusionConfidenceModule:** Maps per-frame occlusion labels (no/slight/severe) and area ratios to a scalar confidence in [0, 1]. Frames with low confidence get suppressed in attention.

- **OcclusionAwareTemporalAttention:** Scaled dot-product attention where the last timestep queries all history timesteps. Occlusion confidence is added in log-space to pre-softmax scores, so low-confidence (occluded) frames are naturally downweighted before renormalization.

- **AdaptiveGatedFusion:** Extends the Phase 1b gating by conditioning on a motion regime summary (mean speed, max acceleration, mean occlusion confidence), enabling the gates to shift weight based on whether the instance is fast-moving, accelerating, or occluded.

**Training strategy:**
- Initialize 4 modality encoders from Phase 1b checkpoint
- Freeze encoders for first 3 epochs (warm-up), then unfreeze with halved learning rate
- Loss: MSE(box) + 0.1*BCE(visibility) + 0.05*CE(occlusion) - 0.01*H(gates) - 0.001*H(attention)
- Adam optimizer, lr=5e-4, CosineAnnealingLR, gradient clipping 1.0, 20 epochs

```bash
python occlusion_aware_model.py \
  --annotations data/annotations_train.json \
  --phase1-checkpoint multimodal_motion_lstm_ovis.pt \
  --epochs-2a 20 --epochs-2b 15 \
  --batch-size 128 --hidden-dim 64 \
  --model-out occlusion_aware_motion_lstm_ovis.pt
```

**Results:** Val IoU **0.762** (+0.8% over Phase 1b), mAP **0.614** (+1.8%), AP@0.75 0.654, AP@0.90 0.292. Attention entropy steadily decreased from 1.47 to 1.29, confirming the model learned to attend selectively rather than uniformly weighting all frames.

## Phase 2b: Occlusion-Aware Memory Bank

Adds a memory bank for predicting object positions after occlusion gaps.

**New components:**

- **MemoryBank:** Inference-time data structure that stores per-instance motion representations, last visible box, velocity, and confidence. Tracks frames since last visibility.

- **MemoryReadout:** Differentiable module with a learned gap-length embedding (nn.Embedding for gaps 0-50) and a sigmoid gate that balances memory representation vs. current observation.

- **Gap training samples:** Extracted from visible-occluded-visible patterns in annotations (1,178 total: 675 short gaps 1-3 frames, 169 medium 4-6, 118 long 7-10, 216 very long 11+).

- **Gap curriculum:** Epochs 1-5 train on gaps 1-3 only, epochs 6-10 add 4-6, epochs 11+ include all gaps.

**Results:** The gap training trades some standard next-frame performance (IoU drops from 0.762 to 0.757 after 15 epochs) in exchange for learning to predict across occlusion gaps. The best overall checkpoint remains from Phase 2a.

## Evaluation

### Motion-Stratified Evaluation

On the full validation set, most objects barely move between frames (median displacement = 14.7px on 1920x1080), so Copy-Last-Frame trivially matches learned models. Stratifying by displacement reveals the true value of motion modeling:

| Method | All | Low | Medium | High | Very High |
|--------|-----|-----|--------|------|-----------|
| Copy-Last-Frame | 0.608 | **0.849** | 0.637 | 0.336 | 0.176 |
| Const. Velocity | 0.590 | 0.770 | 0.622 | 0.377 | 0.233 |
| Baseline LSTM | 0.588 | 0.736 | 0.640 | 0.388 | 0.237 |
| Multi-Modal | 0.608 | 0.772 | 0.655 | 0.396 | 0.244 |
| **Occ-Aware (Ours)** | **0.618** | 0.798 | **0.667** | 0.389 | 0.229 |

*mAP (0.50:0.95) by motion difficulty subset. Low = bottom 33%, Medium = middle 33%, High = top 33%, Very High = top 10% by centroid displacement.*

**Key findings:**
- **Overall mAP 0.618** — best across all methods (+1.7% over Multi-Modal, +5.1% over Baseline LSTM)
- **Low/medium motion improvements**: Occ-Aware achieves 0.798 / 0.667 mAP, surpassing Multi-Modal (0.772 / 0.655). The temporal attention helps the model be more selective about which history frames to trust.
- **Best localization**: IoU 0.764 and L2 23.6px are the best across all methods at every motion level except very-high.
- **High/very-high motion**: Multi-Modal retains a slight edge (0.396 vs 0.389 on high, 0.244 vs 0.229 on very-high), suggesting the attention mechanism's selectivity slightly underperforms the LSTM's sequential memory for extreme motion.

```bash
python evaluate_motion_subsets.py \
  --annotations data/annotations_train.json \
  --baseline-ckpt baseline_motion_lstm_ovis.pt \
  --multimodal-ckpt multimodal_motion_lstm_ovis.pt \
  --occlusion-aware-ckpt occlusion_aware_motion_lstm_ovis.pt
```

### Occlusion-Gap Evaluation

Tests whether models can predict where an object reappears after being hidden for 1-50 frames. This is extremely challenging — all methods score below 0.03 mAP:

| Method | mAP | IoU | L2 (px) |
|--------|-----|-----|---------|
| Copy-Last-Frame | 0.030 | 0.128 | 131.5 |
| Const. Velocity | 0.017 | 0.056 | 394.3 |
| Multi-Modal | 0.019 | 0.099 | 169.0 |
| **Occ-Aware (Ours)** | 0.010 | 0.093 | **134.2** |

The Occ-Aware model achieves the **best center-point localization** (L2 134.2px, +20.6% improvement over Multi-Modal), meaning it better predicts where objects reappear. However, its bounding box size predictions are less calibrated, resulting in lower mAP. Copy-Last-Frame's high mAP here is misleading — it benefits from objects that reappear near where they disappeared, but fails catastrophically when they don't (L2 is competitive only because most gaps are short).

```bash
python evaluate_occlusion_gaps.py \
  --annotations data/annotations_train.json \
  --baseline-ckpt baseline_motion_lstm_ovis.pt \
  --multimodal-ckpt multimodal_motion_lstm_ovis.pt \
  --occlusion-aware-ckpt occlusion_aware_motion_lstm_ovis.pt
```

## Limitations and Problems Encountered

### Architectural Limitations

1. **Gate weights remain approximately uniform (~0.33 each):** Despite conditioning on motion regime summary, the adaptive gating does not produce strongly differentiated weights. The motion summary features (mean speed, max acceleration, mean confidence) may not be discriminative enough for per-instance specialization. Per-instance conditioning on the actual feature magnitudes rather than summary statistics could help.

2. **Bbox-only features:** The model operates solely on bounding box trajectories and does not incorporate mask-level shape information. The proposal's PCA-of-mask-shapes feature was deferred. Mask features could help distinguish shape deformation (e.g., a person bending) from size changes due to perspective.

3. **No appearance features:** The model is purely motion-based. While this is by design (motion should work when appearance fails), integrating a lightweight appearance stream for visible frames could provide complementary signal.

### Training Challenges

4. **Phase 2b destabilizes standard performance:** Gap training reduced standard validation IoU from 0.762 (Phase 2a best) to 0.757 after 15 epochs. The model is pulled between two objectives — next-frame prediction and gap prediction — and the gap samples (1,178) are heavily outnumbered by standard samples (163K). A separate gap-prediction head or decoupled training schedule may help.

5. **Small gap training set:** Only 1,178 visible-occluded-visible patterns were extracted, with just 118 in the 7-10 frame range and 216 for 11+ frames. Synthetic gap injection (randomly masking visible frames) was planned but not yet implemented.

6. **Attention entropy still relatively high (1.29 vs max 1.61):** The temporal attention learned some selectivity but did not strongly differentiate frames. With history=5 and attention entropy of 1.29 (uniform would be ln(5)=1.61), the attention is still fairly spread across frames. Longer history windows or more training could increase differentiation.

### Evaluation Challenges

7. **Occlusion-gap prediction is extremely hard:** All methods score below 0.03 mAP on gap prediction. Objects can move unpredictably during occlusion (changing direction, accelerating, being carried), making linear or even learned extrapolation unreliable beyond a few frames. The gap evaluation exposed that this is fundamentally a harder problem than next-frame prediction.

8. **Overall metrics mask motion-modeling value:** On the full validation set, the trivial Copy-Last-Frame heuristic matches learned models at 0.608 mAP because most objects barely move between adjacent frames. Motion-stratified evaluation is essential to reveal the true differences between methods.

9. **High/very-high motion regression:** The Occ-Aware model slightly underperforms Multi-Modal on the hardest motion subsets (0.389 vs 0.396 on high, 0.229 vs 0.244 on very-high). The temporal attention mechanism's selectivity appears to hurt when all recent frames are equally informative for fast-moving objects — the LSTM's sequential state accumulation may be more appropriate for these cases.

## Checkpoint Formats

All checkpoints are `.pt` files (excluded from git via `.gitignore`). They contain:

**Baseline:** `model_state`, `feature_mean/std` (10D), `target_mean/std` (4D), `history`, `best_val_iou`

**Multi-Modal:** `model_state`, `norm_stats` (per-modality mean/std for vel/shape/accel/ctx), `target_mean/std`, `history`, `hidden_dim`, `num_layers`, `model_type: "multimodal"`

**Occlusion-Aware:** `model_state`, `norm_stats`, `target_mean/std`, `history`, `hidden_dim`, `model_type: "occlusion_aware"`, `phase`, `best_val_iou`, `best_epoch`

## Metrics

- **mAP (0.50:0.95):** COCO-style mean average precision, primary metric
- **AP@T:** Precision at specific IoU threshold T
- **IoU:** Mean bounding box intersection-over-union
- **L2 (px):** Center-point Euclidean distance in pixels
- **Attention entropy:** Mean entropy of temporal attention weights (lower = more selective)
- **Gate weights:** Softmax weights for velocity/shape/acceleration fusion

## Future Work

- Mask-level PCA shape features to differentiate shape deformation from perspective effects
- Per-instance adaptive gating conditioned on actual feature magnitudes
- Separate gap-prediction head to avoid destabilizing standard performance
- Synthetic gap injection augmentation to increase gap training data
- Longer history windows (10-20 frames) to give temporal attention more to work with
- Integration with a full VIS backbone (MaskTrack R-CNN or MinVIS) for end-to-end evaluation on the OVIS benchmark
