# Intro Deep Learning Project: Rudimentary OVIS Motion Baseline

This repository contains a minimal baseline aligned to the project proposal:
- Instance-level motion modeling with an LSTM
- Multi-modal motion features from bbox history (`cx, cy, w, h, vx, vy, ax, ay, area, log_aspect`)
- Next-frame box regression plus visibility confidence head

## Repository Contents
- `baseline_motion_model.py`: train/eval script for the rudimentary baseline
- `tools/make_rudimentary_ovis_json.py`: generates OVIS-style toy annotations
- `data/rudimentary_ovis_train.json`: small sample annotation file

## Environment
Use your project virtual environment and install PyTorch:

```powershell
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio
```

## Generate Rudimentary OVIS-Style Data

Small example:

```powershell
python .\tools\make_rudimentary_ovis_json.py --out .\data\rudimentary_ovis_train.json --num-videos 3 --frames-per-video 20 --tracks-per-video 4
```

Recommended baseline size:

```powershell
python .\tools\make_rudimentary_ovis_json.py --out .\data\rudimentary_ovis_train_big.json --num-videos 40 --frames-per-video 40 --tracks-per-video 8 --occlusion-prob 0.15
```

## Train Baseline

Toy or single-file mode:

```powershell
python .\baseline_motion_model.py --annotations .\data\rudimentary_ovis_train_big.json --history 5 --epochs 20
```

Real OVIS split-aware mode:

```powershell
python .\baseline_motion_model.py --train-annotations .\tools\annotations_train.json --val-annotations .\tools\annotations_valid.json --history 5 --epochs 20 --model-out .\baseline_motion_lstm_real.pt
```

## Output
The script saves:
- `baseline_motion_lstm.pt`

Checkpoint payload includes:
- model weights
- input normalization stats
- target normalization stats
- history length

## Metrics Notes
- `train_loss_norm`: normalized training loss (should quickly decrease)
- `val_mse_px`: validation box MSE in pixel space
- `val_center_l2_px`: center-point error in pixels
- `val_iou`: bbox IoU (higher is better)

## Baseline Scope
This is a deliberately simple baseline to establish a starting point before adding:
- occlusion-aware reweighting
- temporal attention
- memory bank / stronger association modules