"""Generate all figures for the paper comparing baseline vs multi-modal models."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(exist_ok=True)

# Load training logs
baseline_log = json.loads(Path("baseline_motion_lstm_ovis.json").read_text())
multimodal_log = json.loads(Path("multimodal_motion_lstm_ovis.json").read_text())

b_epochs = [e["epoch"] for e in baseline_log]
m_epochs = [e["epoch"] for e in multimodal_log]

# ---- Figure 1: Training Loss Curves ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(b_epochs, [e["train_loss"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=4)
ax.plot(m_epochs, [e["train_loss"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training Loss (normalized)")
ax.set_title("Training Loss")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / "training_loss.png", dpi=200)
plt.close(fig)
print("Saved training_loss.png")

# ---- Figure 2: Validation IoU ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(b_epochs, [e["val_iou"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=4)
ax.plot(m_epochs, [e["val_iou"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Mean IoU")
ax.set_title("Validation IoU")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / "val_iou.png", dpi=200)
plt.close(fig)
print("Saved val_iou.png")

# ---- Figure 3: Validation mAP ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(b_epochs, [e["val_mAP"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=4)
ax.plot(m_epochs, [e["val_mAP"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP (0.50:0.95)")
ax.set_title("Validation mAP (COCO-style)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / "val_map.png", dpi=200)
plt.close(fig)
print("Saved val_map.png")

# ---- Figure 4: Validation Center L2 Distance ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(b_epochs, [e["val_l2"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=4)
ax.plot(m_epochs, [e["val_l2"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Center L2 Distance (px)")
ax.set_title("Validation Center Localization Error")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(FIGS_DIR / "val_l2.png", dpi=200)
plt.close(fig)
print("Saved val_l2.png")

# ---- Figure 5: Per-threshold AP comparison (best epoch) ----
# Find best epoch by mAP for each
b_best = max(baseline_log, key=lambda e: e["val_mAP"])
m_best = max(multimodal_log, key=lambda e: e["val_mAP"])

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
thresh_labels = [f"{t:.2f}" for t in thresholds]
b_aps = [b_best[f"AP@{t:.2f}"] for t in thresholds]
m_aps = [m_best[f"AP@{t:.2f}"] for t in thresholds]

x = np.arange(len(thresholds))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4.5))
bars1 = ax.bar(x - width/2, b_aps, width, label=f"Baseline (epoch {b_best['epoch']})", color="#2196F3", alpha=0.85)
bars2 = ax.bar(x + width/2, m_aps, width, label=f"Multi-Modal (epoch {m_best['epoch']})", color="#E91E63", alpha=0.85)
ax.set_xlabel("IoU Threshold")
ax.set_ylabel("AP")
ax.set_title("Per-Threshold AP at Best Checkpoint")
ax.set_xticks(x)
ax.set_xticklabels(thresh_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(FIGS_DIR / "per_threshold_ap.png", dpi=200)
plt.close(fig)
print("Saved per_threshold_ap.png")

# ---- Figure 6: Gate weight evolution ----
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(m_epochs, [e["gate_vel"] for e in multimodal_log], "o-", label="Velocity", color="#4CAF50", markersize=4)
ax.plot(m_epochs, [e["gate_shape"] for e in multimodal_log], "s-", label="Shape", color="#FF9800", markersize=4)
ax.plot(m_epochs, [e["gate_accel"] for e in multimodal_log], "^-", label="Acceleration", color="#9C27B0", markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Average Gate Weight")
ax.set_title("Multi-Modal Gated Fusion Weights")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0.0, 0.5)
fig.tight_layout()
fig.savefig(FIGS_DIR / "gate_weights.png", dpi=200)
plt.close(fig)
print("Saved gate_weights.png")

# ---- Figure 7: Combined 2x2 summary ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

ax = axes[0, 0]
ax.plot(b_epochs, [e["train_loss"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=3)
ax.plot(m_epochs, [e["train_loss"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=3)
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss"); ax.set_title("(a) Training Loss")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(b_epochs, [e["val_mAP"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=3)
ax.plot(m_epochs, [e["val_mAP"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=3)
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP"); ax.set_title("(b) Validation mAP (0.50:0.95)")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(b_epochs, [e["val_iou"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=3)
ax.plot(m_epochs, [e["val_iou"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=3)
ax.set_xlabel("Epoch"); ax.set_ylabel("Mean IoU"); ax.set_title("(c) Validation IoU")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(b_epochs, [e["val_l2"] for e in baseline_log], "o-", label="Baseline", color="#2196F3", markersize=3)
ax.plot(m_epochs, [e["val_l2"] for e in multimodal_log], "s-", label="Multi-Modal", color="#E91E63", markersize=3)
ax.set_xlabel("Epoch"); ax.set_ylabel("L2 (px)"); ax.set_title("(d) Center Localization Error")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

fig.suptitle("Baseline vs Multi-Modal Motion Model — OVIS Dataset", fontsize=13, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(FIGS_DIR / "summary_2x2.png", dpi=200)
plt.close(fig)
print("Saved summary_2x2.png")

# ---- Print summary table for paper ----
print("\n" + "="*70)
print("SUMMARY FOR PAPER (best checkpoint by mAP)")
print("="*70)
print(f"{'Metric':<25} {'Baseline (ep {})'.format(b_best['epoch']):<20} {'Multi-Modal (ep {})'.format(m_best['epoch']):<20} {'Delta':<10}")
print("-"*70)
for label, bk, mk in [
    ("mAP (0.50:0.95)", "val_mAP", "val_mAP"),
    ("AP@0.50", "AP@0.50", "AP@0.50"),
    ("AP@0.75", "AP@0.75", "AP@0.75"),
    ("AP@0.90", "AP@0.90", "AP@0.90"),
    ("Mean IoU", "val_iou", "val_iou"),
    ("Center L2 (px)", "val_l2", "val_l2"),
    ("Val MSE (px)", "val_mse", "val_mse"),
    ("Train Loss", "train_loss", "train_loss"),
]:
    bv = b_best[bk]
    mv = m_best[mk]
    delta = mv - bv
    sign = "+" if delta > 0 else ""
    print(f"{label:<25} {bv:<20.4f} {mv:<20.4f} {sign}{delta:.4f}")
print("="*70)
