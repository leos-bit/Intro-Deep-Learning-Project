"""Generate comparison plots: Multi-Modal vs Baseline on OVIS data."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --- Multi-modal training log (from JSON) ---
mm_log = json.loads(Path("multimodal_motion_lstm_ovis.json").read_text())
mm_epochs = [r["epoch"] for r in mm_log]
mm_iou = [r["val_iou"] for r in mm_log]
mm_mse = [r["val_mse"] for r in mm_log]
mm_l2 = [r["val_l2"] for r in mm_log]
mm_loss = [r["train_loss"] for r in mm_log]
mm_gate_vel = [r["gate_vel"] for r in mm_log]
mm_gate_shape = [r["gate_shape"] for r in mm_log]
mm_gate_accel = [r["gate_accel"] for r in mm_log]

# --- Baseline results (copied from training output) ---
bl_epochs = list(range(1, 21))
bl_iou = [0.711, 0.702, 0.709, 0.723, 0.731, 0.732, 0.722, 0.704, 0.708, 0.739,
           0.726, 0.723, 0.723, 0.720, 0.735, 0.710, 0.747, 0.730, 0.743, 0.738]
bl_mse = [1678.84, 1667.53, 1607.49, 1631.91, 1618.12, 1587.93, 1582.66, 1630.72,
          1619.10, 1616.02, 1621.31, 1585.81, 1593.73, 1602.40, 1593.11, 1601.90,
          1565.85, 1605.81, 1570.67, 1585.79]
bl_l2 = [27.507, 27.144, 26.820, 26.333, 25.456, 26.168, 26.034, 27.923, 26.384,
         25.193, 26.794, 26.161, 26.670, 26.630, 25.174, 26.978, 24.572, 25.609,
         24.578, 25.553]
bl_loss = [0.0448, 0.0235, 0.0231, 0.0228, 0.0225, 0.0223, 0.0220, 0.0218, 0.0216,
           0.0213, 0.0212, 0.0210, 0.0208, 0.0205, 0.0204, 0.0202, 0.0200, 0.0198,
           0.0196, 0.0194]

# --- Plot style ---
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Multi-Modal Motion Model vs Baseline — OVIS Dataset", fontsize=15, fontweight="bold")

# 1. Val IoU
ax = axes[0, 0]
ax.plot(bl_epochs, bl_iou, "o-", color="#2196F3", label="Baseline (single LSTM)", linewidth=2, markersize=4)
ax.plot(mm_epochs, mm_iou, "s-", color="#E91E63", label="Multi-Modal (gated fusion)", linewidth=2, markersize=4)
ax.axhline(max(bl_iou), color="#2196F3", linestyle="--", alpha=0.5, label=f"Baseline best: {max(bl_iou):.3f}")
ax.axhline(max(mm_iou), color="#E91E63", linestyle="--", alpha=0.5, label=f"Multi-Modal best: {max(mm_iou):.3f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation IoU")
ax.set_title("Bounding Box IoU (higher is better)")
ax.legend(fontsize=9)

# 2. Val Center L2
ax = axes[0, 1]
ax.plot(bl_epochs, bl_l2, "o-", color="#2196F3", label="Baseline", linewidth=2, markersize=4)
ax.plot(mm_epochs, mm_l2, "s-", color="#E91E63", label="Multi-Modal", linewidth=2, markersize=4)
ax.axhline(min(bl_l2), color="#2196F3", linestyle="--", alpha=0.5, label=f"Baseline best: {min(bl_l2):.1f}")
ax.axhline(min(mm_l2), color="#E91E63", linestyle="--", alpha=0.5, label=f"Multi-Modal best: {min(mm_l2):.1f}")
ax.set_xlabel("Epoch")
ax.set_ylabel("Center L2 Distance (px)")
ax.set_title("Center Tracking Error (lower is better)")
ax.legend(fontsize=9)

# 3. Training Loss
ax = axes[1, 0]
ax.plot(bl_epochs, bl_loss, "o-", color="#2196F3", label="Baseline", linewidth=2, markersize=4)
ax.plot(mm_epochs, mm_loss, "s-", color="#E91E63", label="Multi-Modal", linewidth=2, markersize=4)
ax.set_xlabel("Epoch")
ax.set_ylabel("Normalized Training Loss")
ax.set_title("Training Loss Convergence")
ax.legend(fontsize=9)

# 4. Gate Weights
ax = axes[1, 1]
ax.plot(mm_epochs, mm_gate_vel, "^-", color="#4CAF50", label="Velocity", linewidth=2, markersize=5)
ax.plot(mm_epochs, mm_gate_shape, "D-", color="#FF9800", label="Shape Change", linewidth=2, markersize=5)
ax.plot(mm_epochs, mm_gate_accel, "v-", color="#9C27B0", label="Acceleration", linewidth=2, markersize=5)
ax.axhline(1/3, color="gray", linestyle=":", alpha=0.5, label="Uniform (1/3)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Gate Weight")
ax.set_title("Learned Modality Gate Weights")
ax.set_ylim(0.25, 0.42)
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "multimodal_vs_baseline.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'multimodal_vs_baseline.png'}")

# --- Bar chart: best checkpoint comparison ---
fig2, ax2 = plt.subplots(1, 3, figsize=(13, 5))
fig2.suptitle("Best Checkpoint Comparison", fontsize=14, fontweight="bold")

metrics = ["Val IoU", "Center L2 (px)", "Val MSE (px)"]
bl_best = [max(bl_iou), min(bl_l2), min(bl_mse)]
mm_best = [max(mm_iou), min(mm_l2), min(mm_mse)]
colors_bl = "#2196F3"
colors_mm = "#E91E63"

for i, (metric, bl_v, mm_v) in enumerate(zip(metrics, bl_best, mm_best)):
    bars = ax2[i].bar(["Baseline", "Multi-Modal"], [bl_v, mm_v],
                       color=[colors_bl, colors_mm], width=0.5, edgecolor="black", linewidth=0.5)
    ax2[i].set_title(metric, fontsize=12)
    ax2[i].set_ylabel(metric)
    for bar, val in zip(bars, [bl_v, mm_v]):
        ax2[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}" if i == 0 else f"{val:.1f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
    if i == 0:
        ax2[i].set_ylim(0.68, 0.78)
    elif i == 1:
        ax2[i].set_ylim(20, 28)
    else:
        ax2[i].set_ylim(1450, 1700)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "best_checkpoint_comparison.png", dpi=150, bbox_inches="tight")
print(f"Saved: {RESULTS_DIR / 'best_checkpoint_comparison.png'}")

print("\nDone. All plots saved to results/")
