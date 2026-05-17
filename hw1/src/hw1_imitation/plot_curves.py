import pandas as pd
import matplotlib.pyplot as plt

# ── 读取数据 ──────────────────────────────────────────
loss_df_mse = pd.read_csv("/Users/luyongfei/Desktop/AI New Courses/proj-hw/homework_spring2026/hw1/exp/seed_42_20260516_124338/log.csv")
reward_df_mse = pd.read_csv("/Users/luyongfei/Desktop/AI New Courses/proj-hw/homework_spring2026/hw1/exp/seed_42_20260516_124338/wandb_export_2026-05-16T13_01_19.943-07_00.csv")
reward_df_mse.columns = ["step", "reward", "reward_min", "reward_max"]

loss_df_flow = pd.read_csv("/Users/luyongfei/Desktop/AI New Courses/proj-hw/homework_spring2026/hw1/exp/seed_42_20260517_141407/log.csv")
reward_df_flow = pd.read_csv("/Users/luyongfei/Desktop/AI New Courses/proj-hw/homework_spring2026/hw1/exp/seed_42_20260517_141407/wandb_export_2026-05-17T14_27_46.753-07_00.csv")
reward_df_flow.columns = ["step", "reward", "reward_min", "reward_max"]

# ── 画图 ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("MSE vs Flow Matching Training Results", fontsize=14, fontweight='bold')

# --- 图1: Loss 曲线 ---
# MSE loss
ax1.plot(loss_df_mse["step"], loss_df_mse["train/loss"],
         color="#FF5722", linewidth=1.5, alpha=0.3)
loss_smooth_mse = loss_df_mse["train/loss"].rolling(window=20, center=True).mean()
ax1.plot(loss_df_mse["step"], loss_smooth_mse,
         color="#FF5722", linewidth=2.5, label='MSE')

# Flow loss
ax1.plot(loss_df_flow["step"], loss_df_flow["train/loss"],
         color="#9C27B0", linewidth=1.5, alpha=0.3)
loss_smooth_flow = loss_df_flow["train/loss"].rolling(window=20, center=True).mean()
ax1.plot(loss_df_flow["step"], loss_smooth_flow,
         color="#9C27B0", linewidth=2.5, label='Flow Matching')

ax1.set_xlabel("Training Steps", fontsize=12)
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Training Loss", fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# --- 图2: Reward 曲线 ---
# MSE reward
ax2.plot(reward_df_mse["step"], reward_df_mse["reward"],
         color="#FF5722", linewidth=2.5, marker='o', markersize=7, label='MSE')

# Flow reward
ax2.plot(reward_df_flow["step"], reward_df_flow["reward"],
         color="#9C27B0", linewidth=2.5, marker='o', markersize=7, label='Flow Matching')

ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Target (0.5)')

ax2.set_xlabel("Training Steps", fontsize=12)
ax2.set_ylabel("Mean Reward", fontsize=12)
ax2.set_title("Evaluation Reward", fontsize=13)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 0.85)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches='tight')
print("图保存到 training_curves.png")