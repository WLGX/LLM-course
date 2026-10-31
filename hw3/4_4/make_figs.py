# -*- coding: utf-8 -*-
"""4.4 结果绘图：训练损失曲线 + 测试集各类别 F1。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
# 用 Windows 的中文字体渲染标点类名（WSL 内经 /mnt/c 访问）
_zh_font = "/mnt/c/Windows/Fonts/msyh.ttc"
if os.path.exists(_zh_font):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_zh_font)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 图1：训练损失
steps, losses = [], []
with open(os.path.join(RES, "train_log.jsonl"), encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        steps.append(r["step"])
        losses.append(r["loss"])
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(steps, losses, lw=1.2, color="#1f77b4")
ax.set_xlabel("step")
ax.set_ylabel("train loss")
ax.set_title("BERT fine-tune loss (punctuation, 人民日报9801)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_train_loss.png"), dpi=160)
plt.close(fig)

# 图2：测试集各类别 F1（排除 O 的占比干扰，按非 O 类排序）
m = json.load(open(os.path.join(RES, "test_metrics.json"), encoding="utf-8"))
pc = m["per_class"]
names = [k for k in pc if k not in ("O", "accuracy", "macro avg", "weighted avg")]
names.sort(key=lambda k: -pc[k]["f1-score"])
f1s = [pc[k]["f1-score"] for k in names]
fig, ax = plt.subplots(figsize=(7, 3.4))
bars = ax.bar(names, f1s, color="#2ca02c")
for b, v in zip(bars, f1s):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1")
ax.set_title("Test F1 per punctuation class (macro-F1 = %.3f)" % m["test_macro_f1"])
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_test_f1.png"), dpi=160)
plt.close(fig)
print("figs saved:", FIG)
