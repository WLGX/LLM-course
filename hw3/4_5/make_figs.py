# -*- coding: utf-8 -*-
"""4.5 小模型消融实验结果绘图：MLM 验证损失曲线 + 双指标对比柱状图。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(RES, "figs")
os.makedirs(FIG, exist_ok=True)

_zh = "/mnt/c/Windows/Fonts/msyh.ttc"
if os.path.exists(_zh):
    from matplotlib import font_manager
    font_manager.fontManager.addfont(_zh)
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

data = json.load(open(os.path.join(RES, "mini_ablation.json"), encoding="utf-8"))

# 图1：各变体 MLM 验证损失曲线
fig, ax = plt.subplots(figsize=(7, 3.8))
for name, v in data.items():
    steps = [r["step"] for r in v["log"]]
    vals = [r["valid_loss"] for r in v["log"]]
    ax.plot(steps, vals, label=name, lw=1.2)
ax.set_xlabel("step")
ax.set_ylabel("MLM valid loss")
ax.set_title("Micro-encoder ablations: MLM valid loss (fixed budget 3000 steps)")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ablation_mlm_loss.png"), dpi=160)
plt.close(fig)

# 图2：双指标对比：左轴 MLM 损失、右轴探针 F1
names = list(data)
mlm = [data[n]["final_valid_mlm"] for n in names]
f1 = [data[n]["probe_f1"] for n in names]
x = range(len(names))
fig, ax1 = plt.subplots(figsize=(7, 3.8))
c1 = "#1f77b4"
ax1.bar([i - 0.2 for i in x], mlm, 0.38, color=c1, label="MLM valid loss (越低越好)")
ax1.set_ylabel("MLM valid loss", color=c1)
ax1.tick_params(axis="y", labelcolor=c1)
ax2 = ax1.twinx()
c2 = "#d62728"
ax2.bar([i + 0.2 for i in x], f1, 0.38, color=c2, label="断句探针 macro-F1 (越高越好)")
ax2.set_ylabel("probe macro-F1", color=c2)
ax2.tick_params(axis="y", labelcolor=c2)
ax1.set_xticks(list(x))
ax1.set_xticklabels(names)
ax1.set_title("代理指标（MLM 损失）vs 应用指标（断句 F1）")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "ablation_dual_metric.png"), dpi=160)
plt.close(fig)
print("figs saved")
