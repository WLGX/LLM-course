# -*- coding: utf-8 -*-
"""绘制 P226 报告的 2022–2026 后训练技术演进图（输出 演进图.png）。

布局原则：三条泳道各有独立的标签带（上泳道标签朝上、中/下泳道标签朝下），
不同泳道的标签互不重叠；同一泳道内相邻事件横向间距 ≥ 1.0。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

HERE = os.path.dirname(os.path.abspath(__file__))

for name in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.axis("off")

LANE_ALG, LANE_DATA, LANE_FLOW = 3.05, 2.05, 1.05
lanes = {"算法线": LANE_ALG, "数据与奖励线": LANE_DATA, "流程线": LANE_FLOW}
for name, y in lanes.items():
    ax.plot([0.4, 13.6], [y, y], color="#cccccc", lw=1, zorder=1)
    ax.text(0.08, y + 0.06, name, fontsize=11, color="#555555", va="bottom", zorder=5)

events = [
    # (x, lane, label, kind, year)
    (1.0, LANE_ALG, "PPO-RLHF\n(InstructGPT)", "alg"),
    (1.0, LANE_DATA, "人类演示+比较标注\n(13k/33k/31k)", "data"),
    (1.0, LANE_FLOW, "SFT→RM→PPO\n三段式", "flow"),
    (2.6, LANE_ALG, "离线 DPO\n(隐式奖励)", "alg"),
    (2.6, LANE_DATA, "AI 偏好\n固定偏好对", "data"),
    (3.6, LANE_ALG, "在线 DPO 变体\n(Iterative/SPPO)", "alg"),
    (4.6, LANE_ALG, "GRPO · RLOO\n(无 critic, 组优势)", "alg"),
    (4.6, LANE_DATA, "可验证奖励 RLVR\n(规则/判题)", "data"),
    (6.0, LANE_ALG, "R1-Zero: 纯 RL\n涌现长链推理", "alg"),
    (6.0, LANE_DATA, "规则奖励\n(正确率+格式)", "data"),
    (6.0, LANE_FLOW, "R1 多阶段流水线\n(冷启动→RL→拒采→SFT→全场景RL)", "flow"),
    (7.4, LANE_ALG, "Dr. GRPO / DAPO\n(修正逐token裁剪)", "alg"),
    (7.4, LANE_FLOW, "Qwen3 思考模式\n(Long-CoT→融合SFT→通用RL)", "flow"),
    (8.8, LANE_ALG, "V3.2 统一混合 RL\n(推理+智能体+对齐)", "alg"),
    (8.8, LANE_DATA, "合成环境×1827\n生成式 RM", "data"),
    (8.8, LANE_FLOW, "RL 算力\n> 预训练 10%", "flow"),
    (10.2, LANE_ALG, "K2.5 PARL\n(1 模型→100 子智能体)", "alg"),
    (10.2, LANE_DATA, "统一奖励+预算控制\n多维度防 hacking", "data"),
    (11.6, LANE_ALG, "异步规模化 RL\n(slime 框架)", "alg"),
    (11.6, LANE_FLOW, "GLM-5 顺序 RL\n(推理→智能体→通用)", "flow"),
    (12.6, LANE_DATA, "K3 多域 RL\n持久 rollout", "data"),
    (12.6, LANE_FLOW, "GLM-5.3:\n只靠后训练缩放", "flow"),
]

colors = {"alg": "#4C72B0", "data": "#DD8452", "flow": "#55A868"}
for x, y, label, kind in events:
    ax.scatter([x], [y], s=70, color=colors[kind], zorder=3)
    if y == LANE_ALG:
        va, dy = "bottom", 0.14
    else:
        va, dy = "top", -0.14
    ax.text(x, y + dy, label, fontsize=8.2, ha="center", va=va,
            color="#222222", linespacing=1.3, zorder=4)

# 时间轴
ax.plot([0.4, 13.6], [-0.35, -0.35], color="#333333", lw=1.5)
for x, year in [(1.0, "2022"), (2.6, "2023"), (3.6, "2024·上"),
                (4.6, "2024·下"), (6.0, "2025·上"), (7.4, "2025·中"),
                (8.8, "2025·下"), (10.2, "2025·底"), (11.6, "2025·底"),
                (12.6, "2026")]:
    ax.plot([x, x], [-0.35, -0.5], color="#333333", lw=1)
    ax.text(x, -0.74, year, fontsize=9, ha="center", color="#333333")

# 阶段底色与标签（顶部独立条带）
for x0, x1, label in [(0.4, 2.3, "1.0 对齐时代"), (2.4, 5.2, "2.0 在线+可验证奖励"),
                      (5.4, 8.2, "3.0 推理时代"), (8.4, 13.6, "3.5 后训练规模化")]:
    ax.axvspan(x0, x1, ymin=0.0, ymax=0.86, color="#f2f2f2", zorder=0)
    ax.text((x0 + x1) / 2, 4.05, label, fontsize=11, ha="center",
            color="#777777", weight="bold")

ax.set_xlim(0.0, 14.0)
ax.set_ylim(-1.15, 4.35)
fig.tight_layout()
out = os.path.join(HERE, "演进图.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
