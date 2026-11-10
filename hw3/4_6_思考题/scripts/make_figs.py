# -*- coding: utf-8 -*-
"""P115/P123 出图：读 results/*.json，生成 figs/p115_bench.png 与 figs/p123_recipes.png。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figs")
os.makedirs(FIG, exist_ok=True)

FONT = "/mnt/c/Windows/Fonts/msyh.ttc"
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


def fig_p115():
    d = json.load(open(os.path.join(OUT, "micro_bench.json"), encoding="utf-8"))
    grid, sweep = d["grid"], d["batch_sweep"]
    shapes = sorted({r["S"] for r in grid})
    names = ["MHA_full", "MHA_causal", "GQA4_full"]
    colors = {"MHA_full": "#4472C4", "MHA_causal": "#ED7D31", "GQA4_full": "#70AD47"}
    labels = {"MHA_full": "MHA 完整方阵", "MHA_causal": "MHA 因果掩码（只屏蔽不跳过）",
              "GQA4_full": "GQA（KV 头 8→2）"}

    import numpy as np
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.6))
    # A: wall time
    ax = axes[0][0]
    x = np.arange(len(shapes))
    w = 0.26
    for j, nm in enumerate(names):
        ts = [next(r["ms"] for r in grid if r["S"] == s and r["variant"] == nm) for s in shapes]
        ax.bar(x + (j - 1) * w, ts, w, label=labels[nm], color=colors[nm])
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={s}\nB={4096//s}" for s in shapes], fontsize=8)
    ax.set_ylabel("墙钟时间 (ms)")
    ax.set_title("(a) 三种实现 × 5 个形状（B·s=4096）")
    ax.legend(fontsize=8)
    # B: effective TFLOPS
    ax = axes[0][1]
    for j, nm in enumerate(names):
        ts = [next(r["tflops_eff"] for r in grid if r["S"] == s and r["variant"] == nm) for s in shapes]
        ax.bar(x + (j - 1) * w, ts, w, label=labels[nm], color=colors[nm])
    ax.axhline(d["peak"]["gemm_fp32_tflops"], ls="--", c="gray",
               label=f"实测 GEMM 峰值 {d['peak']['gemm_fp32_tflops']} TFLOPS")
    ax.set_xticks(x)
    ax.set_xticklabels([f"s={s}\nB={4096//s}" for s in shapes], fontsize=8)
    ax.set_ylabel("实际达到 TFLOPS（执行口径）")
    ax.set_title("(b) 距离实测峰值的差距")
    ax.legend(fontsize=8)
    # C: batch sweep + recheck
    ax = axes[1][0]
    bs = [r["B"] for r in sweep]
    us = [r["ms_per_token"] for r in sweep]
    ax.plot(bs, us, "o-", color="#4472C4", label="首测")
    if "recheck" in d:
        rch = d["recheck"]
        trials = [r for r in rch if r["shape"] == "s=400,B=32"]
        if trials:
            ms_avg = np.mean([t["ms"] for t in trials])
            ax.plot(32, ms_avg * 1e3 / (32 * 400), "x", ms=11, color="#ED7D31",
                    label=f"B=32 复查 ×{len(trials)}（{ms_avg:.0f} ms）")
    ax.set_xlabel("batch（s=400 固定，每窗口 FLOPs 相同）")
    ax.set_ylabel("每 token 耗时 (µs)")
    ax.set_title("(c) 同 FLOPs → 墙钟时间可差数倍")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    for b, u in zip(bs, us):
        ax.annotate(f"{u}", (b, u), textcoords="offset points", xytext=(0, 6), fontsize=8)
    # D: SDPA causal — 掩码 vs 真正跳过
    ax = axes[1][1]
    sdp = d.get("sdpa", [])
    if sdp:
        sdp_shapes = sorted({r["S"] for r in sdp})
        backends = ["math", "efficient", "flash"]
        bcolors = {"math": "#A5A5A5", "efficient": "#70AD47", "flash": "#ED7D31"}
        w = 0.26
        x = np.arange(len(sdp_shapes))
        for j, bn in enumerate(backends):
            ts = []
            for s in sdp_shapes:
                rows = [r for r in sdp if r["S"] == s and r["backend"] == bn and r["causal"]]
                ts.append(rows[0]["ms"] if rows and rows[0]["ms"] else None)
            ax.bar(x + (j - 1) * w, [t if t else 0 for t in ts], w,
                   label=bn + ("(因果)" if bn == "math" else "(因果)"), color=bcolors[bn])
        ax.set_xticks(x)
        ax.set_xticklabels([f"s={s}\nB={4096//s}" for s in sdp_shapes], fontsize=8)
        ax.set_ylabel("单次 SDPA 耗时 (ms)")
        ax.set_title("(d) 同一因果数学量：math 只掩码，flash 真跳过")
        ax.legend(fontsize=8)
    fig.suptitle("P115 实测：减少计算量 ≠ 运行变快（RTX 3050 Laptop，WSL2，全部真实运行）",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(FIG, "p115_bench.png"), dpi=150)
    plt.close(fig)
    print("saved p115_bench.png")


def fig_p123():
    d = json.load(open(os.path.join(OUT, "recipe_results.json"), encoding="utf-8"))
    results = d["results"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    colors = plt.cm.tab10.colors
    for i, r in enumerate(results):
        ax.plot(r["step_marks"], r["loss_curve"], label=r["name"], color=colors[i % 10], lw=1.3)
    ax.set_xlabel("已处理 token 数（所有配方预算相同 = 相同 FLOPs）")
    ax.set_ylabel("训练损失（EMA）")
    ax.set_title("(a) 相同预算下 9 种配方的训练轨迹")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax = axes[1]
    names = [r["name"] for r in results]
    ppls = [r["final_valid_ppl"] if r["crashed_at_step"] is None else 0 for r in results]
    bars = ax.barh(range(len(names)), ppls, color=[colors[i % 10] for i in range(len(names))])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    for b, p in zip(bars, ppls):
        if p > 0:
            ax.annotate(f"{p:.1f}", (p, b.get_y() + b.get_height() / 2),
                        textcoords="offset points", xytext=(3, -4), fontsize=8)
        else:
            ax.annotate("发散", (0.1, b.get_y() + b.get_height() / 2), fontsize=8, color="#C00000")
    ax.set_xlabel("最终验证困惑度（越低越好；0 = 发散）")
    ax.set_title("(b) 相同 FLOPs 下的最终验证困惑度")
    ax.grid(alpha=0.3, axis="x")
    fig.suptitle("P123 实测：训练配方在相同总 token（= 相同 6ND FLOPs）下的影响（微型字符 GPT，人民日报语料）",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(FIG, "p123_recipes.png"), dpi=150)
    plt.close(fig)
    print("saved p123_recipes.png")


if __name__ == "__main__":
    fig_p115()
    fig_p123()
