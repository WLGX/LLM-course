# -*- coding: utf-8 -*-
"""
汇总所有结果 JSON，画图并生成 Piazza 帖子 Markdown。

输入（results/ 与 data/）：
  clean_stats.json, count_times.json, kenlm_train.json, ppl.json,
  ranking.json, generation.json
输出：
  results/fig_*.png  （中文标题用英文以防字体问题？——matplotlib 默认不含中文字体，
                       图中文字用英文标注，正文用中文说明）
  results/hw1_post.md
"""
import os
import json
import math
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATA = os.path.join(HERE, "data")


def load(name, default=None):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("fig ->", path)


def main():
    clean = load(os.path.join(DATA, "clean_stats.json"), {})
    ct = load(os.path.join(DATA, "count_times.json"), {})
    kt = load("kenlm_train.json", [])
    ppl = load("ppl.json", {})
    rank = load("ranking.json", {})
    gen = load("generation.json", {})

    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "figs"), exist_ok=True)

    # ---- 图1：训练时间对比（SB 逐阶 vs KenLM lmplz 逐阶）----
    sb_orders = [r["order"] for r in ct.get("orders", [])]
    sb_secs = [r["seconds"] for r in ct.get("orders", [])]
    kl_orders = [r["order"] for r in kt if "lmplz_seconds" in r]
    kl_secs = [r["lmplz_seconds"] for r in kt if "lmplz_seconds" in r]
    if sb_orders:
        plt.figure(figsize=(7, 4))
        plt.plot(sb_orders, sb_secs, "o-", label="Stupid Backoff counting (SQLite)")
        if kl_orders:
            plt.plot(kl_orders, kl_secs, "s-", label="KenLM lmplz (MKN)")
        plt.xlabel("n-gram order")
        plt.ylabel("training wall time (s)")
        plt.title("Training time vs order")
        plt.legend()
        plt.grid(alpha=0.3)
        fig(os.path.join(RESULTS, "figs", "fig1_time.png"))

    # ---- 图2：测试集 perplexity 随阶数变化 ----
    if ppl.get("orders"):
        xs = [r["order"] for r in ppl["orders"]]
        ys = [r["ppl"] for r in ppl["orders"]]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, "o-")
        plt.xlabel("order")
        plt.ylabel("test perplexity")
        plt.title("KenLM test perplexity vs order")
        plt.grid(alpha=0.3)
        fig(os.path.join(RESULTS, "figs", "fig2_ppl.png"))

    # ---- 图3：每阶不同 n-gram 组合数 ----
    if sb_orders:
        plt.figure(figsize=(7, 4))
        plt.plot(sb_orders, [r["distinct"] for r in ct["orders"]], "o-")
        plt.yscale("log")
        plt.xlabel("order")
        plt.ylabel("distinct n-grams (log)")
        plt.title("Distinct n-gram counts vs order")
        plt.grid(alpha=0.3)
        fig(os.path.join(RESULTS, "figs", "fig3_distinct.png"))

    # ---- 图4：模型体积 ----
    if kt:
        xs = [r["order"] for r in kt if "binary_mib" in r]
        ys = [r["binary_mib"] for r in kt if "binary_mib" in r]
        plt.figure(figsize=(7, 4))
        plt.plot(xs, ys, "o-")
        plt.xlabel("order")
        plt.ylabel("binary model size (MiB)")
        plt.title("KenLM binary size vs order")
        plt.grid(alpha=0.3)
        fig(os.path.join(RESULTS, "figs", "fig4_size.png"))

    # ---- 图5：受限候选集 Top-1/Top-5 命中率 ----
    if rank.get("methods"):
        plt.figure(figsize=(7, 4))
        for method, rows in rank["methods"].items():
            xs = [r["order"] for r in rows]
            plt.plot(xs, [r["top1"] for r in rows], "o-", label=f"{method} top-1")
            plt.plot(xs, [r["top5"] for r in rows], "s--", label=f"{method} top-5")
        plt.xlabel("order")
        plt.ylabel("hit rate (constrained candidates)")
        plt.title("Ranking accuracy vs order")
        plt.legend()
        plt.grid(alpha=0.3)
        fig(os.path.join(RESULTS, "figs", "fig5_rank.png"))

    # ---- 组装数据表（供帖子正文引用）----
    out = []
    out.append("# n-gram 作业数据表（自动生成，勿手改数字）\n")

    # 训练时间表
    sb_by_order = {r["order"]: r for r in ct.get("orders", [])}
    kl_by_order = {r["order"]: r for r in kt}
    out.append("## 表1：训练时间与模型体积\n")
    out.append("| 阶 | SB计数(SQLite) s | KenLM lmplz s | build_binary s "
               "| ARPA MiB | bin MiB | 该阶 distinct n-gram |")
    out.append("|---|---|---|---|---|---|---|")
    for o in sorted(kl_by_order):
        r = kl_by_order[o]
        sb = sb_by_order.get(o, {})
        sb_s = f'{sb["seconds"]:.2f}' if sb else "—"
        dis = r.get("counts", {})
        if isinstance(dis, dict):
            v = dis.get(o) if o in dis else dis.get(str(o), None)
            dstr = str(v) if v is not None else "—"
        else:
            dstr = "—"
        out.append(f"| {o} | {sb_s} | {r.get('lmplz_seconds', 0):.2f} "
                   f"| {r.get('build_binary_seconds', 0):.2f} | "
                   f"{r.get('arpa_mib', 0):.0f} | {r.get('binary_mib', 0):.0f} | {dstr} |")

    # PPL 表
    out.append("\n## 表2：测试集 perplexity\n")
    out.append("| 阶 | PPL | 评测耗时 s |")
    out.append("|---|---|---|")
    for r in ppl.get("orders", []):
        out.append(f"| {r['order']} | {r['ppl']:.1f} | {r.get('eval_seconds', 0):.1f} |")

    # 排序表
    out.append("\n## 表3：受限候选集排序\n")
    out.append("| 方法 | 阶 | Top-1 | Top-5 | MRR |")
    out.append("|---|---|---|---|---|")
    for method, rows in rank.get("methods", {}).items():
        for r in rows:
            out.append(f"| {method} | {r['order']} | {r['top1']:.3f} "
                       f"| {r['top5']:.3f} | {r['mrr']:.3f} |")

    # 生成表
    out.append("\n## 表4：续写结果\n")
    out.append("| 方法 | 阶 | 解码 | 新词数 | 结束 | 续写 |")
    out.append("|---|---|---|---|---|---|")
    for r in gen.get("runs", []):
        text = "".join(r["words"])
        out.append(f"| {r['method']} | {r['order']} | {r['decode']} "
                   f"| {r['n_new']} | {r['end']} | {text} |")

    with open(os.path.join(RESULTS, "post_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")
    print("tables ->", os.path.join(RESULTS, "post_tables.md"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
