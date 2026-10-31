# -*- coding: utf-8 -*-
"""hw4 诗词英译实验：按类别（时间/否定/例外/术语/关系）汇总单元得分。

输出 results/类别汇总_{split}.md，供实验报告 5.3 节逐项核对使用。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCORE_DIR = os.path.join(HERE, "results", "scores")
VERSIONS = ["A", "B", "C"]
STATUS_VAL = {"完整": 1.0, "部分": 0.5, "缺失": 0.0, "错误": 0.0}


def main():
    split = os.environ.get("HW4_SPLIT", "test")
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    poems = {p["id"]: p for p in corpus["poems"] if p["split"] == split}

    with open(os.path.join(SCORE_DIR, f"scores_{split}.jsonl"), encoding="utf-8") as f:
        scores = {}
        for line in f:
            e = json.loads(line)
            scores[(e["poem_id"], e["version"], e["rep"])] = e

    # (pid, ver, unit_idx) -> [vals]
    vals = {}
    for pid, p in poems.items():
        for ver in VERSIONS:
            for i, u in enumerate(p["units"]):
                arr = []
                for rep in (1, 2):
                    s = scores.get((pid, ver, rep)) or {}
                    sc = s.get("score")
                    if sc and isinstance(sc, dict):
                        units_by_text = {x.get("unit", ""): x for x in sc.get("units", [])}
                        ju = units_by_text.get(u["text"])
                        arr.append(STATUS_VAL.get(ju.get("status"), 0.0) if ju else 0.0)
                    else:
                        arr.append(0.0)
                vals[(pid, ver, i)] = arr

    def mean(xs):
        return sum(xs) / len(xs) if xs else None

    cats = {}
    for p in poems.values():
        for i, u in enumerate(p["units"]):
            cats.setdefault(u["category"], []).append((p["id"], i))

    lines = [f"# hw4 诗词英译实验：类别汇总（{split} 集）\n"]
    lines.append("| 类别 | A 均分 | B 均分 | C 均分 | 单元数 |")
    lines.append("|---|---|---|---|---|")
    cat_rows = []
    for cat, items in sorted(cats.items()):
        row = {}
        for ver in VERSIONS:
            arr = [v for (pid, i) in items for v in vals[(pid, ver, i)]]
            row[ver] = mean(arr)
        cat_rows.append((cat, row, len(items)))
        lines.append(f"| {cat} | {row['A']:.3f} | {row['B']:.3f} | {row['C']:.3f} | {len(items)} |")
    lines.append("")
    # 明细：每类别列出单元与三版本得分
    for cat, row, n in cat_rows:
        lines.append(f"## 类别：{cat}\n")
        lines.append("| 诗 | 单元 | A | B | C |")
        lines.append("|---|---|---|---|---|")
        for pid, i in cats[cat]:
            u = poems[pid]["units"][i]
            a = mean(vals[(pid, "A", i)])
            b = mean(vals[(pid, "B", i)])
            c = mean(vals[(pid, "C", i)])
            lines.append(f"| {pid} {poems[pid]['title']} | {u['text']} | {a:.2f} | {b:.2f} | {c:.2f} |")
        lines.append("")

    out = os.path.join(HERE, "results", f"类别汇总_{split}.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
