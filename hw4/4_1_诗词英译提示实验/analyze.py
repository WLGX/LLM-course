# -*- coding: utf-8 -*-
"""hw4 诗词英译实验：汇总分析。

- 程序化格式检查（协议第 3 节）
- 单元级配对比较：改对 / 新增错误（协议第 5 节）
- 成本汇总（token、重试）
- 输出 results/汇总_{split}.md 与 results/汇总_{split}.json
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "results", "raw")
SCORE_DIR = os.path.join(HERE, "results", "scores")
VERSIONS = ["A", "B", "C"]
STATUS_VAL = {"完整": 1.0, "部分": 0.5, "缺失": 0.0, "错误": 0.0}


def word_count_en(s):
    return len(re.findall(r"[A-Za-z']+", s or ""))


def check_format(parsed, poem):
    viol = []
    if not isinstance(parsed, dict):
        return ["非法JSON"], 0
    if not isinstance(parsed.get("translation"), str):
        viol.append("缺translation")
        return viol, 0
    if not isinstance(parsed.get("notes"), list):
        viol.append("缺notes数组")
        return viol, 0
    n_lines = parsed["translation"].count("\n") + 1
    if n_lines != len(poem["lines"]):
        viol.append(f"行数不一致({n_lines}对{len(poem['lines'])})")
    if word_count_en(parsed["translation"]) > 120:
        viol.append("译文超120词")
    if len(parsed["notes"]) > 2:
        viol.append("notes超2条")
    for note in parsed["notes"]:
        if not isinstance(note, str) or word_count_en(note) > 30:
            viol.append("注释超30词或非字符串")
    # 含题名/多出诗句：译文里出现原诗没有的整行诗句较难程序化，此处只查题名
    title_chars = poem["title"].split("（")[0]
    if title_chars and title_chars in parsed["translation"]:
        viol.append("译文含题名")
    return viol, 1 if not viol else 0


def main():
    split = os.environ.get("HW4_SPLIT", "test")
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    poems = {p["id"]: p for p in corpus["poems"] if p["split"] == split}

    raw_path = os.path.join(RAW_DIR, f"raw_{split}.jsonl")
    score_path = os.path.join(SCORE_DIR, f"scores_{split}.jsonl")

    # 载入生成记录与评分
    gens = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            gens[(e["poem_id"], e["version"], e["rep"])] = e
    scores = {}
    with open(score_path, encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            scores[(e["poem_id"], e["version"], e["rep"])] = e

    # ---- 成本 ----
    cost = {v: {"input_tokens": 0, "output_tokens": 0, "calls": 0, "retry_calls": 0, "elapsed_s": 0.0}
            for v in VERSIONS}
    for (pid, ver, rep), e in gens.items():
        r = e.get("response") or {}
        c = cost[ver]
        c["calls"] += 1
        c["input_tokens"] += (r.get("usage") or {}).get("prompt_tokens", 0)
        c["output_tokens"] += (r.get("usage") or {}).get("completion_tokens", 0)
        c["elapsed_s"] += r.get("elapsed_s", 0)
        if r.get("error") or not e.get("parsed"):
            c["retry_calls"] += 1

    # ---- 格式与单元分 ----
    per = {}  # (pid, ver) -> {"units": {idx: [v1, v2]}, "overall": [..], "fmt_ok": [..], "viol": [...]}
    for pid, p in poems.items():
        for ver in VERSIONS:
            key = (pid, ver)
            per[key] = {"unit_vals": {}, "overall": [], "fmt_ok": [], "viol": []}
            for rep in (1, 2):
                g = gens.get((pid, ver, rep))
                s = scores.get((pid, ver, rep))
                if not g or not g.get("parsed"):
                    per[key]["fmt_ok"].append(0)
                    per[key]["viol"].append(["生成失败"])
                    continue
                viol, ok = check_format(g["parsed"], p)
                per[key]["fmt_ok"].append(ok)
                per[key]["viol"].append(viol)
                if s and s.get("score") and isinstance(s["score"], dict):
                    per[key]["overall"].append(s["score"].get("overall"))
                    units_by_text = {}
                    for u in s["score"].get("units", []):
                        units_by_text[u.get("unit", "")] = u
                    for i, u in enumerate(p["units"]):
                        ju = units_by_text.get(u["text"])
                        val = STATUS_VAL.get(ju.get("status"), 0.0) if ju else 0.0
                        per[key]["unit_vals"].setdefault(i, []).append(val)
                else:
                    per[key]["overall"].append(None)
                    for i in range(len(p["units"])):
                        per[key]["unit_vals"].setdefault(i, []).append(0.0)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    # ---- 单元级改对/新增错误 ----
    changes = {v: {"改对": [], "新增错误": [], "部分变化": []} for v in ("B", "C")}
    for pid, p in poems.items():
        base = per[(pid, "A")]
        for ver in ("B", "C"):
            tgt = per[(pid, ver)]
            for i, u in enumerate(p["units"]):
                a = mean(base["unit_vals"].get(i, [0.0]))
                b = mean(tgt["unit_vals"].get(i, [0.0]))
                lbl = f"{pid}-{p['title']}「{u['text']}」({u['category']})"
                if a is None or b is None:
                    continue
                if a < 0.75 and b >= 0.75:
                    changes[ver]["改对"].append((lbl, round(a, 2), round(b, 2)))
                elif a >= 0.75 and b < 0.75:
                    changes[ver]["新增错误"].append((lbl, round(a, 2), round(b, 2)))
                elif abs(b - a) >= 0.25:
                    changes[ver]["部分变化"].append((lbl, round(a, 2), round(b, 2)))

    # ---- 输出 ----
    out_md = [f"# hw4 诗词英译实验：自动汇总（{split} 集）\n"]
    out_md.append("## 成本（生成阶段）\n")
    out_md.append("| 版本 | 调用数 | 重试 | 输入token | 输出token | 平均每次输出token | 耗时(s) |")
    out_md.append("|---|---|---|---|---|---|---|")
    for ver in VERSIONS:
        c = cost[ver]
        out_md.append(f"| {ver} | {c['calls']} | {c['retry_calls']} | {c['input_tokens']} | "
                      f"{c['output_tokens']} | {c['output_tokens']/max(1,c['calls']):.1f} | {c['elapsed_s']:.1f} |")

    out_md.append("\n## 格式违规率\n")
    out_md.append("| 版本 | 格式合规 | 违规明细（示例） |")
    out_md.append("|---|---|---|")
    for ver in VERSIONS:
        oks = [per[(pid, ver)]["fmt_ok"] for pid in poems]
        total = sum(len(x) for x in oks)
        ok = sum(sum(x) for x in oks)
        viols = [v for pid in poems for v in per[(pid, ver)]["viol"] if v and v != ["生成失败"]]
        uniq = {}
        for v in viols:
            for t in v:
                uniq[t] = uniq.get(t, 0) + 1
        out_md.append(f"| {ver} | {ok}/{total} | {uniq if uniq else '无'} |")

    out_md.append("\n## 分版本单元得分与整体可读性\n")
    out_md.append("| 版本 | 单元均分(0-1) | 整体可读性(1-5) |")
    out_md.append("|---|---|---|")
    for ver in VERSIONS:
        uvals = [v for pid in poems for arr in per[(pid, ver)]["unit_vals"].values() for v in arr]
        ovals = [v for pid in poems for v in per[(pid, ver)]["overall"] if v is not None]
        out_md.append(f"| {ver} | {mean(uvals):.3f} | {mean(ovals):.2f} |")

    out_md.append("\n## 分诗 × 版本 单元均分\n")
    out_md.append("| 诗 | A | B | C |")
    out_md.append("|---|---|---|---|")
    for pid, p in poems.items():
        row = [pid]
        for ver in VERSIONS:
            uvals = [v for arr in per[(pid, ver)]["unit_vals"].values() for v in arr]
            row.append(f"{mean(uvals):.2f}")
        out_md.append("| " + " | ".join(row) + " |")

    out_md.append("\n## 单元级 改对 / 新增错误（相对 A）\n")
    for ver in ("B", "C"):
        out_md.append(f"### {ver} 相对 A\n")
        out_md.append(f"- 改对：{len(changes[ver]['改对'])} 个单元")
        for lbl, a, b in changes[ver]["改对"]:
            out_md.append(f"  - {lbl}: {a} -> {b}")
        out_md.append(f"- 新增错误：{len(changes[ver]['新增错误'])} 个单元")
        for lbl, a, b in changes[ver]["新增错误"]:
            out_md.append(f"  - {lbl}: {a} -> {b}")
        out_md.append(f"- 部分变化：{len(changes[ver]['部分变化'])} 个单元")
        for lbl, a, b in changes[ver]["部分变化"]:
            out_md.append(f"  - {lbl}: {a} -> {b}")

    md_path = os.path.join(HERE, "results", f"汇总_{split}.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_md) + "\n")
    # JSON 版供报告引用
    summary = {
        "cost": cost,
        "per_poem": {f"{pid}-{ver}": {
            "unit_mean": mean([v for arr in per[(pid, ver)]["unit_vals"].values() for v in arr]),
            "overall_mean": mean(per[(pid, ver)]["overall"]),
            "fmt_ok_rate": sum(per[(pid, ver)]["fmt_ok"]) / max(1, len(per[(pid, ver)]["fmt_ok"])),
        } for pid in poems for ver in VERSIONS},
        "changes": {v: {k: [(a, b) for (_, a, b) in lst] for k, lst in changes[v].items()} for v in ("B", "C")},
    }
    with open(os.path.join(HERE, "results", f"汇总_{split}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("wrote", md_path)


if __name__ == "__main__":
    main()
