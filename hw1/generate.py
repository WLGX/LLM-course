# -*- coding: utf-8 -*-
"""
用 n-gram 模型续写作业指定开头：
"在阳光明媚的五月，我们学校胜利召开了……"
按语料词表切分为：在 阳光 明媚 的 五月 ， 我们 学校 胜利 召开 了

实验组：
  A) 阶数扫描：KenLM 与 Stupid Backoff 各在阶数 1..max_order 下，
     固定 T=1.0、top-k=20、随机种子 42 续写；
  B) 参数扫描：阶数 5 下比较 T=0.7 / T=1.0 / T=1.3 / 贪婪，两种模型。

输出：results/generation.json（完整词序列），并打印可贴进报告的表格行。
"""
import os
import json
import math
import random
import argparse
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_sb import StupidBackoff

PROMPT = ["在", "阳光", "明媚", "的", "五月", "，", "我们", "学校", "胜利", "召开", "了"]
MAX_NEW = 80


class KenLMGen:
    def __init__(self, model, top_words):
        self.model = model
        self.top_words = top_words

    def logscore(self, word, context):
        st = kenlm.State()
        self.model.BeginSentenceWrite(st)
        out = kenlm.State()
        for h in context[-200:]:          # 只回填足够长的历史
            self.model.BaseScore(st, h, out)
            st, out = out, st
        return self.model.BaseScore(st, word, out)


def gen_kenlm(model, context, temperature, top_k, rng):
    import kenlm
    words = []
    ctx = list(context)
    st = kenlm.State()
    model.model.BeginSentenceWrite(st)
    out = kenlm.State()
    for h in ctx:
        model.model.BaseScore(st, h, out)
        st, out = out, st
    for _ in range(MAX_NEW):
        cands = list(model.top_words) + ["</s>"]
        scores = []
        for w in cands:
            lp = model.model.BaseScore(st, w, out)   # log10
            scores.append(10 ** (lp / temperature))
        idx = sorted(range(len(cands)), key=lambda i: -scores[i])[:top_k]
        if top_k == 1 or rng is None:                # 贪婪
            pick = cands[idx[0]]
        else:
            pool = [(cands[i], scores[i]) for i in idx]
            total = sum(s for _, s in pool)
            r = rng.random() * total
            acc = 0.0
            pick = pool[-1][0]
            for w, s in pool:
                acc += s
                if r <= acc:
                    pick = w
                    break
        if pick == "</s>":
            return words, "eos"
        words.append(pick)
        model.model.BaseScore(st, pick, out)
        st, out = out, st
    return words, "maxlen"


def gen_sb(sb, context, temperature, top_k, rng):
    words = []
    ctx = list(context)
    for _ in range(MAX_NEW):
        cands = list(sb.top_words) + ["</s>"]
        scores = []
        for w in cands:
            ls = sb.logscore(w, ctx)
            scores.append(math.exp(ls / temperature) if ls != float("-inf") else 0.0)
        idx = sorted(range(len(cands)), key=lambda i: -scores[i])[:top_k]
        if top_k == 1 or rng is None:                # 贪婪
            pick = cands[idx[0]]
        else:
            pool = [(cands[i], scores[i]) for i in idx]
            total = sum(s for _, s in pool)
            r = rng.random() * total
            acc = 0.0
            pick = pool[-1][0]
            for w, s in pool:
                acc += s
                if r <= acc:
                    pick = w
                    break
        if pick == "</s>":
            return words, "eos"
        words.append(pick)
        ctx.append(pick)
        if sb.max_order > 1 and len(ctx) > sb.max_order - 1:
            ctx = ctx[-(sb.max_order - 1):]
    return words, "maxlen"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kenlm-dir", default=r"F:/LLM-course/hw1/models/kenlm")
    ap.add_argument("--sb-db", default=r"F:/LLM-course/hw1/models/ngrams.db")
    ap.add_argument("--sb-max-order", type=int, default=5)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "generation.json"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sb = StupidBackoff(args.sb_db, max_order=args.sb_max_order)
    top_words = sb.top_words   # 两种模型共用同一候选集（train 前 200 高频词）
    # 阶数扫描时 SB 按阶重建实例（max_order 决定回退深度）
    sb_by_order = {o: StupidBackoff(args.sb_db, max_order=o)
                   for o in range(1, args.sb_max_order + 1)}

    results = {"prompt": PROMPT, "runs": []}

    def add_run(method, order, decode, words, end):
        row = {
            "method": method, "order": order, "decode": decode,
            "n_new": len(words), "end": end, "words": words,
        }
        results["runs"].append(row)
        text = "".join(words)
        print(f"{method}\t{order}\t{decode}\t{len(words)}\t{end}\t{text[:60]}")

    # ---- A) 阶数扫描 ----
    rngA = random.Random(42)
    import kenlm
    from kenlm_utils import load
    kenlm_models = {}
    # kenlm 不支持 1 阶模型：从 2 阶加载到 10 阶
    for order in range(2, 11):
        m = load(kenlm, args.kenlm_dir, order)
        if m is not None:
            kenlm_models[order] = KenLMGen(m, top_words)
    for order in range(1, args.sb_max_order + 1):
        t0 = time.time()
        words, end = gen_sb(sb_by_order[order], PROMPT, 1.0, 20, rngA)
        add_run("SB", order, "T=1.0,k=20", words, end)
        print(f"    SB gen {time.time()-t0:.1f}s")
    for order in range(2, 11):
        if order not in kenlm_models:
            continue
        t0 = time.time()
        words, end = gen_kenlm(kenlm_models[order], PROMPT, 1.0, 20, rngA)
        add_run("KenLM", order, "T=1.0,k=20", words, end)
        print(f"    KenLM gen {time.time()-t0:.1f}s")

    # ---- B) 参数扫描（阶数 5）----
    for decode, temp, top_k, greedy in [
        ("T=0.7,k=20", 0.7, 20, False),
        ("T=1.0,k=20", 1.0, 20, False),
        ("T=1.3,k=20", 1.3, 20, False),
        ("greedy", 1.0, 1, True),
    ]:
        rngB = random.Random(42)
        if greedy:
            rngB = None
        words, end = gen_sb(sb, PROMPT, temp, top_k, rngB)
        add_run("SB", 5, decode, words, end)
        if 5 in kenlm_models:
            words, end = gen_kenlm(kenlm_models[5], PROMPT, temp, top_k, rngB)
            add_run("KenLM", 5, decode, words, end)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
