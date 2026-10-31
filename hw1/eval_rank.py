# -*- coding: utf-8 -*-
"""
评测脚本二：受限候选集上的排序对比（KenLM vs Stupid Backoff）。

取测试集中长度 >= 3 词的句子，用种子 42 随机抽 150 个预测位置；
候选集 = train 前 100 高频词 + 真实目标词（去重）。
指标：Top-1 命中率、Top-5 命中率、平均倒数排名（MRR）。

注意：这是受限候选集实验，指标不是完整词表的预测准确率；
两种方法使用相同位置与候选集，因此可以公平比较排序行为。
"""
import os
import sys
import time
import json
import math
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from model_sb import StupidBackoff


def kenlm_logscores(model, words, candidate_list):
    """对同一位置，返回每个候选词在历史 words 下的 log10 条件概率。

    注意：BaseScore(in_state, w, out_state) 返回分数并把新状态写入
    out_state，必须交换 st/out 才能推进历史。
    """
    import kenlm
    st = kenlm.State()
    model.BeginSentenceWrite(st)
    out = kenlm.State()
    for h in words:
        model.BaseScore(st, h, out)
        st, out = out, st
    return [model.BaseScore(st, w, out) for w in candidate_list]


def sb_logscores(sb, words, candidate_list):
    ctx = words[-(sb.max_order - 1):] if sb.max_order > 1 else []
    return [sb.logscore(w, ctx) for w in candidate_list]


def rank_of(scores, gold_idx):
    return 1 + sum(1 for s in scores if s > scores[gold_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kenlm-dir", default=r"F:/LLM-course/hw1/models/kenlm")
    ap.add_argument("--sb-db", default=r"F:/LLM-course/hw1/models/ngrams.db")
    ap.add_argument("--sb-max-order", type=int, default=5)
    ap.add_argument("--test", default=os.path.join(HERE, "data", "test.txt"))
    ap.add_argument("--n-pos", type=int, default=150)
    ap.add_argument("--cand-top", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ranking.json"))
    args = ap.parse_args()

    import kenlm
    from kenlm_utils import load

    lines = [l.split() for l in open(args.test, encoding="utf-8")
             if l.strip() and len(l.split()) >= 3]
    rng = random.Random(args.seed)
    positions = []
    while len(positions) < args.n_pos:
        sent = rng.choice(lines)
        j = rng.randint(1, len(sent) - 1)      # 目标词索引 1..len-1，历史至少 1 词
        positions.append((sent, j))

    sb = StupidBackoff(args.sb_db, max_order=args.sb_max_order)
    top100 = [w for w, _ in sorted(sb.unigram.items(), key=lambda x: -x[1])[:args.cand_top]]

    results = {"positions": [], "methods": {}}
    for method in ["KenLM", "SB"]:
        if method == "KenLM":
            # kenlm 不支持 1 阶模型，从 2 阶起
            models = {o: m for o in range(2, args.sb_max_order + 1)
                      if (m := load(kenlm, args.kenlm_dir, o)) is not None}
        else:
            models = {o: StupidBackoff(args.sb_db, max_order=o)
                      for o in range(1, args.sb_max_order + 1)}
        for order in sorted(models):
            top1 = top5 = 0
            mrr = 0.0
            t0 = time.time()
            for sent, j in positions:
                gold = sent[j]
                cands = list(top100)
                if gold not in cands:
                    cands.append(gold)
                gold_idx = cands.index(gold)
                if method == "KenLM":
                    scores = kenlm_logscores(models[order], sent[:j], cands)
                else:
                    scores = sb_logscores(models[order], sent[:j], cands)
                r = rank_of(scores, gold_idx)
                if r == 1:
                    top1 += 1
                if r <= 5:
                    top5 += 1
                mrr += 1.0 / r
            n = len(positions)
            results["methods"].setdefault(method, []).append({
                "order": order, "top1": top1 / n, "top5": top5 / n,
                "mrr": mrr / n, "seconds": round(time.time() - t0, 1),
            })
            print(f"{method} order {order}: top1={top1/n:.3f} top5={top5/n:.3f} "
                  f"mrr={mrr/n:.3f}  {time.time()-t0:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
