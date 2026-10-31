# -*- coding: utf-8 -*-
"""
评测脚本一：KenLM 模型在完整测试集上的 perplexity。

PPL = exp( -Σ log10 p(w_i|h_i) * ln10 / N )
N 为测试集词数 + 句数（每句一个 </s> 预测位置），p(</s>|全文) 计入。
用 kenlm score(line + ' </s>', bos=False, eos=False) 逐句求和，
即第一词按一元概率、最后按 </s> 条件概率计，等价于逐词条件概率之和。

同时记录每阶评测耗时与词表外词比例。
"""
import os
import sys
import time
import json
import math
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kenlm-dir", default=r"F:/LLM-course/hw1/models/kenlm")
    ap.add_argument("--sb-db", default=r"F:/LLM-course/hw1/models/ngrams.db")
    ap.add_argument("--test", default=os.path.join(HERE, "data", "test.txt"))
    ap.add_argument("--orders", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "ppl.json"))
    args = ap.parse_args()

    import kenlm
    sys.path.insert(0, HERE)
    from kenlm_utils import model_path

    lines = open(args.test, encoding="utf-8").read().splitlines()
    lines = [l for l in lines if l.strip()]
    n_sent = len(lines)
    n_tok = sum(len(l.split()) for l in lines)

    results = {"n_sentences": n_sent, "n_tokens": n_tok,
               "N_with_eos": n_sent + n_tok, "orders": []}
    for order in args.orders:
        if order == 1:
            # kenlm 不支持 1 阶模型（ARPA/binary 均拒绝），1 阶用 SQLite 计数的
            # unigram MLE：把 </s> 作为一元词加入（count = train 句数），
            # 与高阶模型一样计入句末预测位置，保证口径可比。
            import sqlite3
            clean = json.load(open(os.path.join(HERE, "data", "clean_stats.json"),
                                   encoding="utf-8"))
            n_sent_train = clean["split"]["train"]["sentences"]
            conn = sqlite3.connect(args.sb_db)
            uni = dict(conn.execute(
                "SELECT w1, c FROM n1 WHERE w1 != '<s>'").fetchall())
            conn.close()
            uni["</s>"] = n_sent_train
            total = sum(uni.values())
            t0 = time.time()
            logp = 0.0
            oov = 0
            for l in lines:
                for w in l.split():
                    c = uni.get(w, 0)
                    if c:
                        logp += math.log10(c / total)
                    else:
                        oov += 1      # MLE 概率为 0，跳过并单独报告
                logp += math.log10(uni["</s>"] / total)
            secs = time.time() - t0
            ppl = 10 ** (-logp / (n_sent + n_tok))
            results["orders"].append({
                "order": 1, "ppl": ppl, "log10p": logp,
                "eval_seconds": round(secs, 2), "oov_tokens": oov,
                "note": "unigram MLE from SQLite counts (kenlm refuses order-1)",
            })
            print(f"order 1: ppl={ppl:.1f} (oov={oov}) eval {secs:.1f}s", flush=True)
            continue
        path = model_path(args.kenlm_dir, order)
        if path is None:
            continue
        m = kenlm.Model(path)
        t0 = time.time()
        logp = 0.0
        for l in lines:
            logp += m.score(l + " </s>", bos=False, eos=False)
        secs = time.time() - t0
        ppl = 10 ** (-logp / (n_sent + n_tok))
        results["orders"].append({
            "order": order, "ppl": ppl, "log10p": logp,
            "eval_seconds": round(secs, 2),
        })
        print(f"order {order}: ppl={ppl:.1f}  eval {secs:.1f}s", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
