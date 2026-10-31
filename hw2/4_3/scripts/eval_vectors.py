# -*- coding: utf-8 -*-
"""
4.3 统一评测脚本：三个实现（gensim / 原版C / PyTorch SGNS）用同一组探针评测。
探针与 4_2 报告口径一致：近邻、相似度、doesnt_match、类比、OOV。
"""
import json
import os
import pickle
import numpy as np
from gensim.models import KeyedVectors

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # 4_3/
GENSIM_MODEL = os.path.join(ROOT, "..", "4_2", "models", "w2v_sg_d100.model")
C_VEC = os.path.join(ROOT, "results", "c_sg100")
TORCH_IDX2VEC = os.path.join(ROOT, "data_pt", "idx2vec.dat")
TORCH_IDX2WORD = os.path.join(ROOT, "data_pt", "idx2word.dat")
OUT = os.path.join(ROOT, "results", "impl_compare_eval.json")

PROBES_NEAR = ["中国", "经济"]
PROBES_SIM = [("北京", "上海"), ("足球", "体育"), ("中国", "美国")]
PROBE_ODD = ["北京", "上海", "天津", "足球"]
PROBE_ANALOGY = dict(positive=["北京", "美国"], negative=["中国"])
PROBES_OOV = ["深度学习", "神经网络", "微信", "智能手机"]


def probe(kv):
    out = {}
    out["近邻"] = {w: [x for x, _ in kv.most_similar(w, topn=5)] for w in PROBES_NEAR}
    out["相似度"] = {f"{a}_{b}": round(float(kv.similarity(a, b)), 4) for a, b in PROBES_SIM}
    out["doesnt_match"] = kv.doesnt_match(PROBE_ODD)
    out["类比_中国北京_美国"] = [x for x, _ in kv.most_similar(topn=3, **PROBE_ANALOGY)]
    oov = {}
    for w in PROBES_OOV:
        try:
            kv[w]
            oov[w] = "在词表"
        except KeyError:
            oov[w] = "KeyError(不在词表)"
    out["OOV"] = oov
    return out


def load_torch():
    """PyTorch SGNS 输出 -> gensim KeyedVectors（写临时文本向量再读入）"""
    vecs = pickle.load(open(TORCH_IDX2VEC, "rb"))
    idx2word = pickle.load(open(TORCH_IDX2WORD, "rb"))
    assert vecs.shape[0] == len(idx2word)
    tmp = os.path.join(ROOT, "results", "torch_vectors.txt")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(f"{len(idx2word)} {vecs.shape[1]}\n")
        for i, w in enumerate(idx2word):
            if "\t" in w or " " in w or w.strip() != w:
                continue                    # word2vec 文本格式不允许空白/边界空格
            f.write(w + " " + " ".join(f"{x:.6f}" for x in vecs[i]) + "\n")
    return KeyedVectors.load_word2vec_format(tmp)


def main():
    report = {}

    from gensim.models import Word2Vec
    kv_g = Word2Vec.load(GENSIM_MODEL).wv
    report["gensim_sg_d100"] = probe(kv_g)

    kv_c = KeyedVectors.load_word2vec_format(C_VEC, binary=True)
    report["c_original_sg100"] = probe(kv_c)

    if os.path.exists(TORCH_IDX2VEC) and os.path.exists(TORCH_IDX2WORD):
        kv_t = load_torch()
        report["torch_sgns_e100"] = probe(kv_t)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("评测完成 ->", OUT)
    for name, r in report.items():
        print(f"\n== {name} ==")
        print("  中国近邻:", r["近邻"]["中国"][:5])
        print("  经济近邻:", r["近邻"]["经济"][:5])
        print("  类比:", r["类比_中国北京_美国"])
        print("  doesnt_match:", r["doesnt_match"])
        print("  OOV:", {k: v for k, v in r["OOV"].items()})


if __name__ == "__main__":
    main()
