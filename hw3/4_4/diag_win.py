# -*- coding: utf-8 -*-
"""
4.4 任务二补充诊断：窗口长度对照实验。
假设：长文 macro-F1 掉到 ~0.2 的主因是 400 字窗口超出训练分布（训练块 ≤128 字，
绝对位置嵌入只在 0-127 见过）；若是，128 字窗口解码的 F1 应显著回升。
对照：WINDOW ∈ {128, 200, 300, 400}（步长 = 窗口 × 0.75），同一方法 B。
"""
import json
import os

import torch
from sklearn.metrics import f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

import decode_long as DL

HERE = os.path.dirname(os.path.abspath(__file__))


def build_docs(test_path, n_docs=2, min_len=3000):
    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    docs = []
    cur_t, cur_l = "", []
    for r in rows:
        cur_t += r["text"]
        cur_l += r["labels"].split()
        if len(cur_t) >= min_len and len(docs) < n_docs:
            docs.append((cur_t[:min_len], cur_l[:min_len]))
            cur_t, cur_l = "", []
        if len(docs) >= n_docs:
            break
    return docs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(DL.MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(DL.MODEL_DIR).to(device)
    model.eval()
    vocab, id2lab = DL.load_vocab()
    classes = list(range(len(vocab)))
    docs = build_docs(os.path.join(DL.DATA_DIR, "test.jsonl"))

    out = {}
    for win, stride in [(128, 96), (200, 150), (300, 225), (400, 300)]:
        DL.WINDOW, DL.STRIDE = win, stride
        row = {}
        for di, (text, gold) in enumerate(docs):
            gold_ids = [vocab[g] for g in gold]
            pred, t, n = DL.run_method_B(model, tok, id2lab, text, device, 8, len(vocab))
            macro = f1_score(gold_ids, pred.tolist(), labels=classes, average="macro",
                             zero_division=0)
            w = f1_score(gold_ids, pred.tolist(), labels=classes, average="weighted",
                         zero_division=0)
            row[f"doc{di}"] = {"window": win, "n_windows": n, "t_s": round(t, 3),
                               "macro_f1": round(macro, 4), "weighted_f1": round(w, 4)}
            print(f"win {win} doc {di}: windows {n}, {t:.2f}s, "
                  f"macro {macro:.3f}, weighted {w:.3f}", flush=True)
        out[f"win{win}"] = row
    with open(os.path.join(HERE, "results", "diag_window.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved diag_window.json")


if __name__ == "__main__":
    main()
