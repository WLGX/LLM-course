# -*- coding: utf-8 -*-
"""
4.4 任务二诊断：长文 macro-F1 低于任务一的原因归因。
检查两类假设：
  H1 块边界拼接伪影——128 字块的拼接点金标为 O，模型按连续文本预测标点，边界附近错误集中；
  H2 类别退化——罕见类（；：？！""…）在长窗口下进一步退化，拖低 macro 平均。
输出 results/diag_long.json + 打印分类报告。
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer

from decode_long import (DATA_DIR, MODEL_DIR, doc_windows, encode_text,
                         load_vocab, merge_overlap, predict_windows)


def build_docs_with_boundaries(test_path, n_docs=2, min_len=3000):
    rows = []
    with open(test_path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    docs = []
    cur_t, cur_l, bounds, pos = "", [], [], 0
    for r in rows:
        cur_t += r["text"]
        cur_l += r["labels"].split()
        pos += len(r["text"])
        bounds.append(pos)
        if len(cur_t) >= min_len and len(docs) < n_docs:
            docs.append((cur_t[:min_len], cur_l[:min_len],
                         [b for b in bounds if b < min_len]))
            cur_t, cur_l, bounds, pos = "", [], [], 0
        if len(docs) >= n_docs:
            break
    return docs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    vocab, id2lab = load_vocab()
    n_classes = len(vocab)
    classes = list(range(n_classes))

    docs = build_docs_with_boundaries(os.path.join(DATA_DIR, "test.jsonl"))
    out = {}
    for di, (text, gold, bounds) in enumerate(docs):
        gold_ids = [vocab[g] for g in gold]
        wins = doc_windows(len(text))
        prepared = []
        for s, e in wins:
            ids, char_pos = encode_text(tok, text[s:e])
            prepared.append((s, e, ids, char_pos))
        groups = [prepared[i:i + 8] for i in range(0, len(prepared), 8)]
        wp = predict_windows(model, tok, id2lab, groups, device)
        merged = merge_overlap(wp, len(text), n_classes)
        pred = merged.argmax(-1).tolist()

        report = classification_report(gold_ids, pred, labels=classes,
                                       target_names=[id2lab[i] for i in classes],
                                       zero_division=0, output_dict=True)
        # 错误位置与块边界的距离关系
        err_pos = [i for i in range(len(gold_ids)) if gold_ids[i] != pred[i]]
        near = sum(1 for p in err_pos if min(abs(p - b) for b in bounds) <= 5)
        out[f"doc{di}"] = {
            "len": len(text), "n_boundaries": len(bounds),
            "n_err": len(err_pos), "n_err_near_boundary": near,
            "share_near_boundary": round(near / len(err_pos), 4) if err_pos else 0,
            "per_class_f1": {k: round(v["f1-score"], 3) for k, v in report.items()
                             if isinstance(v, dict)},
            "macro_f1": round(f1_score(gold_ids, pred, labels=classes,
                                       average="macro", zero_division=0), 4),
            "weighted_f1": round(f1_score(gold_ids, pred, labels=classes,
                                          average="weighted", zero_division=0), 4),
        }
        print(f"doc {di}: errors {len(err_pos)}/{len(gold_ids)} "
              f"({100*near/max(1,len(err_pos)):.0f}% 距块边界≤5字), "
              f"macro {out[f'doc{di}']['macro_f1']:.3f}, "
              f"weighted {out[f'doc{di}']['weighted_f1']:.3f}")
        print(classification_report(gold_ids, pred, labels=classes,
                                    target_names=[id2lab[i] for i in classes],
                                    zero_division=0, digits=2))
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "results", "diag_long.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved diag_long.json")


if __name__ == "__main__":
    main()
