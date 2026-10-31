# -*- coding: utf-8 -*-
"""
4.4 任务二：长文断句标点的 batch 解码方案。

朴素方案 A（课件描述的基线）：滑窗 400 字，只保留前 300 字；在预测的句号后断开
作为新起点（下一个窗口起点依赖本窗口输出 -> 只能逐窗串行，无法并行）。

batch 方案 B：保持同样的窗口几何（400 字、步长 300），但把"句号后断开"的串行依赖
替换为固定步长 + 重叠区概率融合，从而所有窗口相互独立，可装入同一 batch 并行前向。

测量：吞吐（字/秒、窗口/秒）、显存峰值、与 A 的一致性、两方案相对 gold 的 F1。
"""
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models", "punct_best")
DATA_DIR = os.path.join(HERE, "data")
OUT_RES = os.path.join(HERE, "results")

WINDOW = 400      # 窗口字符数（与课件基线一致）
STRIDE = 300      # 步长（与课件基线一致）
END_PUNCT = {"。", "？", "！", "…"}


def load_vocab():
    with open(os.path.join(DATA_DIR, "label_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    return vocab, {v: k for k, v in vocab.items()}


def encode_text(tok, text):
    enc = tok(text, add_special_tokens=True, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    char_pos = []  # 每个字符对应的 token 下标（首子词）
    prev_span = None
    for j, (s, e) in enumerate(offsets):
        if (s, e) == (0, 0):
            continue
        if prev_span is not None and (s, e) == prev_span:
            continue
        char_pos.append(j)
        prev_span = (s, e)
    return torch.tensor(ids, dtype=torch.long), char_pos


@torch.no_grad()
def predict_windows(model, tok, id2lab, batches, device):
    """batches: list of (start, end, ids_tensor, char_pos)；返回 {start: (probs, labels)}"""
    out = {}
    for group in batches:
        ids_list = [g[2] for g in group]
        L = max(len(x) for x in ids_list)
        ids = torch.zeros(len(group), L, dtype=torch.long, device=device)
        mask = torch.zeros(len(group), L, dtype=torch.long, device=device)
        for i, x in enumerate(ids_list):
            ids[i, :len(x)] = x.to(device)
            mask[i, :len(x)] = 1
        logits = model(ids, attention_mask=mask).logits
        probs = torch.softmax(logits, dim=-1)
        C = probs.shape[-1]
        for i, (start, end, _, char_pos) in enumerate(group):
            n_chars = end - start
            p = np.zeros((n_chars, C))
            sel = [char_pos[j] for j in range(min(len(char_pos), n_chars))]
            p[:len(sel)] = probs[i, sel].cpu().numpy()
            labels = [id2lab[int(x)] for x in p.argmax(-1)]
            out[start] = (p, labels)
    return out


def doc_windows(doc_len):
    """固定步长窗口：返回 [(start, end)]，全部覆盖文档。"""
    starts = list(range(0, max(1, doc_len - WINDOW + 1), STRIDE))
    if not starts or starts[-1] + WINDOW < doc_len:
        starts.append(max(0, doc_len - WINDOW))
    return [(s, min(s + WINDOW, doc_len)) for s in starts]


def merge_overlap(windows_pred, doc_len, n_classes):
    """重叠区概率平均 -> 每字标签。"""
    acc = np.zeros((doc_len, n_classes))
    cnt = np.zeros(doc_len)
    for start, (probs, _) in windows_pred.items():
        w = probs.shape[0]
        acc[start:start + w] += probs
        cnt[start:start + w] += 1
    cnt[cnt == 0] = 1
    return acc / cnt[:, None]


@torch.no_grad()
def run_method_A(model, tok, id2lab, doc, device, n_classes):
    """朴素串行：滑窗 400，取前 300，在预测句号后断开作为新起点。"""
    t0 = time.time()
    n_windows = 0
    s = 0
    L = len(doc)
    final_probs = np.zeros((L, n_classes))
    covered = np.zeros(L)
    while s < L:
        end = min(s + WINDOW, L)
        ids, char_pos = encode_text(tok, doc[s:end])
        ids = ids.unsqueeze(0).to(device)
        mask = torch.ones_like(ids)
        logits = model(ids, attention_mask=mask).logits
        probs_t = torch.softmax(logits, dim=-1)[0]
        C = probs_t.shape[-1]
        probs = np.zeros((end - s, C))
        sel = [char_pos[j] for j in range(min(len(char_pos), end - s))]
        probs[:len(sel)] = probs_t[sel].cpu().numpy()
        labels = [id2lab[int(x)] for x in probs.argmax(-1)]
        keep = min(STRIDE, end - s)
        final_probs[s:s + keep] += probs[:keep]
        covered[s:s + keep] += 1
        n_windows += 1
        # 在 [s+keep, end) 内找第一个预测句尾，作为新起点；否则 s+=STRIDE
        nxt = None
        for j in range(keep, end - s):
            if labels[j] in END_PUNCT:
                nxt = s + j + 1
                break
        s = nxt if nxt is not None else s + STRIDE
        if s >= L:
            break
    torch.cuda.synchronize() if device == "cuda" else None
    covered[covered == 0] = 1
    pred = final_probs.argmax(-1)
    return pred, time.time() - t0, n_windows


def run_method_B(model, tok, id2lab, doc, device, batch, n_classes):
    """batch 解码：固定步长窗口全部并行，重叠区概率平均。"""
    t0 = time.time()
    wins = doc_windows(len(doc))
    prepared = []
    for s, e in wins:
        ids, char_pos = encode_text(tok, doc[s:e])
        prepared.append((s, e, ids, char_pos))
    groups = [prepared[i:i + batch] for i in range(0, len(prepared), batch)]
    windows_pred = predict_windows(model, tok, id2lab, groups, device)
    torch.cuda.synchronize() if device == "cuda" else None
    t = time.time() - t0
    merged = merge_overlap(windows_pred, len(doc), n_classes)
    return merged.argmax(-1), t, len(wins)


def label_f1(pred_labels, gold_labels, classes):
    from sklearn.metrics import f1_score
    return f1_score(gold_labels, pred_labels, labels=classes, average="macro", zero_division=0)


def label_f1_weighted(pred_labels, gold_labels, classes):
    from sklearn.metrics import f1_score
    return f1_score(gold_labels, pred_labels, labels=classes, average="weighted", zero_division=0)


def class_f1(pred_labels, gold_labels, cls_id):
    from sklearn.metrics import f1_score
    return f1_score([1 if x == cls_id else 0 for x in gold_labels],
                    [1 if x == cls_id else 0 for x in pred_labels], zero_division=0)


def build_long_docs(test_path, n_docs=8, min_len=3000):
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
    print("device:", device)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()
    vocab, id2lab = load_vocab()
    classes = list(range(len(vocab)))
    n_classes = len(vocab)

    docs = build_long_docs(os.path.join(DATA_DIR, "test.jsonl"))
    print(f"docs: {len(docs)}, lengths: {[len(t) for t, _ in docs]}")

    # CUDA 预热：消除首次调用的 cuDNN/cuBLAS autotune 对计时的影响
    warm_ids = torch.zeros((4, 128), dtype=torch.long, device=device)
    warm_mask = torch.ones_like(warm_ids)
    for _ in range(2):
        model(warm_ids, attention_mask=warm_mask)
    torch.cuda.synchronize() if device == "cuda" else None
    print("warmup done")

    results = {"docs": [], "batch_sweep": []}
    # 方案 A（串行基线）与方案 B（batch=8）质量/耗时对比
    o_id = vocab.get("O", 0)
    for di, (text, gold) in enumerate(docs):
        predA, tA, nA = run_method_A(model, tok, id2lab, text, device, n_classes)
        predB, tB, nB = run_method_B(model, tok, id2lab, text, device, 8, n_classes)
        gold_ids = [vocab[g] for g in gold]
        f1A = label_f1(predA.tolist(), gold_ids, classes)
        f1B = label_f1(predB.tolist(), gold_ids, classes)
        wA = label_f1_weighted(predA.tolist(), gold_ids, classes)
        wB = label_f1_weighted(predB.tolist(), gold_ids, classes)
        oA = class_f1(predA.tolist(), gold_ids, o_id)
        oB = class_f1(predB.tolist(), gold_ids, o_id)
        agree = float((predA == predB).mean())
        results["docs"].append({
            "doc": di, "len": len(text), "windows_A": nA, "windows_B": nB,
            "t_A_s": round(tA, 3), "t_B_s": round(tB, 3),
            "chars_s_A": round(len(text) / tA), "chars_s_B": round(len(text) / tB),
            "f1_A": round(f1A, 4), "f1_B": round(f1B, 4),
            "f1w_A": round(wA, 4), "f1w_B": round(wB, 4),
            "f1_O_A": round(oA, 4), "f1_O_B": round(oB, 4),
            "agree_AB": round(agree, 4),
        })
        print(f"doc {di}: A {tA:.2f}s ({len(text)/tA:.0f}字/s, F1 {f1A:.3f}) | "
              f"B8 {tB:.2f}s ({len(text)/tB:.0f}字/s, F1 {f1B:.3f}) | agree {agree:.3f}")

    # batch 大小扫描（方法 B，仅吞吐，用第 0 篇文档）
    text0 = docs[0][0]
    for b in [1, 2, 4, 8, 16, 32]:
        try:
            pred, t, n = run_method_B(model, tok, id2lab, text0, device, b, n_classes)
            mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0
            results["batch_sweep"].append(
                {"batch": b, "t_s": round(t, 3), "chars_s": round(len(text0) / t),
                 "gpu_mem_MB": round(mem, 1)})
            print(f"batch {b}: {t:.3f}s, {len(text0)/t:.0f}字/s, mem {mem:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            results["batch_sweep"].append({"batch": b, "oom": True})
            print(f"batch {b}: OOM")
            torch.cuda.empty_cache()

    os.makedirs(OUT_RES, exist_ok=True)
    with open(os.path.join(OUT_RES, "task2_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("saved task2_results.json")


if __name__ == "__main__":
    main()
