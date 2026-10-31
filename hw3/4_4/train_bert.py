# -*- coding: utf-8 -*-
"""
4.4 任务一：微调 bert-base-chinese 做断句标点（标点恢复式字级序列标注）。

输入：hw3/4_4/data/{train,valid,test}.jsonl（text 不含主标点，labels 为每字后插入的标点）
输出：hw3/4_4/models/punct_best/（最优模型）
      hw3/4_4/results/{train_log.jsonl, valid_metrics.json, test_report.txt, samples.txt, run_info.json}

用法：
  python train_bert.py --epochs 2 --batch 16 --train-limit 40000 --lr 3e-5
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForTokenClassification, AutoTokenizer
from sklearn.metrics import classification_report, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
OUT_MODEL = os.path.join(HERE, "models", "punct_best")
OUT_RES = os.path.join(HERE, "results")
MODEL_NAME = "google-bert/bert-base-chinese"
MAX_CHARS = 128  # 与数据准备的块长一致；BERT 512 token 上限内留有裕量


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PunctDataset(Dataset):
    def __init__(self, rows, tokenizer, vocab, max_chars=MAX_CHARS):
        self.rows = rows
        self.tok = tokenizer
        self.vocab = vocab
        self.max_chars = max_chars

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        text = self.rows[i]["text"][: self.max_chars]
        labels = self.rows[i]["labels"].split()[: self.max_chars]
        enc = self.tok(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
            return_attention_mask=False,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]  # 与 ids 等长；(start,end) 为字符跨度
        tok_labels = []
        prev_span = None
        for j, (s, e) in enumerate(offsets):
            if j == 0 or j == len(ids) - 1 or (s, e) == (0, 0):
                tok_labels.append(-100)  # [CLS]/[SEP]/pad
                prev_span = None
                continue
            if prev_span is not None and (s, e) == prev_span:
                tok_labels.append(-100)  # 同一字符的后续子词
                continue
            tok_labels.append(self.vocab[labels[s]])
            prev_span = (s, e)
        return {"ids": torch.tensor(ids, dtype=torch.long),
                "labels": torch.tensor(tok_labels, dtype=torch.long)}


def collate(batch):
    ids = pad_sequence([b["ids"] for b in batch], batch_first=True, padding_value=0)
    labs = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
    mask = ids.ne(0)
    return ids, mask, labs


def load_rows(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def compute_metrics(tok, model, rows, vocab, device, batch_size=32, tag=""):
    model.eval()
    gold, pred = [], []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            ds = PunctDataset(rows[start:start + batch_size], tok, vocab)
            ids, mask, labs = collate(list(ds))
            out = model(ids.to(device), attention_mask=mask.to(device))
            logits = out.logits  # (B, T, C)
            p = logits.argmax(-1)
            for b in range(ids.size(0)):
                for t in range(ids.size(1)):
                    if mask[b, t] and labs[b, t].item() != -100:
                        gold.append(labs[b, t].item())
                        pred.append(p[b, t].item())
    id2lab = {v: k for k, v in vocab.items()}
    labels = list(range(len(vocab)))
    names = [id2lab[i] for i in labels]
    report = classification_report(gold, pred, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    macro = f1_score(gold, pred, labels=labels, average="macro", zero_division=0)
    return report, macro, gold, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--train-limit", type=int, default=40000)
    ap.add_argument("--valid-limit", type=int, default=1000)
    ap.add_argument("--test-limit", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)

    with open(os.path.join(DATA_DIR, "label_vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    n_labels = len(vocab)
    print("labels:", vocab, flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=n_labels, id2label={v: k for k, v in vocab.items()},
        label2id=vocab)
    model.to(device)

    train_rows = load_rows(os.path.join(DATA_DIR, "train.jsonl"), args.train_limit)
    valid_rows = load_rows(os.path.join(DATA_DIR, "valid.jsonl"), args.valid_limit)
    print(f"train={len(train_rows)} valid={len(valid_rows)}", flush=True)

    train_ds = PunctDataset(train_rows, tok, vocab)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                          num_workers=0, drop_last=True)

    # 类别权重（按训练集比例，截断防止极端值；忽略 -100 特殊标签）
    all_labs = np.concatenate([b["labels"].numpy() for b in train_ds])
    all_labs = all_labs[all_labs >= 0]
    counts = np.bincount(all_labs, minlength=n_labels).astype(float)
    counts[counts == 0] = 1.0
    weights = counts.sum() / (n_labels * counts)
    weights = np.clip(weights, 0.1, 10.0)
    weights[0] *= 0.5  # O 类适当降权
    print("class weights:", dict(zip(vocab.keys(), weights.round(2))), flush=True)
    crit = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device),
                                     ignore_index=-100)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = len(train_dl)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup)

    def lr_at(step):
        if step < warmup_steps:
            return args.lr * (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return args.lr * (1 - p)

    os.makedirs(OUT_RES, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MODEL), exist_ok=True)
    log_path = os.path.join(OUT_RES, "train_log.jsonl")
    logf = open(log_path, "w", encoding="utf-8")

    best_macro = -1.0
    t0 = time.time()
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for ids, mask, labs in train_dl:
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            ids, mask, labs = ids.to(device), mask.to(device), labs.to(device)
            out = model(ids, attention_mask=mask, labels=labs)
            loss = out.loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if step % 50 == 0:
                logf.write(json.dumps({"step": step, "epoch": epoch,
                                       "loss": float(loss), "lr": lr,
                                       "sec": time.time() - t0}) + "\n")
                logf.flush()
                print(f"step {step}/{total_steps} loss {loss:.4f} lr {lr:.2e} "
                      f"{(time.time()-t0):.0f}s", flush=True)
            step += 1
        report, macro, _, _ = compute_metrics(tok, model, valid_rows, vocab, device,
                                              tag=f"epoch{epoch}")
        print(f"[epoch {epoch}] valid macro-F1={macro:.4f}", flush=True)
        if macro > best_macro:
            best_macro = macro
            model.save_pretrained(OUT_MODEL)
            tok.save_pretrained(OUT_MODEL)
            print("saved best", flush=True)

    train_sec = time.time() - t0
    logf.close()

    # 最终测试（用最优模型）
    model = AutoModelForTokenClassification.from_pretrained(OUT_MODEL).to(device)
    test_rows = load_rows(os.path.join(DATA_DIR, "test.jsonl"), args.test_limit)
    report, macro, gold, pred = compute_metrics(tok, model, test_rows, vocab, device,
                                                tag="test")
    with open(os.path.join(OUT_RES, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(gold, pred, target_names=list(vocab.keys()),
                                      labels=list(range(n_labels)), zero_division=0))
    metrics = {
        "test_macro_f1": macro,
        "per_class": {k: {"precision": v["precision"], "recall": v["recall"],
                          "f1-score": v["f1-score"], "support": v["support"]}
                      for k, v in report.items() if isinstance(v, dict)},
        "train_seconds": train_sec,
        "steps": step,
        "samples_per_sec": step * args.batch / max(train_sec, 1e-6),
        "device": device,
        "gpu_max_mem_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if device == "cuda" else None,
        "train_rows": len(train_rows),
        "epochs": args.epochs,
        "batch": args.batch,
        "lr": args.lr,
    }
    with open(os.path.join(OUT_RES, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=1)
    print(json.dumps(metrics, ensure_ascii=False, indent=1), flush=True)

    # 样例：验证集前若干条的重建效果
    id2lab = {v: k for k, v in vocab.items()}
    model.eval()
    samples = []
    with torch.no_grad():
        for row in valid_rows[:5]:
            text = row["text"][:MAX_CHARS]
            gold = row["labels"].split()[:MAX_CHARS]
            ds = PunctDataset([{"text": text, "labels": " ".join(gold)}], tok, vocab)
            ids, mask, labs = collate(list(ds))
            out = model(ids.to(device), attention_mask=mask.to(device))
            p = out.logits.argmax(-1)[0]
            pred = []
            for t in range(ids.size(1)):
                if mask[0, t] and labs[0, t].item() != -100:
                    pred.append(id2lab[p[t].item()])
            def rebuild(chars, ls):
                s = ""
                for c, l in zip(chars, ls):
                    s += c + (l if l != "O" else "")
                return s
            samples.append({"gold": rebuild(text, gold), "pred": rebuild(text, pred)})
    with open(os.path.join(OUT_RES, "samples.txt"), "w", encoding="utf-8") as f:
        for s in samples:
            f.write("GOLD: " + s["gold"] + "\nPRED: " + s["pred"] + "\n\n")
    print("done")


if __name__ == "__main__":
    main()
