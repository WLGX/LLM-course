# -*- coding: utf-8 -*-
"""
4.5 小规模验证实验：微型编码器的架构消融（方法学演示）。

固定数据与预算，比较四个变体：
  BASE    BERT 经典：绝对位置嵌入 + 全注意力 + GELU-FFN(4d) + LayerNorm
  ROPE    RoPE 相对位置（其余同 BASE）
  GEGLU   GeGLU 门控 FFN（扩展率 8/3，其余同 ROPE）
  LOCAL   局部窗口 64 + 末层全局 + RoPE（其余同 GEGLU）

双指标：MLM 验证损失（代理指标） + 4.4 断句标点任务 macro-F1（应用指标）。
输出：results/mini_ablation.json + 图。
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_44 = os.path.join(os.path.dirname(HERE), "4_4", "data")
OUT_RES = os.path.join(HERE, "results")
os.makedirs(OUT_RES, exist_ok=True)

D = 256            # 隐藏宽
N_HEAD = 8
HEAD_DIM = D // N_HEAD
N_LAYER = 4
SEQ = 128
MASK_RATE = 0.30
PRETRAIN_STEPS = 3000
PRETRAIN_BATCH = 128
PROBE_STEPS = 300
PROBE_BATCH = 256
LOCAL_WIN = 64


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- 数据 ----------------
class CharVocab:
    def __init__(self, path, max_vocab=8000):
        counts = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for c in row["text"]:
                    counts[c] = counts.get(c, 0) + 1
        chars = sorted(counts, key=counts.get, reverse=True)[:max_vocab]
        self.c2i = {"[PAD]": 0, "[MASK]": 1, "[CLS]": 2, "[SEP]": 3, "[UNK]": 4}
        for c in chars:
            self.c2i[c] = len(self.c2i)
        self.i2c = {v: k for k, v in self.c2i.items()}

    def encode(self, text):
        return [self.c2i.get(c, 4) for c in text]

    def __len__(self):
        return len(self.c2i)


def load_label_vocab():
    with open(os.path.join(DATA_44, "label_vocab.json"), encoding="utf-8") as f:
        return json.load(f)


def load_corpus(path, max_chars):
    texts = []
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            texts.append(row["text"])
            total += len(row["text"])
            if total >= max_chars:
                break
    return texts


class MLMData:
    """从字流采样 (seq) 长度的 MLM 样本，30% mask，80/10/10。"""

    def __init__(self, texts, vocab, seq=SEQ):
        self.seq = seq
        self.vocab = vocab
        self.stream = [vocab.c2i["[CLS]"]]
        for t in texts:
            self.stream.extend(vocab.encode(t))
            self.stream.append(vocab.c2i["[SEP]"])
        self.n = len(self.stream) - seq - 1

    def batch(self, bsz, rng):
        ids = np.zeros((bsz, self.seq), dtype=np.int64)
        labs = np.full((bsz, self.seq), -100, dtype=np.int64)
        for b in range(bsz):
            s = rng.randint(0, self.n)
            ids[b] = self.stream[s:s + self.seq]
            # 随机掩码（除 CLS/SEP 特例：开头位置的 CLS 保留）
            for i in range(1, self.seq):
                if rng.random() < MASK_RATE:
                    r = rng.random()
                    if r < 0.8:
                        labs[b, i] = ids[b, i]
                        ids[b, i] = self.vocab.c2i["[MASK]"]
                    elif r < 0.9:
                        labs[b, i] = ids[b, i]
                        ids[b, i] = rng.randint(4, len(self.vocab.c2i))
                    else:
                        labs[b, i] = ids[b, i]
        return ids, labs


# ---------------- 模型 ----------------
def precompute_rope(hd, seq, theta=10000.0):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2).float() / hd))
    pos = torch.arange(seq).float()
    freqs = torch.outer(pos, inv)  # (seq, hd/2)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    """x: (B,H,T,hd)"""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, None, :, :], sin[None, None, :, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class Attention(nn.Module):
    def __init__(self, d, n_head, rope=False, bias=False):
        super().__init__()
        self.d, self.n_head, self.rope = d, n_head, rope
        self.hd = d // n_head
        self.qkv = nn.Linear(d, 3 * d, bias=bias)
        self.o = nn.Linear(d, d, bias=bias)

    def forward(self, x, mask=None, cos=None, sin=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hd).transpose(1, 2)
        if self.rope:
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        att = q @ k.transpose(-2, -1) / math.sqrt(self.hd)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[None, None]  # (T,T) -> (1,1,T,T) 对齐 (B,H,T,T)
            att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        return self.o(y.transpose(1, 2).reshape(B, T, self.d))


class FFN(nn.Module):
    def __init__(self, d, geglu=False, bias=False):
        super().__init__()
        self.geglu = geglu
        if geglu:
            self.w1 = nn.Linear(d, (8 * d) // 3, bias=bias)
            self.w2 = nn.Linear(d, (8 * d) // 3, bias=bias)
            self.w3 = nn.Linear((8 * d) // 3, d, bias=bias)
        else:
            self.w1 = nn.Linear(d, 4 * d, bias=bias)
            self.w2 = nn.Linear(4 * d, d, bias=bias)

    def forward(self, x):
        if self.geglu:
            return self.w3(F.gelu(self.w1(x)) * self.w2(x))
        return self.w2(F.gelu(self.w1(x)))


class Block(nn.Module):
    def __init__(self, d, n_head, rope, geglu, bias, ln):
        super().__init__()
        self.attn = Attention(d, n_head, rope=rope, bias=bias)
        self.ffn = FFN(d, geglu=geglu, bias=bias)
        self.ln1 = ln(d)
        self.ln2 = ln(d)

    def forward(self, x, mask, cos, sin):
        x = x + self.attn(self.ln1(x), mask, cos, sin)
        x = x + self.ffn(self.ln2(x))
        return x


class TinyBert(nn.Module):
    def __init__(self, vocab_size, d, n_head, n_layer, rope, geglu, local, bias, rms):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.abs_pos = None if rope else nn.Embedding(SEQ, d)
        ln = (lambda dd: RMSNorm(dd)) if rms else (lambda dd: nn.LayerNorm(dd))
        self.blocks = nn.ModuleList(
            [Block(d, n_head, rope, geglu, bias, ln) for _ in range(n_layer)])
        self.local = local
        self.final_norm = ln(d)
        if rope:
            cos, sin = precompute_rope(d // n_head, SEQ)
            self.register_buffer("cos", cos)
            self.register_buffer("sin", sin)
        self.head = nn.Linear(d, vocab_size, bias=False)  # 输出与嵌入共享

    def forward(self, ids):
        B, T = ids.shape
        x = self.emb(ids)
        if self.abs_pos is not None:
            x = x + self.abs_pos(torch.arange(T, device=ids.device))[None]
        full = torch.ones(T, T, dtype=torch.bool, device=ids.device)
        if self.local:
            local_mask = (torch.arange(T, device=ids.device)[:, None] -
                          torch.arange(T, device=ids.device)[None, :]).abs() <= LOCAL_WIN
        cos = self.cos[:T] if self.abs_pos is None else None
        sin = self.sin[:T] if self.abs_pos is None else None
        for i, blk in enumerate(self.blocks):
            m = local_mask if (self.local and i < len(self.blocks) - 1) else full
            x = blk(x, m, cos, sin)
        return self.final_norm(x)  # 返回隐藏态；MLM 侧自行套 self.head


# ---------------- 训练 ----------------
def mlm_step(model, ids, labs, opt, scaler=None):
    logits = model.head(model(ids))  # (B,T,V)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labs.view(-1),
                           ignore_index=-100)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss)


@torch.no_grad()
def mlm_valid_loss(model, data, vocab_size, rng, n_batch=16, bsz=64):
    model.eval()
    tot, cnt = 0.0, 0
    for _ in range(n_batch):
        ids, labs = data.batch(bsz, rng)
        ids = torch.tensor(ids, device=next(model.parameters()).device)
        labs = torch.tensor(labs, device=ids.device)
        logits = model.head(model(ids))
        loss = F.cross_entropy(logits.view(-1, vocab_size), labs.view(-1), ignore_index=-100)
        tot += float(loss) * bsz
        cnt += bsz
    model.train()
    return tot / cnt


def probe(model, train_rows, valid_rows, vocab, lab_vocab, device, steps=PROBE_STEPS,
          bsz=PROBE_BATCH):
    """冻结编码器，线性 head 做断句标点，返回 macro-F1 与训练细节。"""
    n_lab = len(lab_vocab)
    head = nn.Linear(D, n_lab).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    model.eval()

    def rows_to_batches(rows, rng, shuffle):
        ids_list, lab_list = [], []
        for r in rows:
            ids = [vocab.c2i["[CLS]"]] + vocab.encode(r["text"][:SEQ - 2]) + [vocab.c2i["[SEP]"]]
            labs = [-100] + [lab_vocab[l] for l in r["labels"].split()[:SEQ - 2]] + [-100]
            ids_list.append(torch.tensor(ids))
            lab_list.append(torch.tensor(labs))
        idx = list(range(len(ids_list)))
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), bsz):
            b = idx[s:s + bsz]
            ids = torch.nn.utils.rnn.pad_sequence([ids_list[i] for i in b], batch_first=True)
            labs = torch.nn.utils.rnn.pad_sequence([lab_list[i] for i in b], batch_first=True,
                                                   padding_value=-100)
            yield ids, labs

    rng = np.random.RandomState(0)
    train_rows_small = train_rows[:6000]
    for epoch in range(1):
        rng2 = np.random.RandomState(1)
        for ids, labs in rows_to_batches(train_rows_small, rng2, True):
            ids, labs = ids.to(device), labs.to(device)
            with torch.no_grad():
                h = model(ids)
            logits = head(h)
            loss = F.cross_entropy(logits.view(-1, n_lab), labs.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # 评估
    gold, pred = [], []
    with torch.no_grad():
        for ids, labs in rows_to_batches(valid_rows, np.random.RandomState(2), False):
            h = model(ids.to(device))
            p = head(h).argmax(-1).cpu()
            for b in range(ids.size(0)):
                for t in range(ids.size(1)):
                    if labs[b, t].item() != -100:
                        gold.append(labs[b, t].item())
                        pred.append(p[b, t].item())
    return f1_score(gold, pred, labels=list(range(n_lab)), average="macro", zero_division=0)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_variant(name, cfg, vocab, mlm_data, train_rows, valid_rows, lab_vocab, device, rng):
    torch.cuda.empty_cache()
    seed_all(42)
    model = TinyBert(len(vocab), D, N_HEAD, N_LAYER, **cfg).to(device)
    n_params = count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    log = []
    t0 = time.time()
    for step in range(PRETRAIN_STEPS):
        ids, labs = mlm_data.batch(PRETRAIN_BATCH, rng)
        loss = mlm_step(model, torch.tensor(ids, device=device),
                        torch.tensor(labs, device=device), opt)
        if step % 100 == 0 or step == PRETRAIN_STEPS - 1:
            vl = mlm_valid_loss(model, mlm_data, len(vocab), rng)
            log.append({"step": step, "train_loss": loss, "valid_loss": round(vl, 4)})
            print(f"  [{name}] step {step} loss {loss:.3f} valid {vl:.3f}", flush=True)
    pretrain_sec = time.time() - t0
    f1 = probe(model, train_rows, valid_rows, vocab, lab_vocab, device)
    print(f"  [{name}] probe macro-F1 = {f1:.4f}, params = {n_params/1e6:.2f}M, "
          f"{pretrain_sec:.0f}s", flush=True)
    return {"name": name, "config": cfg, "params": n_params, "final_valid_mlm": log[-1]["valid_loss"],
            "probe_f1": round(float(f1), 4), "pretrain_sec": pretrain_sec, "log": log}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    vocab = CharVocab(os.path.join(DATA_44, "train.jsonl"))
    print("vocab size:", len(vocab))
    lab_vocab = load_label_vocab()
    texts = load_corpus(os.path.join(DATA_44, "train.jsonl"), 2_000_000)
    rng = np.random.RandomState(42)
    mlm_data = MLMData(texts, vocab)
    print("mlm stream chars:", mlm_data.n + SEQ)

    train_rows, valid_rows = [], []
    with open(os.path.join(DATA_44, "train.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            train_rows.append(json.loads(line))
            if i > 20000:
                break
    with open(os.path.join(DATA_44, "valid.jsonl"), encoding="utf-8") as f:
        for i, line in enumerate(f):
            valid_rows.append(json.loads(line))
            if i > 2000:
                break

    variants = {
        "BASE": dict(rope=False, geglu=False, local=False, bias=True, rms=False),
        "ROPE": dict(rope=True, geglu=False, local=False, bias=True, rms=False),
        "GEGLU": dict(rope=True, geglu=True, local=False, bias=False, rms=True),
        "LOCAL": dict(rope=True, geglu=True, local=True, bias=False, rms=True),
    }
    out = {}
    res_path = os.path.join(OUT_RES, "mini_ablation.json")
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            out = json.load(f)
        print("resume: already have", list(out))
    for name, cfg in variants.items():
        if name in out:
            print(f"=== variant {name} === (skip, already in checkpoint)")
            continue
        print(f"=== variant {name} ===")
        out[name] = run_variant(name, cfg, vocab, mlm_data, train_rows, valid_rows,
                                lab_vocab, device, rng)
        # 每变体即时落盘，防止后续变体崩溃丢失已完成结果
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
    print("saved mini_ablation.json")


if __name__ == "__main__":
    main()
