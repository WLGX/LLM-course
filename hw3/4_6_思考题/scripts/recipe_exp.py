# -*- coding: utf-8 -*-
"""P123 实测：相同总 token 预算（= 相同 6ND 主项 FLOPs）下，训练配方的影响有多大。

任务：人民日报 1998 清洗语料（hw3/4_4 数据）上的字符级下一字预测（微型 GPT）。
固定预算：总训练窗口 B × steps = 1,048,576（每窗 64 token，总 token = 67.1M；对同一模型即相同总 FLOPs）。
9 种配方只改一项/一组设置，记录训练损失轨迹、最终验证损失与墙钟时间。
注：4GB 卡（WSL2 与 Windows 桌面共享）放不下 B=1024 的整图，采用两项数值透明的显存措施——
① 词表头损失按位置分块计算（head_loss，chunk=8），② 训练时块级梯度重计算（checkpoint）。
两者均不改变数学结果（分块仅求和顺序不同、重计算逐位一致），全部 9 种配方共用同一条代码路径。
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
DATA = os.path.normpath(os.path.join(ROOT, "..", "4_4", "data", "train.jsonl"))

DEV = torch.device("cuda")
SEQ = 64
D = 64
NLAYER = 2
NHEAD = 4
BUDGET = 1_048_576  # 总 token 预算（所有配方相同）


def load_corpus(n_chars=1_300_000, n_valid=100_000):
    chunks, valid_chunks, n = [], [], 0
    with open(DATA, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj["text"]
            if n < n_chars:
                chunks.append(t)
                n += len(t)
            elif sum(len(x) for x in valid_chunks) < n_valid:
                valid_chunks.append(t)
            else:
                break
    train_text = "".join(chunks)[:n_chars]
    valid_text = "".join(valid_chunks)[:n_valid]
    vocab = sorted(set(train_text) | set(valid_text))
    c2i = {c: i for i, c in enumerate(vocab)}
    return train_text, valid_text, c2i


class MiniGPT(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Embedding(SEQ, D)
        self.blocks = nn.ModuleList([self._block() for _ in range(NLAYER)])
        self.ln = nn.LayerNorm(D)
        self.head = nn.Linear(D, V, bias=False)

    def _block(self):
        return nn.ModuleDict({
            "ln1": nn.LayerNorm(D),
            "attn": nn.MultiheadAttention(D, NHEAD, batch_first=True),
            "ln2": nn.LayerNorm(D),
            "fc1": nn.Linear(D, 4 * D),
            "fc2": nn.Linear(4 * D, D),
        })

    def forward(self, ids, dropout=0.1):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        mask = torch.triu(torch.ones(S, S, device=ids.device, dtype=torch.bool), 1)
        for b in self.blocks:
            def blk(x, b=b):
                h = b["attn"](b["ln1"](x), b["ln1"](x), b["ln1"](x),
                              attn_mask=mask, need_weights=False)[0]
                x = x + F.dropout(h, dropout, self.training)
                x = x + F.dropout(b["fc2"](F.gelu(b["fc1"](b["ln2"](x)))), dropout, self.training)
                return x
            # 训练时块级重计算：数值透明（重算前向逐位一致），显存峰值大幅下降，
            # 代价是块计算约 +30% 墙钟。B=1024 时 4GB 卡（与 Windows 桌面共享）放不下整图。
            if self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return self.ln(x)  # 只到隐藏表示 (B,S,D)；词表头分块计算，避免 (B,S,V) 大张量


def head_loss(head, h, yb, V, label_smoothing=0.0, chunk=8):
    """按位置分块计算词表头损失（head 逐位置独立，分块与整块数学等价，仅求和顺序不同）。

    B=1024 时整块 logits (1024,64,4566,fp32) ≈ 1.2GB，加 log_softmax 中间量超 4GB 显存；
    分块后每块 (B,8,V) ≈ 0.15GB。返回标量损失（带计算图，训练路径）；验证路径在 no_grad 下使用。
    """
    B, S, _ = h.shape
    tot, cnt = 0.0, 0
    for s0 in range(0, S, chunk):
        logits = head(h[:, s0:s0 + chunk])          # (B,c,V)
        logp = F.log_softmax(logits, dim=-1)
        yc = yb[:, s0:s0 + chunk]
        if label_smoothing > 0:
            smooth = label_smoothing / V
            target = torch.full_like(logp, smooth)
            target.scatter_(-1, yc.unsqueeze(-1), 1 - label_smoothing + smooth)
            tot = tot + -(target * logp).sum()
        else:
            tot = tot + F.nll_loss(logp.reshape(-1, V), yc.reshape(-1),
                                   reduction="sum")
        cnt += yc.numel()
    return tot / cnt


def make_batches(text, c2i, B):
    """固定顺序切窗（stride=SEQ），每轮随机起点洗牌由固定 seed 控制。"""
    ids = [c2i.get(c, 0) for c in text]
    wins = [ids[i:i + SEQ + 1] for i in range(0, len(ids) - SEQ, SEQ)]
    data = []
    for w in wins:
        if len(w) == SEQ + 1:
            data.append((torch.tensor(w[:-1], dtype=torch.long),
                         torch.tensor(w[1:], dtype=torch.long)))
    return data


def run_recipe(name, text, valid, c2i, B, steps, lr, warmup, schedule,
               dropout, label_smoothing, seed=42, log_every=128):
    g = torch.Generator().manual_seed(seed)
    V = len(c2i)
    model = MiniGPT(V).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    data = make_batches(text, c2i, B)
    vdata = make_batches(valid, c2i, 128)
    vb = []
    for i in range(0, min(400, len(vdata)), 128):
        chunk = vdata[i:i + 128]
        vb.append((torch.stack([c[0] for c in chunk]).to(DEV),
                   torch.stack([c[1] for c in chunk]).to(DEV)))
    t0 = time.time()
    losses, valid_ppls, step_marks = [], [], []
    ema = None
    n_update = 0
    crashed = None
    total_steps = steps
    for step in range(steps):
        idx = torch.randint(0, len(data), (B,), generator=g)
        xb = torch.stack([data[i][0] for i in idx]).to(DEV)
        yb = torch.stack([data[i][1] for i in idx]).to(DEV)
        h = model(xb, dropout=dropout)   # (B,S,D) 隐藏表示
        loss = head_loss(model.head, h, yb, V, label_smoothing)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        # 学习率日程
        if schedule == "cosine":
            if step < warmup:
                lr_now = lr * (step + 1) / warmup
            else:
                p = (step - warmup) / max(1, total_steps - warmup)
                lr_now = lr * 0.5 * (1 + math.cos(math.pi * p))
        else:  # constant
            lr_now = lr
        for pg in opt.param_groups:
            pg["lr"] = lr_now
        if not math.isfinite(loss.item()):
            crashed = step
            break
        ema = loss.item() if ema is None else 0.9 * ema + 0.1 * loss.item()
        n_update += 1
        if (step + 1) % log_every == 0:
            losses.append(round(ema, 4))
            step_marks.append((step + 1) * B)
            with torch.no_grad():
                model.eval()
                tot, cnt = 0.0, 0
                for xv, yv in vb[:64]:
                    hv = model(xv, dropout=0.0)
                    tot += head_loss(model.head, hv, yv, V, 0.0).item() * yv.numel()
                    cnt += yv.numel()
                model.train()
            valid_ppls.append(round(math.exp(tot / cnt), 2))
    # 最终全验证集评估
    with torch.no_grad():
        model.eval()
        tot, cnt = 0.0, 0
        for xv, yv in vb:
            hv = model(xv, dropout=0.0)
            tot += head_loss(model.head, hv, yv, V, 0.0).item() * yv.numel()
            cnt += yv.numel()
        model.train()
    final_ppl = math.exp(tot / cnt)
    wall = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    flops = 6 * n_params * (n_update * B * SEQ)  # 6ND 主项（实际执行）
    rec = {"name": name, "B": B, "steps": steps, "tokens": n_update * B * SEQ,
           "lr": lr, "warmup": warmup, "schedule": schedule,
           "dropout": dropout, "label_smoothing": label_smoothing,
           "final_valid_ppl": round(final_ppl, 2),
           "final_train_loss": round(ema, 4) if ema is not None else None,
           "wall_s": round(wall, 1), "params": n_params,
           "flops_6ND": flops, "crashed_at_step": crashed,
           "loss_curve": losses, "valid_ppl_curve": valid_ppls,
           "step_marks": step_marks}
    print(f"[{name}] ppl={rec['final_valid_ppl']:8.2f} "
          f"train_loss={rec['final_train_loss']} wall={wall:5.1f}s"
          + (f" CRASHED@{crashed}" if crashed else ""))
    return rec


RECIPES = [
    dict(name="R1 基线(cos)",       B=512, steps=BUDGET // 512, lr=3e-3, warmup=102, schedule="cosine",  dropout=0.1, label_smoothing=0.0),
    dict(name="R2 lr小10x",         B=512, steps=BUDGET // 512, lr=3e-4, warmup=102, schedule="cosine",  dropout=0.1, label_smoothing=0.0),
    dict(name="R3 lr大10x",         B=512, steps=BUDGET // 512, lr=3e-2, warmup=102, schedule="cosine",  dropout=0.1, label_smoothing=0.0),
    dict(name="R4 批量翻倍",        B=1024, steps=BUDGET // 1024, lr=3e-3, warmup=51, schedule="cosine",  dropout=0.1, label_smoothing=0.0),
    dict(name="R5 批量翻倍+lr√2",   B=1024, steps=BUDGET // 1024, lr=3e-3 * math.sqrt(2), warmup=51, schedule="cosine", dropout=0.1, label_smoothing=0.0),
    dict(name="R6 无dropout",       B=512, steps=BUDGET // 512, lr=3e-3, warmup=102, schedule="cosine",  dropout=0.0, label_smoothing=0.0),
    dict(name="R7 标签平滑0.1",     B=512, steps=BUDGET // 512, lr=3e-3, warmup=102, schedule="cosine",  dropout=0.1, label_smoothing=0.1),
    dict(name="R8 常数lr",          B=512, steps=BUDGET // 512, lr=3e-3, warmup=102, schedule="constant", dropout=0.1, label_smoothing=0.0),
    dict(name="R9 长warmup50%",     B=512, steps=BUDGET // 512, lr=3e-3, warmup=1024, schedule="cosine", dropout=0.1, label_smoothing=0.0),
]


def main():
    torch.manual_seed(0)
    train_text, valid_text, c2i = load_corpus()
    V = len(c2i)
    print(f"corpus: train={len(train_text)} chars, valid={len(valid_text)}, vocab={V}")
    results = []
    for r in RECIPES:
        rec = run_recipe(r["name"], train_text, valid_text, c2i,
                         r["B"], r["steps"], r["lr"], r["warmup"], r["schedule"],
                         r["dropout"], r["label_smoothing"])
        results.append(rec)
        with open(os.path.join(OUT, "recipe_results.json"), "w", encoding="utf-8") as f:
            json.dump({"vocab": V, "budget_tokens": BUDGET, "results": results},
                      f, ensure_ascii=False, indent=1)
    print("saved ->", os.path.join(OUT, "recipe_results.json"))


if __name__ == "__main__":
    main()
