# -*- coding: utf-8 -*-
"""冒烟测试：分块 head 损失在 B=1024 与 label_smoothing 路径下可运行、不 OOM。"""
import sys
import os
import math

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recipe_exp import MiniGPT, head_loss, load_corpus, make_batches, DEV  # noqa: E402

train_text, valid_text, c2i = load_corpus()
V = len(c2i)
model = MiniGPT(V).to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.1)
data = make_batches(train_text, c2i, 1024)
vdata = make_batches(valid_text, c2i, 128)

torch.manual_seed(0)
for step in range(3):
    idx = torch.randint(0, len(data), (1024,))
    xb = torch.stack([data[i][0] for i in idx]).to(DEV)
    yb = torch.stack([data[i][1] for i in idx]).to(DEV)
    h = model(xb, dropout=0.1)
    loss = head_loss(model.head, h, yb, V, 0.0)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(f"step {step}: B=1024 loss={loss.item():.4f} finite={math.isfinite(loss.item())}")

# 标签平滑路径（B=512）
idx = torch.randint(0, len(data), (512,))
xb = torch.stack([data[i][0] for i in idx]).to(DEV)
yb = torch.stack([data[i][1] for i in idx]).to(DEV)
h = model(xb, dropout=0.1)
loss = head_loss(model.head, h, yb, V, 0.1)
print(f"label_smoothing loss={loss.item():.4f} finite={math.isfinite(loss.item())}")

# 验证路径
with torch.no_grad():
    model.eval()
    tot, cnt = 0.0, 0
    xv, yv = torch.stack([c[0] for c in vdata[:128]]).to(DEV), torch.stack([c[1] for c in vdata[:128]]).to(DEV)
    hv = model(xv, dropout=0.0)
    tot += head_loss(model.head, hv, yv, V, 0.0).item() * yv.numel()
    cnt += yv.numel()
    print(f"valid ppl={math.exp(tot / cnt):.2f}")
print("SMOKE OK")
