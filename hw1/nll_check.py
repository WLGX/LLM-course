# -*- coding: utf-8 -*-
"""P109 思考题验证：实际训练中如何用 PyTorch 实现 NLL 的计算。

逐条验证幻灯片上 Qwen 示例代码的工程细节问题，并给出正确写法、
验证其与 F.cross_entropy 的等价性。运行：python nll_check.py
"""
import torch
import torch.nn.functional as F

torch.manual_seed(42)
V = 50000          # 词表
SEQ = 6            # 序列长度
print("torch", torch.__version__)
print("=" * 60)

# ---------- 1) 逐字复现幻灯片上的 Qwen 代码 ----------
logits = torch.randn(SEQ, V) * 3          # [6, 50000]
targets = torch.tensor([100, 200, 300, 400, 500, 600])

probs = F.softmax(logits, dim=-1)         # Step 1: 算了概率……
log_probs = F.log_softmax(logits, dim=-1)  # Step 2: 实际用的是 log_softmax
predicted_log_probs = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze()
nll = -predicted_log_probs.sum()          # Step 3: 求和取负

loss_default = F.cross_entropy(logits.view(-1, V), targets.view(-1))
loss_sum = F.cross_entropy(logits.view(-1, V), targets.view(-1), reduction="sum")

print("1) 复现幻灯片代码")
print("   probs 变量之后从未被使用（死代码，注释与实现脱节）")
print(f"   nll (sum)            = {nll.item():.4f}")
print(f"   cross_entropy 默认mean = {loss_default.item():.4f}")
print(f"   cross_entropy sum    = {loss_sum.item():.4f}")
print(f"   nll / ce(mean) = {(nll / loss_default).item():.1f}  -> 差 seq_len={SEQ} 倍，"
      "幻灯片说'等价'并不成立（sum 才等价）")
print("   nll == ce(sum):", torch.allclose(nll, loss_sum, atol=1e-4))
print("=" * 60)

# ---------- 2) 坑：padding 位置被当作真实标签 ----------
batch = torch.tensor([[3, 4, 5, 6, 7],
                      [8, 9, 10, 11, 0],
                      [12, 13, 0, 0, 0]])   # 长度 5/4/2，pad_id=0
logits_b = torch.randn(3, 5, V) * 3
loss_no_mask = F.cross_entropy(logits_b.view(-1, V), batch.view(-1))
loss_mask = F.cross_entropy(logits_b.view(-1, V), batch.view(-1), ignore_index=0)
print("2) padding 掩码")
print(f"   不掩码 loss = {loss_no_mask.item():.4f}   (pad 位置的预测错误被计入了 loss)")
print(f"   ignore_index=0 loss = {loss_mask.item():.4f}")
print("=" * 60)

# ---------- 3) 坑：语言模型的 targets 是输入右移一位 ----------
ids = torch.tensor([[101, 2054, 2003, 2501, 102, 0]])   # [BOS] this is nice [EOS] pad
logits_lm = torch.randn(1, 6, V) * 3
print("3) LM 的 shift（teacher forcing 对齐）")
print("   input ids        :", ids.tolist())
print("   预测目标(右移)   :", ids[:, 1:].tolist(), " <- 位置 t 的 logits 预测位置 t+1 的词")
print("=" * 60)

# ---------- 4) 坑：log(softmax) vs log_softmax ----------
extreme = torch.tensor([1000.0, 0.0, -1000.0])
print("4) 数值稳定性（极端 logits）")
print("   log(softmax)  =", torch.log(F.softmax(extreme, dim=-1)).tolist())
print("   log_softmax   =", F.log_softmax(extreme, dim=-1).tolist())
print("=" * 60)

# ---------- 5) 正确写法 + 与 cross_entropy 的等价性验证 ----------
pad = (ids == 0)
lg = F.log_softmax(logits_lm[:, :-1], dim=-1)                    # [1,5,V]
g = lg.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)          # [1,5]
mask = ~pad[:, 1:]                                               # pad 对应位置不参与
nll_manual = -(g * mask).sum() / mask.sum()                      # masked mean NLL
ce = F.cross_entropy(logits_lm[:, :-1].reshape(-1, V),
                     ids[:, 1:].reshape(-1), ignore_index=0)
print("5) 正确写法与等价性")
print(f"   manual masked mean NLL          = {nll_manual.item():.6f}")
print(f"   F.cross_entropy(ignore_index=0) = {ce.item():.6f}")
print("   allclose:", torch.allclose(nll_manual, ce, atol=1e-5))
