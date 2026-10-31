# -*- coding: utf-8 -*-
"""P226 小规模验证实验：SFT 起点上比较 离线 DPO / 迭代(在线) DPO / 在线 GRPO。

对应课件 L3 part2 第 124 张思考题与第 226 张大作业的"小规模验证"要求：
"是否由模型自己生成新回答，是离线 DPO 与在线 DPO 的分界；
数学、代码等可验证任务可用验证结果获得新信号。"

任务：三位数加法（a+b，a,b∈[100,999]），字符级自回归生成答案（3–4 位）。
设计说明（v1→v2→v3 迭代）：两位数加法（3240 对）可被完全背表（v1 实验 SFT 即达
acc=1.000，DPO/GRPO 无提升空间，弃用）；四位数加法对 CPU 上的微型模型过难
（v2 探针 SFT 500 步 acc≈0.005，无法为对比提供起点，弃用）；三位数加法共 81 万对
组合、train/eval 划分后 held-out 必须真学进位算法，难度适中。
SFT 步数控制在留出提升空间的水平（用 SFT_ONLY 探针沿训练曲线选点）。
可验证奖励：answer 奖励（答案是否等于 a+b）+ format 奖励（是否纯数字）。

四条路线（同一 SFT 起点 π_sft，同一模型结构、同一参考策略 π_ref=π_sft）：
  sft            —— 随机初始化 → 仅 SFT（基线）
  offline_dpo    —— 偏好对一次性从 π_sft 采样（训练期间分布不再刷新）→ DPO 若干步
  iterative_dpo  —— "在线 DPO"：每轮从当前策略重新采样偏好对（多轮）
  grpo           —— 在线 GRPO：每组 4 样本、组内相对优势、KL 到 π_sft

记录：训练损失曲线、测试准确率、平均答案长度（字符数）、训练耗时。
全部自写实现（无 TRL/verl），CPU 可运行。
"""
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
random.seed(0)

DEVICE = "cpu"

# ---------------- 任务与词表 ----------------
CHARS = list("0123456789") + ["+", "="]
PAD_ID, EOS_ID = 0, len(CHARS) + 1  # pad=0；eos 单独一个 id
VOCAB = len(CHARS) + 2
stoi = {c: i + 1 for i, c in enumerate(CHARS)}  # 字符从 1 开始，0 留给 pad
stoi["<eos>"] = EOS_ID

MAX_LEN = 13  # prompt 8 位（"123+456="）+ 答案最多 4 位 + EOS = 13


def encode(s, add_eos=False):
    ids = [stoi[c] for c in s]
    if add_eos:
        ids.append(EOS_ID)
    return ids


def make_pairs(n_train=5000, n_eval=1000):
    """随机采样 3 位数加法对（81 万对组合不可能背表），train/eval 不重叠。"""
    train_set = set()
    while len(train_set) < n_train:
        train_set.add((random.randint(100, 999), random.randint(100, 999)))
    ev_set = set()
    while len(ev_set) < n_eval:
        p = (random.randint(100, 999), random.randint(100, 999))
        if p not in train_set:
            ev_set.add(p)
    return list(train_set), list(ev_set)


def prompt_str(a, b):
    return f"{a}+{b}="


def answer_str(a, b):
    return str(a + b)


# ---------------- 微型 GPT ----------------
class TinyGPT(nn.Module):
    def __init__(self, d=128, n_layer=3, n_head=4, block_size=MAX_LEN):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(block_size, d)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d, nhead=n_head, dim_feedforward=4 * d,
                dropout=0.0, batch_first=True, norm_first=True,
            ) for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, VOCAB, bias=False)

    def forward(self, x):
        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device)).unsqueeze(0)
        for blk in self.blocks:
            h = blk(h, src_mask=causal)
        return self.head(self.ln(h))

    @torch.no_grad()
    def generate(self, prompts, max_new=5, temperature=1.0):
        """prompts: list[int...] 或 LongTensor。返回 (ids, logp_sum) 两列表。
        temperature<=0 时为贪心解码。"""
        self.eval()
        out_ids, logps = [], []
        for p in prompts:
            x = torch.tensor([p], dtype=torch.long, device=DEVICE)
            cur_logp = 0.0
            for _ in range(max_new):
                logits = self(x)[0, -1]
                if temperature <= 0:
                    nxt = torch.argmax(logits, dim=-1).item()
                else:
                    probs = F.softmax(logits / temperature, dim=-1)
                    nxt = torch.multinomial(probs, 1).item()
                    cur_logp += math.log(max(probs[nxt].item(), 1e-12))
                x = torch.cat([x, torch.tensor([[nxt]], device=DEVICE)], dim=1)
                if nxt == EOS_ID:
                    break
            out_ids.append(x[0].tolist())
            logps.append(cur_logp)
        self.train()
        return out_ids, logps


def decode_ids(ids, prompt_len):
    toks = ids[prompt_len:]
    if EOS_ID in toks:
        toks = toks[:toks.index(EOS_ID)]
    return "".join(CHARS[t - 1] for t in toks if 1 <= t <= len(CHARS))


def batchify(lst, bs):
    for i in range(0, len(lst), bs):
        yield lst[i:i + bs]


def pad_to(seq, length):
    return seq + [PAD_ID] * (length - len(seq))


def make_sft_batch(pairs):
    xs, ys = [], []
    for a, b in pairs:
        src = encode(prompt_str(a, b))
        tgt = encode(answer_str(a, b), add_eos=True)
        x = src + tgt[:-1]
        y = src[1:] + tgt
        xs.append(pad_to(x, MAX_LEN))
        ys.append(pad_to(y, MAX_LEN))
    return xs, ys


@torch.no_grad()
def evaluate(model, pairs, n_eval=None):
    """贪心解码测试准确率与平均答案长度。返回 (acc, avg_len)。"""
    pairs = pairs if n_eval is None else pairs[:n_eval]
    model.eval()
    correct, lens = 0, 0
    prompts = []
    for a, b in pairs:
        prompts.append((a, b, encode(prompt_str(a, b))))
    for chunk in batchify(prompts, 128):
        for a, b, p in chunk:
            ids = model.generate([p], max_new=5, temperature=0)[0][0]
            ans = decode_ids(ids, len(p))
            lens += len(ans)
            if ans == answer_str(a, b):
                correct += 1
    model.train()
    return correct / len(pairs), lens / len(pairs)


# ---------------- 训练循环 ----------------
def train_sft(model, pairs, steps=500, bs=128, lr=1e-3, eval_every=0, eval_fn=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    xs, ys = make_sft_batch(pairs)
    log, t0 = [], time.time()
    for step in range(steps):
        i = (step * bs) % len(xs)
        bx = torch.tensor(xs[i:i + bs], device=DEVICE)
        by = torch.tensor(ys[i:i + bs], device=DEVICE)
        logits = model(bx)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1), ignore_index=PAD_ID)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 40 == 0 or step == steps - 1:
            log.append((step, loss.item()))
            print(f"  [sft] step {step} loss {loss.item():.4f}", flush=True)
        if eval_every and step > 0 and step % eval_every == 0 and eval_fn is not None:
            acc, alen = eval_fn()
            print(f"  [sft] @{step} acc {acc:.3f} len {alen:.2f} t {time.time() - t0:.0f}s",
                  flush=True)
    return log, time.time() - t0


def dpo_loss(model, ref_model, xw, yw, xl, yl, beta=0.1):
    """x/y 为 [bs, MAX_LEN]；只在非 pad 位置计算 logπ（pad 掩掉）。
    prompt 部分在 w/l 间完全相同、相互抵消，实际比较的是答案段。"""
    def logp_of(m, x, y):
        logits = m(x).log_softmax(-1)  # [bs,T,V]
        lp = logits.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)  # [bs,T]
        mask = (y >= 1).float()  # pad=0 掩掉；字符/EOS 参与
        return (lp * mask).sum(-1)  # [bs]
    pi_w, pi_l = logp_of(model, xw, yw), logp_of(model, xl, yl)
    r_w, r_l = logp_of(ref_model, xw, yw), logp_of(ref_model, xl, yl)
    logits_pair = beta * (pi_w - pi_l) - beta * (r_w - r_l)
    return -F.logsigmoid(logits_pair).mean()


def dpo_train(model, ref, pairs_data, steps, bs=64, lr=3e-4, beta=0.1, tag="dpo"):
    """pairs_data: list of (xw,yw,xl,yl) 固定偏好对（离线）或采样函数（在线）。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log, t0 = [], time.time()
    for step in range(steps):
        data = pairs_data() if callable(pairs_data) else pairs_data
        i = (step * bs) % len(data)
        batch = data[i:i + bs]
        xw = torch.tensor([d[0] for d in batch], device=DEVICE)
        yw = torch.tensor([d[1] for d in batch], device=DEVICE)
        xl = torch.tensor([d[2] for d in batch], device=DEVICE)
        yl = torch.tensor([d[3] for d in batch], device=DEVICE)
        loss = dpo_loss(model, ref, xw, yw, xl, yl, beta)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 40 == 0 or step == steps - 1:
            log.append((step, float(loss)))
            print(f"  [{tag}] step {step} loss {loss.item():.4f}", flush=True)
    return log, time.time() - t0


def sample_pairs(model, ref, n_pairs, seed_pool):
    """从当前策略采样偏好对：每组抽 2 个答案，正确者为 chosen、错误者为 rejected。"""
    data = []
    attempts = 0
    while len(data) < n_pairs and attempts < n_pairs * 6:
        attempts += 1
        a, b = random.choice(seed_pool)
        p = encode(prompt_str(a, b))
        ids, _ = model.generate([p, p], max_new=5, temperature=1.0)
        cands = [decode_ids(x, len(p)) for x in ids]
        good = [c for c in cands if c == answer_str(a, b)]
        bad = [c for c in cands if c != answer_str(a, b)]
        if good and bad:
            src = encode(prompt_str(a, b))
            y_w = encode(good[0], add_eos=True)
            y_l = encode(bad[0], add_eos=True)
            xw = src + y_w[:-1]; yw_ = src[1:] + y_w
            xl = src + y_l[:-1]; yl_ = src[1:] + y_l
            data.append((pad_to(xw, MAX_LEN), pad_to(yw_, MAX_LEN),
                         pad_to(xl, MAX_LEN), pad_to(yl_, MAX_LEN)))
    print(f"  [sample_pairs] got {len(data)}/{n_pairs}", flush=True)
    return data


def answer_logp_vec(model, p_list, ans_list):
    """只对答案 token 计算 logπ（prompt 部分不参与，保证与采样时 old_logp 口径一致）。"""
    L = max(len(p) + len(a) for p, a in zip(p_list, ans_list))
    x = torch.zeros(len(p_list), L, dtype=torch.long, device=DEVICE)
    y = torch.full((len(p_list), L), -1, dtype=torch.long, device=DEVICE)
    for i, (p, a) in enumerate(zip(p_list, ans_list)):
        x[i, :len(p) + len(a)] = torch.tensor(p + a, device=DEVICE)
        y[i, len(p) - 1:len(p) - 1 + len(a)] = torch.tensor(a, device=DEVICE)
    mask = (y >= 0)
    logsoft = model(x).log_softmax(-1)
    lp = logsoft.gather(-1, y.clamp(min=0).unsqueeze(-1)).squeeze(-1) * mask
    return lp.sum(-1)


def grpo_train(model, ref, seed_pool, steps=200, bs=64, group=4, lr=3e-4, eps=0.2, kl_coef=0.02):
    """组相对优势的 GRPO：r_group 标准化，clip 代理目标 + KL 到 ref。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    log, t0 = [], time.time()
    for step in range(steps):
        # 采样：bs 个 prompt × group 个答案
        prompts, answers, old_logps = [], [], []
        for _ in range(bs):
            a, b = random.choice(seed_pool)
            p = encode(prompt_str(a, b))
            ids, lps = model.generate([p] * group, max_new=5, temperature=1.0)
            for x, lp in zip(ids, lps):
                prompts.append(p)
                answers.append(x[len(p):])
                old_logps.append(lp)
        rewards = []
        for p, ans in zip(prompts, answers):
            s = decode_ids(p + ans, len(p))
            a_val = int("".join(CHARS[t - 1] for t in p[:3]))
            b_val = int("".join(CHARS[t - 1] for t in p[4:7]))
            acc = 1.0 if s == answer_str(a_val, b_val) else 0.0
            fmt = 1.0 if s.isdigit() else 0.0
            rewards.append(0.9 * acc + 0.1 * fmt)
        rewards = torch.tensor(rewards, device=DEVICE)
        # 组内标准化优势
        adv = []
        for g in range(bs):
            r = rewards[g * group:(g + 1) * group]
            adv.append((r - r.mean()) / (r.std() + 1e-8))
        adv = torch.cat(adv)
        # 当前策略与参考策略在答案段上的 logp（prompt 部分不参与 ratio）
        cur_logp = answer_logp_vec(model, prompts, answers)
        with torch.no_grad():
            ref_logp = answer_logp_vec(ref, prompts, answers)
        old_logp = torch.tensor(old_logps, device=DEVICE)
        ratio = torch.exp(cur_logp - old_logp)
        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - eps, 1 + eps) * adv).mean()
        kl = (cur_logp - ref_logp).mean()
        loss = pg + kl_coef * kl
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 30 == 0 or step == steps - 1:
            log.append((step, float(loss), float(pg), float(kl),
                        float((rewards > 0.9).sum()) / len(rewards)))
            print(f"  [grpo] step {step} loss {loss.item():.4f} pg {pg.item():.4f} "
                  f"kl {kl.item():.4f} acc@batch {log[-1][-1]:.3f}", flush=True)
    return log, time.time() - t0


# ---------------- 主流程 ----------------
def main():
    smoke = os.environ.get("SMOKE") == "1"
    sft_only = os.environ.get("SFT_ONLY") == "1"
    sft_steps_probe = int(os.environ.get("SFT_STEPS", "2400"))
    if smoke:
        n_train, n_eval = 300, 150
        sft_steps, dpo_steps, it_rounds, it_steps = 200, 60, 2, 40
        grpo_steps, n_off, n_it = 60, 150, 100
    else:
        n_train, n_eval = 5000, 1000
        sft_steps, dpo_steps, it_rounds, it_steps = int(os.environ.get("SFT_STEPS", "1600")), 400, 3, 120
        grpo_steps, n_off, n_it = 200, 600, 400

    train_pairs, eval_pairs = make_pairs(n_train, n_eval)
    print(f"train {len(train_pairs)} eval {len(eval_pairs)}", flush=True)

    def new_model():
        m = TinyGPT().to(DEVICE)
        m.train()
        return m

    if sft_only:
        # 探针：单次 train_sft 调用内每 50 步评估一次（与正式运行完全同轨迹），
        # 找出 acc 落入 30%–80% 留白区的步数
        sft = new_model()
        train_sft(sft, train_pairs, steps=sft_steps_probe, eval_every=50,
                  eval_fn=lambda: evaluate(sft, eval_pairs))
        return

    report = {"task": "three-digit addition a+b (a,b in [100,999], 810k pairs, no memorization)",
              "config": {"d": 128, "n_layer": 3, "n_head": 4, "block_size": MAX_LEN,
                         "n_train": n_train, "n_eval": n_eval},
              "conditions": {}}

    # ---- SFT 基线 ----
    print("== SFT ==", flush=True)
    sft = new_model()
    sft_log, sft_time = train_sft(sft, train_pairs, steps=sft_steps)
    acc, alen = evaluate(sft, eval_pairs)
    report["conditions"]["sft"] = {"acc": acc, "avg_len": alen, "time_s": sft_time,
                                   "loss": sft_log}
    torch.save(sft.state_dict(), os.path.join(OUT, "sft.pt"))
    print(f"== SFT done: acc {acc:.3f} len {alen:.2f} time {sft_time:.1f}s", flush=True)

    # ---- 离线 DPO：偏好对一次性从 π_sft 采样 ----
    print("== offline DPO ==", flush=True)
    dpo_off = new_model(); dpo_off.load_state_dict(sft.state_dict())
    ref_off = new_model(); ref_off.load_state_dict(sft.state_dict()); ref_off.eval()
    fixed_pairs = sample_pairs(sft, ref_off, n_off, train_pairs)
    dpo_off_log, dpo_off_time = dpo_train(dpo_off, ref_off, fixed_pairs, dpo_steps,
                                          tag="dpo_off")
    acc, alen = evaluate(dpo_off, eval_pairs)
    report["conditions"]["offline_dpo"] = {"acc": acc, "avg_len": alen,
                                           "time_s": dpo_off_time, "loss": dpo_off_log,
                                           "n_pairs": len(fixed_pairs)}
    torch.save(dpo_off.state_dict(), os.path.join(OUT, "dpo_offline.pt"))
    print(f"== offline DPO done: acc {acc:.3f} len {alen:.2f}", flush=True)

    # ---- 迭代(在线) DPO：每轮从当前策略重新采样 ----
    print("== iterative DPO ==", flush=True)
    dpo_it = new_model(); dpo_it.load_state_dict(sft.state_dict())
    ref_it = new_model(); ref_it.load_state_dict(sft.state_dict()); ref_it.eval()
    it_log, t0 = [], time.time()
    for rnd in range(it_rounds):
        fresh = sample_pairs(dpo_it, ref_it, n_it, train_pairs)
        lg, _ = dpo_train(dpo_it, ref_it, fresh, it_steps, tag=f"dpo_it_r{rnd}")
        it_log.append({"round": rnd, "loss": lg, "n_pairs": len(fresh)})
        acc, alen = evaluate(dpo_it, eval_pairs)
        print(f"  [dpo_it] round {rnd} eval acc {acc:.3f} len {alen:.2f}", flush=True)
    dpo_it_time = time.time() - t0
    acc, alen = evaluate(dpo_it, eval_pairs)
    report["conditions"]["iterative_dpo"] = {"acc": acc, "avg_len": alen,
                                             "time_s": dpo_it_time, "rounds": it_log}
    torch.save(dpo_it.state_dict(), os.path.join(OUT, "dpo_iterative.pt"))
    print(f"== iterative DPO done: acc {acc:.3f} len {alen:.2f}", flush=True)

    # ---- 在线 GRPO ----
    print("== GRPO ==", flush=True)
    grpo = new_model(); grpo.load_state_dict(sft.state_dict())
    ref_g = new_model(); ref_g.load_state_dict(sft.state_dict()); ref_g.eval()
    grpo_log, grpo_time = grpo_train(grpo, ref_g, train_pairs, steps=grpo_steps)
    acc, alen = evaluate(grpo, eval_pairs)
    report["conditions"]["grpo"] = {"acc": acc, "avg_len": alen, "time_s": grpo_time,
                                    "loss": grpo_log}
    torch.save(grpo.state_dict(), os.path.join(OUT, "grpo.pt"))
    print(f"== GRPO done: acc {acc:.3f} len {alen:.2f}", flush=True)

    with open(os.path.join(OUT, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    plot(report)
    print("wrote", os.path.join(OUT, "report.json"), flush=True)


def plot(report):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    c = report["conditions"]
    for name, color in [("sft", "gray"), ("offline_dpo", "tab:blue"),
                        ("iterative_dpo", "tab:orange"), ("grpo", "tab:red")]:
        if name == "iterative_dpo":
            rounds = c[name].get("rounds") or []
            if not rounds:
                continue
            xs, ys = [], []
            for rnd in rounds:
                for step, v in rnd["loss"]:
                    xs.append(rnd["round"] * 120 + step)
                    ys.append(v)
        else:
            logs = c[name].get("loss")
            if not logs:
                continue
            xs = [x[0] for x in logs]
            ys = [x[1] for x in logs]
        ax.plot(xs, ys, label=name, color=color)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.set_title("training loss (per method)")
    # 右：最终准确率条形
    ax2 = axes[1]
    names = ["sft", "offline_dpo", "iterative_dpo", "grpo"]
    accs = [c[n]["acc"] for n in names]
    bars = ax2.bar(names, accs, color=["gray", "tab:blue", "tab:orange", "tab:red"])
    for b, v in zip(bars, accs):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    ax2.set_ylim(0, 1.15)
    ax2.set_ylabel("test accuracy")
    ax2.set_title("final test accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig.png"), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
