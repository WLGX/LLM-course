# -*- coding: utf-8 -*-
"""P115 实测：如果一种改进减少了计算量，运行却没有变快，下一步应该测量什么？

实验设计（全部真实运行，WSL Ubuntu + RTX 3050 Laptop 4GB，torch 2.15.0+cu132）：
1. 微型 GPT 块栈（L=6, d=512, h=8），三种注意力实现：
   - MHA_full      多头、完整方阵（显式计算全部 s^2 配对）
   - MHA_causal    多头 + 数学因果掩码（执行时仍算全部 s^2 配对，只是加 -inf 掩码）
   - GQA4_full     8 个 query 头共享 2 个 KV 头（规范 GQA：einsum 分组，不 repeat 物化；
                   按课件口径，s^2 配对乘加不变，只有 K/V 投影按 n_kv/h 缩小）
2. 固定 B*s=4096 的 5 个形状，测墙钟时间并折算实际达到的 TFLOPS（执行口径）。
3. 对照实验（计算量相同、墙钟时间不同）：固定 s=400，batch 1->32 扫描。
4. SDPA 内核对照（正例）：同一"因果"数学量，math 后端（只掩码不跳过）vs
   flash 后端（真正跳过未来配对）vs efficient 后端，fp16，三种形状。
5. 硬件上限：大 GEMM 峰值 TFLOPS（fp32/fp16）、显存带宽（copy 与流式 add 两种探针）。
6. 利用率采样：nvidia-smi 在两种形状下采样 GPU 利用率与功耗。
7. 发射开销：CPU-only profiler 统计算子发射数；CPU 墙钟 vs CUDA 事件时间对比。
8. 异常复查：batch=32 点重测 3 次 + 邻近形状（内存规模假设 vs batch 假设）。
（CUPTI 的 CUDA-activity profiler 在 WSL2 下极慢，改用 CPU-only + CUDA events 的组合。）
输出：results/micro_bench.json（每节完成后增量落盘）；图由 make_figs.py 生成。
"""
import json
import os
import subprocess
import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)
OUT_JSON = os.path.join(OUT, "micro_bench.json")

torch.manual_seed(0)
DEV = torch.device("cuda")
L, D, H = 6, 512, 8
HD = D // H


class Attn(nn.Module):
    """手写注意力：显式控制执行哪些配对，FLOPs 可精确核算。
    n_kv < H 时为规范 GQA（分组 einsum，不 repeat 物化 K/V）。"""

    def __init__(self, d=D, h=H, n_kv=None, causal=False):
        super().__init__()
        self.d, self.h = d, h
        self.n_kv = n_kv or h
        self.hd = d // h
        self.rep = h // self.n_kv
        self.causal = causal
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(d, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(d, d, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q = self.wq(x).view(B, S, self.h, self.hd).transpose(1, 2)          # (B,h,S,hd)
        k = self.wk(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, S, self.n_kv, self.hd).transpose(1, 2)
        if self.n_kv == self.h:
            scores = q @ k.transpose(-2, -1) / (self.hd ** 0.5)             # 完整 s^2 配对
        else:
            # 规范 GQA：query 头分组后与共享 KV 头配对；配对乘加总数与 MHA 相同
            qg = q.view(B, self.n_kv, self.rep, S, self.hd)
            scores = torch.einsum("bgnih,bgjh->bgnij", qg, k) / (self.hd ** 0.5)
            scores = scores.view(B, self.h, S, S)
        if self.causal:                                                     # 数学掩码，不减少执行
            m = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), 1)
            scores = scores.masked_fill(m, float("-inf"))
        att = F.softmax(scores, dim=-1)
        if self.n_kv == self.h:
            y = att @ v
        else:
            attg = att.view(B, self.n_kv, self.rep, S, S)
            y = torch.einsum("bgnij,bgjh->bgnih", attg, v).view(B, self.h, S, self.hd)
        return self.wo(y.transpose(1, 2).reshape(B, S, self.d))


class Block(nn.Module):
    def __init__(self, d=D, h=H, n_kv=None, causal=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attn(d, h, n_kv, causal)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 4 * d)
        self.fc2 = nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


def build(n_kv=None, causal=False):
    return nn.Sequential(*[Block(D, H, n_kv, causal) for _ in range(L)]).to(DEV).eval()


def flops_of(B, S, n_kv, causal, executed=True):
    """按课件口径核算一次前向 FLOPs（MAC 计 2）。
    投影项（q 2d^2 + k 2d^2(n_kv/h) + v 2d^2(n_kv/h) + o 2d^2 + FFN 16d^2）
      = L*B*S*d^2*(20 + 4*n_kv/h)；n_kv=h 时为 24LBSd^2。
    注意力配对项 qk + av = 4*L*B*S^2*d：GQA 不改变它（query 总宽度决定配对乘加，
    课件 GQA 页明文）；causal 的"数学配对"减半，但掩码实现（executed=True）
    实际执行完整方阵（课件：数学掩码 ≠ 执行跳算）。
    """
    proj = L * B * S * D * D * (20 + 4 * n_kv / H)
    qk = 2 * L * B * S * S * D
    av = 2 * L * B * S * S * D
    if causal and not executed:
        qk /= 2.0
        av /= 2.0
    return proj + qk + av


def bench(model, B, S, warmup=10, reps=50):
    x = torch.randn(B, S, D, device=DEV)
    for _ in range(warmup):
        model(x)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        model(x)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) / reps  # ms/iter


def save(out, tag):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[saved:{tag}] -> {OUT_JSON}", flush=True)


def nvidia_smi(query, n=1):
    r = subprocess.run(["nvidia-smi", f"--query-gpu={query}",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=10)
    lines = [l for l in r.stdout.strip().split("\n") if l]
    return lines[:n] if n > 1 else (lines[0] if lines else "?")


def main():
    out = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "config": {"L": L, "d": D, "h": H}}
    ctx = nvidia_smi("driver_version,clocks.sm,clocks.mem,power.limit,power.draw,temperature.gpu")
    out["gpu_context"] = ctx
    print(f"[ctx] driver,sm_clock,mem_clock,power_limit,power_draw,temp = {ctx}", flush=True)
    save(out, "init")

    # ---- 1. 主网格：B*s=4096，三种实现 ----
    grid = [(128, 32), (256, 16), (512, 8), (1024, 4), (2048, 2)]
    variants = [("MHA_full", 8, False), ("MHA_causal", 8, True), ("GQA4_full", 2, False)]
    bench_rows = []
    for S, B in grid:
        for name, n_kv, causal in variants:
            model = build(n_kv, causal)
            t = bench(model, B, S)
            f_math = flops_of(B, S, n_kv, causal, executed=False)
            f_exec = flops_of(B, S, n_kv, causal, executed=True)
            row = {"S": S, "B": B, "variant": name, "ms": round(t, 3),
                   "flops_math": f_math, "flops_exec": f_exec,
                   "tflops_eff": round(f_exec / (t * 1e-3) / 1e12, 2)}
            bench_rows.append(row)
            print(f"[grid] S={S:5d} B={B:2d} {name:12s} {t:7.3f} ms  "
                  f"math={f_math/1e9:6.2f}G exec={f_exec/1e9:6.2f}G "
                  f"eff={row['tflops_eff']:5.2f} TFLOPS", flush=True)
            del model
            torch.cuda.empty_cache()
    out["grid"] = bench_rows
    save(out, "grid")

    # ---- 2. 对照：固定 s=400，batch 1->32（同 FLOPs/窗口，不同墙钟时间） ----
    sweep = []
    model = build(8, True)
    for B in [1, 2, 4, 8, 16, 32]:
        t = bench(model, B, 400)
        per_token = t * 1e-3 / (B * 400)
        row = {"B": B, "ms": round(t, 3), "ms_per_token": round(per_token * 1e6, 2)}
        sweep.append(row)
        print(f"[sweep] s=400 B={B:2d} {t:7.3f} ms  {per_token*1e6:6.2f} us/token", flush=True)
    out["batch_sweep"] = sweep
    save(out, "sweep")
    del model
    torch.cuda.empty_cache()

    # ---- 3. 硬件上限：GEMM 峰值与显存带宽（copy 探针 + 流式探针） ----
    a = torch.randn(4096, 4096, device=DEV)
    b = torch.randn(4096, 4096, device=DEV)
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(20):
        torch.matmul(a, b)
    ev1.record()
    torch.cuda.synchronize()
    gemm_ms = ev0.elapsed_time(ev1) / 20
    gemm_flops = 2 * 4096 ** 3
    peak_fp32 = gemm_flops / (gemm_ms * 1e-3) / 1e12
    print(f"[peak] fp32 GEMM 4096^3: {gemm_ms:.2f} ms -> {peak_fp32:.2f} TFLOPS", flush=True)

    with torch.autocast("cuda", torch.float16):
        for _ in range(5):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        ev0.record()
        for _ in range(20):
            torch.matmul(a, b)
        ev1.record()
        torch.cuda.synchronize()
    gemm16_ms = ev0.elapsed_time(ev1) / 20
    peak_fp16 = gemm_flops / (gemm16_ms * 1e-3) / 1e12
    print(f"[peak] fp16 GEMM 4096^3: {gemm16_ms:.2f} ms -> {peak_fp16:.2f} TFLOPS", flush=True)
    del a, b
    torch.cuda.empty_cache()

    def bw_probe(mb, reps, kind):
        n = mb * 1024 * 1024 // 4
        x = torch.randn(n, device=DEV)
        y = torch.randn(n, device=DEV)
        z = torch.empty_like(x) if kind == "add" else None

        def op():
            if kind == "copy":
                y.copy_(x)
            elif kind == "add":
                torch.add(x, y, out=z)
            else:  # add_inplace
                x.add_(y)

        for _ in range(3):
            op()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record()
        for _ in range(reps):
            op()
        e1.record()
        torch.cuda.synchronize()
        t = e0.elapsed_time(e1) / reps
        traffic_mul = {"copy": 2, "add": 3, "add_inplace": 2}[kind]
        gbs = traffic_mul * mb * 1024 * 1024 / (t * 1e-3) / 1e9
        print(f"[bw] {kind} {mb}MB: {t:7.2f} ms -> {gbs:7.1f} GB/s", flush=True)
        del x, y, z
        return {"kind": kind, "mb": mb, "ms": round(t, 2), "gbs": round(gbs, 1)}

    bw_rows = []
    for mb in (64, 256, 512):
        bw_rows.append(bw_probe(mb, 50 if mb <= 64 else 20, "copy"))
    bw_rows.append(bw_probe(512, 20, "add"))
    bw_rows.append(bw_probe(512, 20, "add_inplace"))
    out["peak"] = {"gemm_fp32_tflops": round(peak_fp32, 2),
                   "gemm_fp16_tflops": round(peak_fp16, 2),
                   "bandwidth_probes": bw_rows}
    save(out, "peak")

    # ---- 4. SDPA 内核对照：同一"因果"数学量，掩码 vs 真正跳过 ----
    from torch.nn.attention import SDPBackend, sdpa_kernel
    sdp_rows = []
    for S, B in [(512, 8), (1024, 4), (2048, 2)]:
        q = torch.randn(B, 8, S, 64, dtype=torch.float16, device=DEV)
        k = torch.randn(B, 8, S, 64, dtype=torch.float16, device=DEV)
        v = torch.randn(B, 8, S, 64, dtype=torch.float16, device=DEV)
        for bname, backend in [("math", SDPBackend.MATH),
                               ("efficient", SDPBackend.EFFICIENT_ATTENTION),
                               ("flash", SDPBackend.FLASH_ATTENTION)]:
            for causal in (False, True):
                try:
                    def run():
                        with sdpa_kernel([backend]):
                            return F.scaled_dot_product_attention(
                                q, k, v, is_causal=causal)
                    for _ in range(20):
                        run()
                    torch.cuda.synchronize()
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record()
                    for _ in range(100):
                        run()
                    e1.record()
                    torch.cuda.synchronize()
                    t = e0.elapsed_time(e1) / 100
                    # 执行口径：math/efficient 掩码不跳过；flash causal 真正跳过未来配对
                    exec_frac = 0.5 if (backend == SDPBackend.FLASH_ATTENTION and causal) else 1.0
                    f_exec = 4 * B * S * S * D * exec_frac
                    sdp_rows.append({"S": S, "B": B, "backend": bname,
                                     "causal": causal, "ms": round(t, 4),
                                     "tflops_eff": round(f_exec / (t * 1e-3) / 1e12, 2)})
                    print(f"[sdp] S={S:5d} {bname:9s} causal={int(causal)} "
                          f"{t:7.4f} ms", flush=True)
                except Exception as e:
                    sdp_rows.append({"S": S, "B": B, "backend": bname,
                                     "causal": causal, "ms": None,
                                     "err": f"{type(e).__name__}"})
                    print(f"[sdp] S={S:5d} {bname:9s} causal={int(causal)} "
                          f"fail: {type(e).__name__}", flush=True)
        del q, k, v
    out["sdpa"] = sdp_rows
    save(out, "sdpa")

    # ---- 5. SDPA 后端可用性（记录环境限制） ----
    from torch.nn.attention import SDPBackend as _SB
    qq = torch.randn(8, 8, 512, HD, device=DEV)
    kk = torch.randn(8, 8, 512, HD, device=DEV)
    vv = torch.randn(8, 8, 512, HD, device=DEV)
    backend_info = {}
    for bname, backend in [("flash", _SB.FLASH_ATTENTION),
                           ("efficient", _SB.EFFICIENT_ATTENTION),
                           ("math", _SB.MATH)]:
        for prec, ctx in [("fp32", None), ("fp16", torch.autocast("cuda", torch.float16))]:
            try:
                def run():
                    with sdpa_kernel([backend]):
                        _ = F.scaled_dot_product_attention(qq, kk, vv, is_causal=True)
                if ctx is not None:
                    with ctx:
                        run()
                else:
                    run()
                torch.cuda.synchronize()
                backend_info[f"{bname}_{prec}"] = "ok"
            except Exception as e:
                backend_info[f"{bname}_{prec}"] = f"fail: {type(e).__name__}"
    print("[backends]", backend_info, flush=True)
    out["sdp_backends"] = backend_info
    save(out, "backends")
    del qq, kk, vv
    torch.cuda.empty_cache()

    # ---- 6. 利用率采样：小形状 vs 大形状 ----
    def sample_util(B, S, seconds=6.0):
        m = build(8, False)
        stop = threading.Event()
        x = torch.randn(B, S, D, device=DEV)

        def worker():
            while not stop.is_set():
                m(x)

        th = threading.Thread(target=worker)
        th.start()
        utils, powers = [], []
        t0 = time.time()
        while time.time() - t0 < seconds:
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                line = r.stdout.strip().split("\n")[0]
                u, p = line.split(",")
                utils.append(float(u))
                powers.append(float(p))
            except Exception:
                pass
            time.sleep(0.5)
        stop.set()
        th.join()
        del m
        torch.cuda.empty_cache()
        return round(sum(utils) / len(utils), 1), round(sum(powers) / len(powers), 1)

    util_rows = []
    for B, S, tag in [(32, 128, "small(128,32)"), (2, 2048, "large(2048,2)")]:
        u, p = sample_util(B, S)
        util_rows.append({"shape": tag, "util_pct": u, "power_w": p})
        print(f"[util] {tag}: GPU util {u}%, power {p}W", flush=True)
    out["util"] = util_rows
    save(out, "util")

    # ---- 7. 发射开销：CPU 墙钟 vs CUDA 事件时间 + CPU-only profiler ----
    from torch.profiler import profile, ProfilerActivity
    m = build(8, True)
    launch_rows = []
    for B, S, tag in [(32, 128, "small(128,32)"), (2, 2048, "large(2048,2)")]:
        x = torch.randn(B, S, D, device=DEV)
        for _ in range(5):
            m(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            m(x)
        cpu_ms = (time.perf_counter() - t0) / 30 * 1e3
        torch.cuda.synchronize()
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        ev0.record()
        for _ in range(30):
            m(x)
        ev1.record()
        torch.cuda.synchronize()
        cuda_ms = ev0.elapsed_time(ev1) / 30
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            for _ in range(10):
                m(x)
        evts = prof.key_averages()
        n_ops = sum(e.count for e in evts)
        top = [{"name": e.key[:60], "count": e.count,
                "cpu_us": round(e.self_cpu_time_total, 1)}
               for e in sorted(evts, key=lambda e: -e.self_cpu_time_total)[:6]]
        row = {"shape": tag, "cpu_ms_per_iter": round(cpu_ms, 3),
               "cuda_ms_per_iter": round(cuda_ms, 3), "n_ops_per_iter": n_ops // 10,
               "top_cpu_ops": top}
        launch_rows.append(row)
        print(f"[launch] {tag}: cpu={cpu_ms:.3f}ms cuda={cuda_ms:.3f}ms "
              f"ops/iter={n_ops // 10}", flush=True)
    out["launch"] = launch_rows
    save(out, "launch")
    del m
    torch.cuda.empty_cache()

    # ---- 8. 异常复查：B=32 点重测 + 邻近形状 ----
    recheck = []
    for trial in range(3):
        mm = build(8, True)
        t = bench(mm, 32, 400)
        recheck.append({"shape": "s=400,B=32", "trial": trial + 1, "ms": round(t, 3)})
        print(f"[recheck] s=400 B=32 trial{trial+1}: {t:7.3f} ms", flush=True)
        del mm
        torch.cuda.empty_cache()
    for B, S in [(16, 400), (24, 400), (32, 300)]:
        mm = build(8, True)
        t = bench(mm, B, S)
        recheck.append({"shape": f"s={S},B={B}", "trial": 1, "ms": round(t, 3)})
        print(f"[recheck] s={S} B={B}: {t:7.3f} ms", flush=True)
        del mm
        torch.cuda.empty_cache()
    out["recheck"] = recheck
    save(out, "done")
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
