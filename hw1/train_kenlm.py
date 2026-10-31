# -*- coding: utf-8 -*-
"""
用 KenLM lmplz 训练 1..max_order 阶语言模型（Modified Kneser-Ney + discount fallback），
随后 build_binary 转二进制并删除 ARPA 以省磁盘。

本机没有 Windows 版 lmplz（conda-forge 无 win-64 包），因此 lmplz/build_binary 在
WSL Ubuntu 24.04 中从源码编译（kpu/kenlm master），由本脚本通过 wsl 调用；
ARPA 与二进制写到 F:/LLM-course/hw1/models/kenlm（D 盘空间不足已整体迁移），Windows 侧用 pip kenlm 加载查询。

记录每阶 lmplz 墙钟时间与 ARPA 头部各阶 n-gram 计数。
"""
import os
import re
import sys
import time
import json
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_WIN = os.path.join(HERE, "data", "train.txt")
OUT_WIN = r"F:/LLM-course/hw1/models/kenlm"
TMP_WSL = "/root/lmplz_tmp"
LMPLZ_WSL = "/root/kenlm/build/bin/lmplz"
BUILD_BINARY_WSL = "/root/kenlm/build/bin/build_binary"


def to_wsl(win_path):
    p = win_path.replace("\\", "/")
    drive = p[0].lower()
    return f"/mnt/{drive}" + p[2:]


def parse_arpa_counts(path):
    counts = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = re.match(r"ngram (\d+)=(\d+)", line.strip())
            if m:
                counts[int(m.group(1))] = int(m.group(2))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--sort-mem", default="2G")
    ap.add_argument("--no-build-binary", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_WIN, exist_ok=True)
    train_wsl = to_wsl(TRAIN_WIN)
    out_wsl = to_wsl(OUT_WIN)

    results = []
    for order in args.orders:
        bin_win = os.path.join(OUT_WIN, f"order{order}.bin")
        arpa_wsl = f"{TMP_WSL}/order{order}.arpa"
        t0 = time.time()
        cmd = ["wsl", "-d", "Ubuntu", "-e", "sh", "-lc",
               f"mkdir -p {TMP_WSL} && {LMPLZ_WSL} -o {order} --discount_fallback "
               f"-S {args.sort_mem} -T {TMP_WSL} --arpa {arpa_wsl} < {train_wsl} "
               f"> {TMP_WSL}/order{order}.log 2>&1"]
        proc = subprocess.run(cmd, capture_output=True)
        secs = time.time() - t0
        # ARPA 在 ext4 里写，再查大小需要走一次 wsl stat
        stat = subprocess.run(
            ["wsl", "-d", "Ubuntu", "-e", "sh", "-lc",
             f"stat -c %s {arpa_wsl} 2>/dev/null"], capture_output=True)
        arpa_size = 0
        try:
            arpa_size = int(stat.stdout.strip()) / 2**20
        except ValueError:
            pass
        if proc.returncode != 0 or arpa_size == 0:
            print(f"[order {order}] lmplz failed rc={proc.returncode}")
            print(proc.stdout.decode("utf-8", "replace")[-300:])
            results.append({"order": order, "error": True})
            continue
        counts_raw = subprocess.run(
            ["wsl", "-d", "Ubuntu", "-e", "sh", "-lc",
             f"grep '^ngram' {arpa_wsl}"], capture_output=True).stdout.decode("utf-8", "replace")
        counts = {}
        for line in counts_raw.splitlines():
            m = re.match(r"ngram (\d+)=(\d+)", line.strip())
            if m:
                counts[int(m.group(1))] = int(m.group(2))
        log_wsl = f"{TMP_WSL}/order{order}.log"
        if not args.no_build_binary:
            t1 = time.time()
            bproc = subprocess.run(["wsl", "-d", "Ubuntu", "-e", "sh", "-lc",
                                    f"{BUILD_BINARY_WSL} {arpa_wsl} {TMP_WSL}/order{order}.bin "
                                    f"&& cp {TMP_WSL}/order{order}.bin {out_wsl}/ "
                                    f"&& rm -f {arpa_wsl}"],
                                   capture_output=True)
            bsecs = time.time() - t1
            bin_size = os.path.getsize(bin_win) / 2**20 if os.path.exists(bin_win) else 0
            if bin_size == 0:
                print(f"[order {order}] build_binary failed rc={bproc.returncode}:")
                print(bproc.stdout.decode("utf-8", "replace")[-500:])
        else:
            bsecs = 0.0
            bin_size = 0.0
        results.append({
            "order": order,
            "lmplz_seconds": round(secs, 2),
            "build_binary_seconds": round(bsecs, 2),
            "arpa_mib": round(arpa_size, 1),
            "binary_mib": round(bin_size, 1),
            "counts": counts,
            "log": log_wsl,
        })
        print(f"order {order}: lmplz {secs:.1f}s, bin {bsecs:.1f}s, "
              f"arpa {arpa_size:.0f}MiB -> bin {bin_size:.0f}MiB, counts={counts}",
              flush=True)

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "kenlm_train.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved -> results/kenlm_train.json")


if __name__ == "__main__":
    main()
