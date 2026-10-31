# -*- coding: utf-8 -*-
"""
n-gram 计数：把清洗后的训练语料按阶数统计进 SQLite（磁盘存储，控制内存）。

用法：
  python count_ngrams.py --orders 1 2 3 4 5 [--min-count 1] [--db data/ngrams.db]

每个句子用 <s> 左补齐 n-1 个、尾部补一个 </s>。SQLite 每阶一张表 n{n}(w1..wn, c)，
UNIQUE(w1..wn)。逐阶独立跑一遍语料，记录每阶的墙钟时间（这就是"对比训练时间"的数据）。
"""
import os
import re
import sys
import time
import json
import sqlite3
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def count_order(db_path, order, train_path, min_count, batch=200_000):
    """对训练语料统计 order 阶 n-gram，返回该阶统计信息。"""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    cols = ", ".join(f"w{i}" for i in range(1, order + 1))
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS n{order} ({cols} TEXT, c INTEGER, "
        f"UNIQUE({cols}))"
    )
    sql = (
        f"INSERT INTO n{order}({cols}, c) VALUES ({', '.join('?' * order)}, 1) "
        f"ON CONFLICT({cols}) DO UPDATE SET c = c + 1"
    )

    t0 = time.time()
    n_tokens = 0
    n_grams = 0
    batch_rows = []
    pad = ["<s>"] * (order - 1)
    with open(train_path, encoding="utf-8") as f:
        for ln in f:
            toks = ln.split()
            if not toks:
                continue
            n_tokens += len(toks)
            seq = pad + toks + ["</s>"]
            for j in range(len(pad), len(seq)):
                batch_rows.append(tuple(seq[j - order + 1:j + 1]))
                n_grams += 1
                if len(batch_rows) >= batch:
                    conn.executemany(sql, batch_rows)
                    batch_rows = []
    if batch_rows:
        conn.executemany(sql, batch_rows)

    if min_count > 1:
        conn.execute(f"DELETE FROM n{order} WHERE c < ?", (min_count,))
    conn.commit()

    n_distinct = conn.execute(f"SELECT COUNT(*) FROM n{order}").fetchone()[0]
    n_pruned_rows = conn.execute(
        f"SELECT COUNT(*) FROM n{order} WHERE c < ?", (min_count,)
    ).fetchone()[0]
    total_rows = conn.execute(f"SELECT COUNT(*) FROM n{order}").fetchone()[0]
    conn.close()

    return {
        "order": order,
        "seconds": round(time.time() - t0, 2),
        "tokens": n_tokens,
        "n_grams": n_grams,
        "distinct": n_distinct,
        "rows_kept": total_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orders", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--db", default=os.path.join(DATA, "ngrams.db"))
    ap.add_argument("--train", default=os.path.join(DATA, "train.txt"))
    args = ap.parse_args()

    d = os.path.dirname(args.db)
    if d:
        os.makedirs(d, exist_ok=True)
    results = []
    for order in args.orders:
        print(f"[count] order {order} ...", flush=True)
        r = count_order(args.db, order, args.train, args.min_count)
        results.append(r)
        print(f"  {r['seconds']}s, tokens={r['tokens']:,}, distinct={r['distinct']:,}")

    out = {"min_count": args.min_count, "orders": results}
    with open(os.path.join(DATA, "count_times.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("count_times.json written")


if __name__ == "__main__":
    main()
