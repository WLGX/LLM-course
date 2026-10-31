# -*- coding: utf-8 -*-
"""
Stupid Backoff（Goodman 2001 简化回退）计分器。

分数递归定义（对数域）：
    S(w|h) = log( c(h,w)/c(h) )                若 c(h,w)>0
           = log(0.4) + S(w|h[1:])             否则（h 非空）
           = log( c(w)/N )                     空历史（一阶）

计数存放在 SQLite（D 盘），查询带一个简单 LRU 缓存。
注意：SB 分数不是归一化概率，只用于候选排序与续写，不计算 perplexity。
"""
import math
import sqlite3
import os

ALPHA = 0.4
CACHE_CAP = 4_000_000


class StupidBackoff:
    def __init__(self, db_path, max_order=5):
        self.conn = sqlite3.connect(db_path)
        self.max_order = max_order
        self.cache = {}
        self.n_hits = 0
        self.n_misses = 0
        self._load_unigram()

    def _load_unigram(self):
        rows = self.conn.execute(
            "SELECT w1, c FROM n1 WHERE w1 != '<s>'"
        ).fetchall()
        self.total = sum(c for _, c in rows)
        self.unigram = dict(rows)
        self.top_words = [w for w, _ in
                          sorted(rows, key=lambda x: -x[1])[:200]]

    def count(self, order, ngram):
        """查询 ngram 的计数（含缓存）。ngram 为元组。"""
        key = (order,) + ngram
        v = self.cache.get(key)
        if v is not None:
            self.n_hits += 1
            return v
        cols = [f"w{i}" for i in range(1, order + 1)]
        sql = f"SELECT c FROM n{order} WHERE " + \
            " AND ".join(f"{c} = ?" for c in cols)
        row = self.conn.execute(sql, ngram).fetchone()
        v = row[0] if row else 0
        if len(self.cache) >= CACHE_CAP:
            self.cache.clear()   # 简单容量控制：满则整体清空
        self.cache[key] = v
        self.n_misses += 1
        return v

    def logscore(self, word, context):
        """log S(word | context)。context 为词列表（不含当前词）。"""
        s = self._rec(word, tuple(context), min(self.max_order, len(context) + 1))
        return math.log(s) if s > 0 else float("-inf")

    def _rec(self, word, context, order):
        if order <= 1 or not context:
            c = self.unigram.get(word, 0)
            return c / self.total if c else 0.0
        context = context[-(order - 1):]           # 只保留最近 order-1 个历史词
        joint = self.count(order, context + (word,))
        if joint:
            denom = self.count(order - 1, context)
            return joint / denom
        return ALPHA * self._rec(word, context[1:], order - 1)

    def sample_next(self, context, temperature=1.0, top_k=20, rng=None):
        """在 top-200 词 + </s> 候选上按 logscore 做温度缩放 + top-k 采样。"""
        cands = list(self.top_words) + ["</s>"]
        scores = []
        for w in cands:
            ls = self.logscore(w, context)
            scores.append(math.exp(ls / temperature) if ls != float("-inf") else 0.0)
        order_idx = sorted(range(len(cands)), key=lambda i: -scores[i])[:top_k]
        pool = [(cands[i], scores[i]) for i in order_idx]
        total = sum(s for _, s in pool)
        r = (rng.random() if rng else 0.5) * total
        acc = 0.0
        for w, s in pool:
            acc += s
            if r <= acc:
                return w
        return pool[-1][0]


if __name__ == "__main__":
    # 自检：小库上验证回退逻辑
    import tempfile
    db = tempfile.mktemp(suffix=".db")
    c = sqlite3.connect(db)
    for n, ddl in {
        1: "CREATE TABLE n1(w1 TEXT, c INTEGER, UNIQUE(w1))",
        2: "CREATE TABLE n2(w1 TEXT, w2 TEXT, c INTEGER, UNIQUE(w1,w2))",
        3: "CREATE TABLE n3(w1 TEXT, w2 TEXT, w3 TEXT, c INTEGER, UNIQUE(w1,w2,w3))",
    }.items():
        c.execute(ddl)
    # 语料：a b a b c d
    c.execute("INSERT INTO n1(w1,c) VALUES('a',2),('b',2),('c',1),('d',1)")
    c.execute("INSERT INTO n2(w1,w2,c) VALUES('a','b',2),('b','a',1),('b','c',1),('c','d',1),('a','a',1)")
    c.execute("INSERT INTO n3(w1,w2,w3,c) VALUES('a','b','a',1),('a','b','c',1)")
    c.commit(); c.close()
    sb = StupidBackoff(db)
    # S(b|a) = 2/2 = 1
    assert abs(math.exp(sb.logscore('b', ['a'])) - 1.0) < 1e-9
    # trigram (a,b,c) 存在：S(c|a b) = 1/2 = 0.5
    assert abs(math.exp(sb.logscore('c', ['a', 'b'])) - 0.5) < 1e-9
    # S(d|a b): 三元未观测 -> 0.4*S(d|b): 二元未观测 -> 0.4*S(d) = 0.4*0.4*(1/6)
    assert abs(math.exp(sb.logscore('d', ['a', 'b'])) - 0.4 * 0.4 / 6) < 1e-9
    # 完全未见词 -> unigram 概率 0 -> -inf
    assert sb.logscore('x', ['a', 'b']) == float('-inf')
    print("self-check passed")
    sb.conn.close()
    os.remove(db)
