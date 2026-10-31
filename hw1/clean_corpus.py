# -*- coding: utf-8 -*-
"""
清洗《人民日报 1998 上半年·带音（2003 版）》语料。

原始格式（GBK 编码）：
    19980101-01-001-001 迈向 充满 希望 的 新 世纪 —— 一九九八年 新年 讲话 （ 附 图片 １ 张 ）
    19980101-01-001-007 １９９７年 ， 是 中国 发展 历史 上{shang5} 非常 重要 的 ...

每行行首为文章句编号（日期-版次-文章号-句号），词后可能带 {拼音} 注音标签。
本脚本完成：
  1) gb18030 解码，记录并丢弃无法解码的行；
  2) 去掉行首编号；
  3) 去掉 {拼音} 标签（这是作业要求的"清理多余 tag"）；
  4) 按文章（编号前 15 位）做内容级去重；
  5) 在 。！？； 处切句；
  6) 丢弃不含任何汉字/数字内容的纯标点"句子"；
  7) 按文章编号哈希做 80/10/10 的 train/valid/test 划分（文章级隔离）。

输出：
  data/train.txt data/valid.txt data/test.txt   每行一句，空格分词
  data/clean_stats.json                         清洗统计
"""
import re
import json
import glob
import hashlib
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS_DIR = os.path.join(ROOT, "1998-上半年-带音(1)", "1998-上半年-带音")
DATA_DIR = os.path.join(HERE, "data")

ID_RE = re.compile(r"^\s*(\d{8}-\d{2}-\d{3}-\d{3})\s+")
ARTICLE_KEY_RE = re.compile(r"^(\d{8}-\d{2}-\d{3})")
PINYIN_RE = re.compile(r"\{[^{}]*\}")
SENT_END = set("。！？；")
HANZI_OR_DIGIT = re.compile(r"[\u4e00-\u9fff0-9A-Za-z]")

os.makedirs(DATA_DIR, exist_ok=True)


def main():
    files = sorted(glob.glob(os.path.join(CORPUS_DIR, "*.txt")))
    stats = {
        "input_files": [os.path.basename(f) for f in files],
        "raw_lines": 0,
        "decode_error_lines": [],
        "articles_seen": 0,
        "duplicate_articles_removed": 0,
        "tokens_before_clean": 0,
        "tokens_after_clean": 0,
        "sentences_kept": 0,
        "sentences_pure_punct_dropped": 0,
        "split": {},
    }

    # 第一遍：逐文件清洗，按文章收集句子，并做去重
    article_sentences = {}   # article_key -> list of sentences
    article_hash_seen = {}   # content_hash -> article_key
    dup_ids = []
    for f in files:
        raw = open(f, "rb").read()
        text = raw.decode("gb18030", errors="replace")
        for i, ln in enumerate(text.splitlines()):
            if not ln.strip():
                continue
            stats["raw_lines"] += 1
            if "\ufffd" in ln:
                stats["decode_error_lines"].append(
                    {"file": os.path.basename(f), "line_no": i, "text": ln[:50]}
                )
                continue
            m = ID_RE.match(ln)
            if not m:
                # 无编号行：直接跳过并记录（正常情况下不应出现）
                stats.setdefault("lines_without_id", 0)
                stats["lines_without_id"] += 1
                continue
            art = ARTICLE_KEY_RE.match(m.group(1)).group(1)
            toks = ln[m.end():].split()
            stats["tokens_before_clean"] += len(toks)
            # 去拼音标签
            clean = []
            for tk in toks:
                tk2 = PINYIN_RE.sub("", tk)
                if tk2:
                    clean.append(tk2)
                    stats["tokens_after_clean"] += 1
            if not clean:
                continue
            # 切句
            buf = []
            for tk in clean:
                buf.append(tk)
                if tk in SENT_END:
                    _add_sentence(article_sentences, art, buf, stats)
                    buf = []
            if buf:
                _add_sentence(article_sentences, art, buf, stats)

    stats["articles_seen"] = len(article_sentences)

    # 文章级去重：以整篇文章内容哈希为准
    deduped = {}
    for art, sents in article_sentences.items():
        h = hashlib.md5("\n".join(" ".join(s) for s in sents).encode("utf-8")).hexdigest()
        if h in article_hash_seen:
            dup_ids.append(art)
            stats["duplicate_articles_removed"] += 1
            continue
        article_hash_seen[h] = art
        deduped[art] = sents

    # 划分：md5(article_key) 首字节 % 10：0 -> test, 1 -> valid, 其余 -> train
    buckets = {"train": [], "valid": [], "test": []}
    for art, sents in sorted(deduped.items()):
        b = int(hashlib.md5(art.encode("utf-8")).hexdigest(), 16) % 10
        if b == 0:
            buckets["test"].append((art, sents))
        elif b == 1:
            buckets["valid"].append((art, sents))
        else:
            buckets["train"].append((art, sents))

    for name, items in buckets.items():
        path = os.path.join(DATA_DIR, name + ".txt")
        n_sent = 0
        n_tok = 0
        with open(path, "w", encoding="utf-8") as out:
            for _, sents in items:
                for s in sents:
                    out.write(" ".join(s) + "\n")
                    n_sent += 1
                    n_tok += len(s)
        stats["split"][name] = {"articles": len(items), "sentences": n_sent, "tokens": n_tok}
        print(f"{name}: {len(items)} articles, {n_sent} sentences, {n_tok} tokens")

    with open(os.path.join(DATA_DIR, "clean_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("decode-error lines dropped:", len(stats["decode_error_lines"]))
    print("duplicate articles removed:", len(dup_ids), dup_ids[:10])
    print("stats ->", os.path.join(DATA_DIR, "clean_stats.json"))


def _add_sentence(store, art, buf, stats):
    if not any(HANZI_OR_DIGIT.search(t) for t in buf):
        stats["sentences_pure_punct_dropped"] += 1
        return
    store.setdefault(art, []).append(tuple(buf))
    stats["sentences_kept"] += 1


if __name__ == "__main__":
    main()
