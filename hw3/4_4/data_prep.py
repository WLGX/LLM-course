# -*- coding: utf-8 -*-
"""
4.4 数据准备：从《人民日报》1998 上半年（9801 语料，2003 分词版带音）构造
"断句标点"字级标注数据。

输入：F:/LLM-course/1998-上半年-带音(1)/1998-上半年-带音/1998-0?-2003版-带音.txt
      每行格式：<编号> <空格分隔的分词，标点独占一词>
输出：hw3/4_4/data/{train,valid,test}.jsonl  每行 {"text": "...", "labels": "O O ， O ..."}
      hw3/4_4/data/label_vocab.json
      hw3/4_4/data/clean_stats.json

标注约定：对每个字符，label = 紧跟其后的下一个字符（若该字符属于主标点集），否则 O。
主标点集：，。、？！：；""…（左/右引号各自归并为 “ 与 ” 两类）。
其余标点（《》（）——· 等）保留在文本中但不作为预测目标（label 记 O），
此设定与局限性写入报告。
清洗规则：仅把全角数字/字母/空格归一为半角（不动标点）；去掉行首编号。
切块规则：按句（以。？！… 为句界）贪心装入 max_chars=128 的块；单句超过 128 字
时在句中截断（罕见，计数写入统计）。块内不足 32 字符丢弃。
"""
import json
import os
import random
import re
from collections import Counter

SRC_DIR = r"F:/LLM-course/1998-上半年-带音(1)/1998-上半年-带音"
OUT_DIR = r"F:/LLM-course/hw3/4_4/data"
MAX_CHARS = 128
MIN_CHARS = 32

# 主标点集（预测目标）
PUNCT = set("，。、？！：；\u201c\u201d\u2018\u2019…")
END_PUNCT = set("。？！…")
# 归并：右引号 -> ”，左引号 -> “
RQUOTE = set("\u201d\u2019\"'")
LQUOTE = set("\u201c\u2018")

def fw2hw(ch: str) -> str:
    """仅归一全角空格/数字/字母为半角；标点保持全角不动。"""
    o = ord(ch)
    if o == 0x3000:
        return " "
    if 0xFF10 <= o <= 0xFF19 or 0xFF21 <= o <= 0xFF3A or 0xFF41 <= o <= 0xFF5A:
        return chr(o - 0xFEE0)
    return ch

def normalize(s: str) -> str:
    # 去除带音标注（{na4} / [pin1] 形式，仅 ASCII 括号为注音包装）与分隔符 〓
    s = re.sub(r"[{\[][^{}\[\]]*[}\]]", "", s)
    s = s.replace("\u3013", "")
    return "".join(fw2hw(c) for c in s)

def merge_quote(w: str) -> str:
    if w in RQUOTE:
        return "\u201d"
    if w in LQUOTE:
        return "\u201c"
    return w

def label_line(seg_words: str):
    """由分词串（标点独占一词）还原字串，并给出每个字的标签（归并引号类）。"""
    chars, labels = [], []
    for w in seg_words.split():
        if w and all(c in PUNCT for c in w):
            # 标点词（含 "……" 这类多字标点词）：给上一个字打上首字标签
            if labels and labels[-1] == "O":
                labels[-1] = merge_quote(w[0])
        else:
            for c in w:
                if c in PUNCT:
                    # 混合词内的标点：撇号（如 ’98）保留为文本，其余转为前一字的标签
                    if c in "\u2018\u2019" and any(x.isalnum() for x in w):
                        chars.append(c)
                        labels.append("O")
                    elif labels and labels[-1] == "O":
                        labels[-1] = merge_quote(c)
                else:
                    chars.append(c)
                    labels.append("O")
    return chars, labels

def chunk_stream(chars, labels):
    """按句贪心装入 MAX_CHARS 块；单句超长时句中截断并计数。"""
    chunks = []
    cut_oversize = 0
    buf_c, buf_l = [], []
    def emit():
        nonlocal buf_c, buf_l
        if len(buf_c) >= MIN_CHARS:
            chunks.append(("".join(buf_c[:MAX_CHARS]), " ".join(buf_l[:MAX_CHARS])))
        buf_c, buf_l = [], []
    for c, l in zip(chars, labels):
        buf_c.append(c)
        buf_l.append(l)
        if l in END_PUNCT:  # 该字之后是句尾标点 -> 句界
            if len(buf_c) >= MAX_CHARS:
                cut_oversize += 1
                emit()
        elif len(buf_c) >= MAX_CHARS:  # 超长句：句中截断
            cut_oversize += 1
            emit()
    if buf_c:
        emit()
    return chunks, cut_oversize

def main():
    random.seed(42)
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".txt"))
    print("files:", files)
    total_lines = 0
    total_chars = 0
    cut_oversize = 0
    chunks = []
    for fn in files:
        with open(os.path.join(SRC_DIR, fn), encoding="gb18030", errors="replace") as f:
            for line in f:
                line = line.rstrip("\n")
                total_lines += 1
                if not line.strip():
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                seg = normalize(parts[1])
                chars, labels = label_line(seg)
                total_chars += len(chars)
                cs, cut = chunk_stream(chars, labels)
                chunks.extend(cs)
                cut_oversize += cut

    random.shuffle(chunks)
    n = len(chunks)
    n_train = int(n * 0.9)
    n_valid = int(n * 0.05)
    splits = {
        "train": chunks[:n_train],
        "valid": chunks[n_train:n_train + n_valid],
        "test": chunks[n_train + n_valid:],
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    label_counter = Counter()
    for name, cs in splits.items():
        with open(os.path.join(OUT_DIR, name + ".jsonl"), "w", encoding="utf-8") as f:
            for text, labels in cs:
                label_counter.update(labels.split())
                f.write(json.dumps({"text": text, "labels": labels}, ensure_ascii=False) + "\n")
    vocab = {"O": 0}
    for lbl, _ in label_counter.most_common():
        if lbl not in vocab:
            vocab[lbl] = len(vocab)
    with open(os.path.join(OUT_DIR, "label_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    stats = {
        "lines_read": total_lines,
        "total_chars": total_chars,
        "chunks": {"train": n_train, "valid": n_valid, "test": n - n_train - n_valid},
        "cut_oversize_sentences": cut_oversize,
        "label_counts": dict(label_counter.most_common()),
        "vocab": vocab,
        "max_chars": MAX_CHARS,
        "min_chars": MIN_CHARS,
    }
    with open(os.path.join(OUT_DIR, "clean_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: stats[k] for k in ("lines_read", "total_chars", "chunks", "cut_oversize_sentences")}, ensure_ascii=False))
    print("label_counts:", dict(label_counter.most_common()))

if __name__ == "__main__":
    main()
