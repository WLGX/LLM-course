# -*- coding: utf-8 -*-
"""
4.2 数据准备：清洗人民日报 1998 上半年带音语料
- 去掉拼音标注 {shang5}、行首编号；NFKC 规范化全角字符；去掉纯标点/纯数字 token。
- 产出：
  data/rmrb_9801_06.seg.txt   1998-01~06 全部句子（表示学习训练语料，语料自带北大分词）
  data/rmrb_9801_docs.tsv     1998-01 按文章聚合（分类实验用，标签=版次）
  data/rmrb_9801_raw_jieba.txt 1998-01 原文拼接后用 jieba 分词（分词器对比用）
"""
import os, re, unicodedata
from collections import Counter

import jieba

SRC_DIR = r"../../1998-上半年-带音(1)/1998-上半年-带音"
OUT_DIR = "data"
os.makedirs(OUT_DIR, exist_ok=True)

PINYIN = re.compile(r"\{[a-zA-Z]+[0-9]?\}")           # {shang5}
PURE_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)      # 纯标点/符号
PURE_NUM = re.compile(r"^\d+([\.\-:]\d+)*$")          # 纯数字（含 12:30、3.14）

def clean_token(tok):
    tok = PINYIN.sub("", tok)             # 去拼音
    if tok == "":
        return None
    tok = unicodedata.normalize("NFKC", tok)  # 全角数字/字母→半角
    if PURE_PUNCT.match(tok) or PURE_NUM.match(tok):
        return None
    return tok

def clean_line(line):
    parts = line.split()
    if len(parts) < 2:
        return None, None
    fid = parts[0]                        # 19980101-01-001-001
    toks = [t for t in (clean_token(w) for w in parts[1:]) if t]
    return fid, toks

def main():
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".txt"))
    print("输入文件:", files)

    n_sent = n_tok = 0
    vocab = Counter()
    docs = {}                              # 文章号 -> (版次, [tokens])
    doc_order = []
    all_lines = []
    raw_texts = []                         # 原文拼接（jieba 用）

    for fn in files:
        raw = open(os.path.join(SRC_DIR, fn), "rb").read()
        # 语料里混有少量"裸"0xAD 字节（软连字符），它既是 GBK 合法尾字节、又会单独出现，
        # 因此不能按字节直接删，只能先宽松解码再把替换符清掉。
        text = raw.decode("gb18030", errors="replace")
        n_bad = text.count("\ufffd")
        if n_bad:
            print(f"  {fn}: 宽松解码替换裸 0xAD 等非法字节 x{n_bad}")
            text = text.replace("\ufffd", "")
        for line in text.splitlines():
            fid, toks = clean_line(line.strip())
            if not fid:
                continue
            n_sent += 1
            n_tok += len(toks)
            vocab.update(toks)
            all_lines.append(" ".join(toks))
            art = "-".join(fid.split("-")[:3])
            ed = fid.split("-")[1]
            if not fn.startswith("1998-01"):
                continue               # 分类实验只用 9801
            if art not in docs:
                docs[art] = [ed, []]
                doc_order.append(art)
            docs[art][1].extend(toks)
            raw_texts.append("".join(toks))

    with open(os.path.join(OUT_DIR, "rmrb_9801_06.seg.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_lines) + "\n")
    print(f"表示学习语料(01~06)：句子 {n_sent:,}，词次 {n_tok:,}，词型 {len(vocab):,}")

    with open(os.path.join(OUT_DIR, "rmrb_9801_docs.tsv"), "w", encoding="utf-8") as f:
        f.write("label\tdoc_id\ttext\n")
        for art in doc_order:
            ed, toks = docs[art]
            f.write(f"{ed}\t{art}\t{' '.join(toks)}\n")
    edc = Counter(docs[a][0] for a in docs)
    print(f"分类语料(9801)：文章 {len(docs):,}，版次分布 {dict(sorted(edc.items()))}")

    # jieba 对比版：把语料已有分词连起来再让 jieba 切（观察两种分词器的差异）
    with open(os.path.join(OUT_DIR, "rmrb_9801_raw_jieba.txt"), "w", encoding="utf-8") as f:
        for text in raw_texts:
            f.write(" ".join(jieba.cut(text)) + "\n")
    print("jieba 分词版已写出（用于分词器对比的一次训练）")

    # 抽样看清洗前后对比
    print("\n清洗示例（前 2 条）：")
    for line in all_lines[:2]:
        print("  ", line[:80])

if __name__ == "__main__":
    main()
