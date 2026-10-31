# -*- coding: utf-8 -*-
"""
P48 文本分段思考题 · 演示实验 1：句子分段（sentence segmentation）

两部分：
  A. 英文：句号歧义演示 —— 朴素按句号切 vs pySBD（规则方法代表）
     课件例子：Brown 语料只按句号切只有 93.20% 正确（因为缩写、小数、引号里的句号）。
  B. 中文：人民日报 1998 语料上做规则分句统计 —— 展示规范新闻文本中规则方法的效果，
     并用构造的"刁钻例子"展示规则方法在对话引号、省略号等场景的局限。

运行：PYTHONIOENCODING=utf-8 python seg_demo.py
输出：../results/seg_demo_log.txt（运行日志）与 ../results/seg_demo_stats.json（统计）
"""
import json
import os
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
# 原始语料（GBK、带标点）；4_2 清洗后的语料已去掉标点，不适合做分句演示
RAW_CORPUS = os.path.join(HERE, "..", "..", "..", "1998-上半年-带音(1)",
                          "1998-上半年-带音", "1998-01-2003版-带音.txt")

log_lines = []


def log(msg=""):
    log_lines.append(str(msg))
    print(msg)


def save_stats(d):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "seg_demo_stats.json"), "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    log(f"[saved] {os.path.join(RESULTS, 'seg_demo_stats.json')}")


# ---------------------------------------------------------------- A. 英文
def naive_en_split(text):
    """朴素规则：按 . ! ? 切分（最简单基线）"""
    return [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]


def demo_english():
    log("=" * 60)
    log("A. 英文句号歧义：朴素切分 vs pySBD")
    log("=" * 60)
    import pysbd
    seg_en = pysbd.Segmenter(language="en", clean=False)

    # 构造含歧义句号的例子（缩写、小数点、引号、时间）
    cases = [
        "Dr. Smith lives in New York. He arrived at 3.5 p.m. yesterday.",
        "I spoke with Mr. Jones about the U.S. policy. Then we left at 5.30 p.m.",
        "She said \"We will meet at 8 a.m.\" Then she left. We waited for her.",
        "The price is $19.99. e.g. this is a classic example. See St. Paul's.",
        "Version 3.14 was released on Jan. 2, 1998. Everyone was happy.",
    ]
    gold = [2, 2, 3, 3, 2]  # 人工判定的正确句数（作为演示的"标准答案"）

    naive_ok, pysbd_ok = 0, 0
    table = []
    for text, g in zip(cases, gold):
        n = len(naive_en_split(text))
        p = len(seg_en.segment(text))
        naive_ok += int(n == g)
        pysbd_ok += int(p == g)
        table.append({
            "text": text,
            "gold_sentences": g,
            "naive_split_count": n,
            "pysbd_split_count": p,
        })
        log(f"  文本: {text}")
        log(f"    正确句数 {g} | 朴素按句号 {n} | pySBD {p}")
        log(f"    pySBD 切分结果: {seg_en.segment(text)}")

    log(f"  [小结] 5 例中：朴素切分正确 {naive_ok}/5，pySBD 正确 {pysbd_ok}/5")
    log("  朴素方法把缩写/小数里的句号当成句边界（假阳性），pySBD 靠规则表+上下文消歧。")
    log("")
    return {"english_cases": table,
            "naive_correct": naive_ok, "pysbd_correct": pysbd_ok, "total": len(cases)}


# ---------------------------------------------------------------- B. 中文
CN_END = re.compile(r"[。！？…!?]+[」』”\"'）)]*")  # 句末标点后可能跟引号/括号


def rule_cn_split(paragraph):
    """简单中文规则分句：在句末标点处切。"""
    return [s for s in CN_END.split(paragraph) if s.strip()]


def demo_chinese():
    log("=" * 60)
    log("B. 中文规则分句：人民日报 1998 原始语料统计")
    log("=" * 60)
    n_lines = 3000  # 取前 3000 行原始语料
    with open(RAW_CORPUS, encoding="gbk", errors="replace") as f:
        raw_lines = [ln.strip() for ln in f.readlines()[:n_lines] if ln.strip()]

    # 去掉行首的语料编号（如 "19980101-01-001-004"），只留正文
    lines = [re.sub(r"^\S+?\s+", "", ln, count=1) for ln in raw_lines]
    lines = [re.sub(r"\{[^}]*\}", "", ln) for ln in lines]  # 去注音占位符 {shang5}
    text = "".join(lines)  # 拼成长文：不给换行信号，模拟"纯文本流"

    sents = rule_cn_split(text)
    sents = [re.sub(r"\s+", "", s) for s in sents if s.strip()]
    n_terminal = len(re.findall(r"[。！？…]", text))  # 文本中的句末标点数
    lens = [len(s) for s in sents]

    log(f"  读入 {len(lines)} 行原始语料（GBK），拼成长文 {len(text)} 字符")
    log(f"  文中句末标点（。！？…）共 {n_terminal} 个；规则切出句子 {len(sents)} 句")
    log(f"  差值说明：长文里规则在标点处切得基本干净（新闻文本几乎无缩写/小数干扰），"
        f"少量出入来自标题行无标点（切不出）与引号内标点。")
    log(f"  句子长度（字符）: 均值 {sum(lens)/len(lens):.1f}，"
        f"中位数 {sorted(lens)[len(lens)//2]}，最长 {max(lens)}，最短 {min(lens)}")
    log("  例句（前 5 句）:")
    for s in sents[:5]:
        log(f"    | {s[:60]}")
    log("  新闻语料标点规范 → 规则分句几乎无歧义；真正难的是网文对话、引号嵌套、省略号等场景。")

    log("")
    log("-" * 60)
    log("B2. 构造刁钻例子：规则方法的局限")
    log("-" * 60)
    tricky = [
        # 1. 引号嵌套：内层单引号里的句号被误当句末
        "我对他说：‘你好。’然后他就笑了。",
        # 2. 省略号在句中（不是句末）
        "他说……然后转身走了。",
        # 3. 网文对话：没有句末标点，靠换行分句
        "他问\n吃了吗\n吃了\n",
    ]
    for t in tricky:
        ss = rule_cn_split(t)
        log(f"  原文: {t!r}")
        log(f"  规则切分({len(ss)} 句): {ss}")
    log("  例1/例2：规则把引号内句号、句中省略号当句末 → 一句话被拆成两截；")
    log("  例3：网文'一句一段'靠换行分句，纯标点规则完全看不到边界 → 三段被并成一段。")
    log("  结论：规则方法要补三类信息——引号配对、省略号上下文、换行边界；")
    log("      学习式分句器（序列标注，如 LTP / BERT 类）把'是否句边界'当每个字符的分类问题，")
    log("      能利用上下文，天然覆盖这些情况。")
    return {
        "lines_read": len(lines),
        "text_chars": len(text),
        "terminal_marks": n_terminal,
        "sentences": len(sents),
        "mean_sent_len": round(sum(lens) / len(lens), 1),
        "median_sent_len": sorted(lens)[len(lens) // 2],
        "max_sent_len": max(lens),
        "min_sent_len": min(lens),
        "tricky_examples": len(tricky),
    }


# ---------------------------------------------------------------- C. 玩具分段模型
def demo_toy_model(lines):
    """玩具"学习式"分段模型：字符级二分类（句边界 / 非句边界）。

    - 特征：目标字符前后 2 字符的窗口 + 该字符是否句末标点/引号；
    - 标签：由规则分句器在新闻语料上自动生成（弱监督——人工标注不可行时的一种做法）；
    - 模型：sklearn LogisticRegression（真正的系统会用 BiLSTM-CRF / BERT 序列标注）。
    演示目的：走通"自己写一个分段模型"的完整流程（特征、标签、训练、评测），
    并诚实指出：在标点规范的新闻文本上学习模型 ≈ 规则；它真正的价值在规则失效的领域文本。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    text = "".join(lines)
    FEAT_OFFS = (-2, -1, 0, 1, 2)
    PUNCT_SET = set("。！？…!?")

    def feats(pos, s):
        out = []
        for o in FEAT_OFFS:
            j = pos + o
            out.append(ord(s[j]) % 4096 if 0 <= j < len(s) else -1)
        c = s[pos]
        out.append(1 if c in PUNCT_SET else 0)
        out.append(1 if c in "”’」』）)]" else 0)
        out.append(1 if pos + 1 < len(s) and s[pos + 1] in "\n\r" else 0)
        out.append(1 if pos > 0 and s[pos - 1] in "\n\r" else 0)
        return out

    def labels(s):
        lab = [0] * len(s)
        for m in CN_END.finditer(s):
            lab[m.end() - 1] = 1
        return lab

    # 80% 训练 / 20% 测试（按字符切分）
    cut = int(len(text) * 0.8)
    Xtr = [feats(i, text) for i in range(cut)]
    ytr = labels(text[:cut])
    Xte = [feats(i, text) for i in range(cut, len(text))]
    yte = labels(text[cut:])

    clf = LogisticRegression(max_iter=500, C=0.1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred)
    # 规则基线在测试集上的表现（标签本身由规则生成，故规则=100%，仅供对照说明）
    rule_pred = yte
    log("=" * 60)
    log("C. 玩具分段模型：字符级二分类（LogisticRegression）")
    log("=" * 60)
    log(f"  训练字符 {cut:,} / 测试字符 {len(Xte):,}（新闻语料，弱监督标签来自规则）")
    log(f"  测试集句边界 F1 = {f1:.3f}（规则自身作为上界 = 1.000）")
    log("  模型学到的正是规则逻辑：句末标点后是边界、引号配对后是边界；")
    log("  在新闻文本上两者等价；差异要在网文/口语/无标点文本上才显现（见 B2）。")
    return {"train_chars": cut, "test_chars": len(Xte), "f1": round(float(f1), 4)}


def main():
    stats = {"english": demo_english()}
    ch = demo_chinese()
    stats["chinese"] = ch
    # 复用 demo_chinese 读入的语料训练玩具模型
    with open(RAW_CORPUS, encoding="gbk", errors="replace") as f:
        raw_lines = [ln.strip() for ln in f.readlines()[:3000] if ln.strip()]
    lines = [re.sub(r"^\S+?\s+", "", ln, count=1) for ln in raw_lines]
    lines = [re.sub(r"\{[^}]*\}", "", ln) for ln in lines]
    stats["toy_model"] = demo_toy_model(lines)
    save_stats(stats)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "seg_demo_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[saved] log -> {os.path.join(RESULTS, 'seg_demo_log.txt')}")


if __name__ == "__main__":
    main()
