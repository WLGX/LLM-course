# -*- coding: utf-8 -*-
"""
4.2 文本分类小实验：人民日报 9801 文章按"版次"分类，标签由少到多逐步展开
- 特征 A：word2vec(CBOW d300) 词向量均值  ->  LogisticRegression / LinearSVC
- 特征 B：gensim FastText 监督模式原生分类器（同样的分层 80/20 划分）
- 类数展开：2 类(01 vs 04，要闻 vs 副刊) -> 4 类 -> 6 类 -> 12 类（全部版次）
- 报告准确率/宏平均 F1，12 类给出混淆最大的类别对。
"""
import os, sys, io, time, json
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
from scipy.spatial.distance import cosine
from gensim.models import Word2Vec, FastText
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

MODEL_DIR, RESULT_DIR = "models", "results"
os.makedirs(RESULT_DIR, exist_ok=True)

def load_docs(path="data/rmrb_9801_docs.tsv"):
    labels, texts = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            lab, _, text = line.rstrip("\n").split("\t")
            labels.append(lab)
            texts.append(text.split())
    return labels, texts

def mean_vec_features(texts, wv, dim, idx=None):
    idx = idx if idx is not None else range(len(texts))
    X = np.zeros((len(idx), dim), dtype=np.float32)
    for r, i in enumerate(idx):
        vs = [wv[t] for t in texts[i] if t in wv]
        if vs:
            X[r] = np.mean(vs, axis=0)          # 词全部 OOV 的文档保持零向量
    return X

def main():
    labels, texts = load_docs()
    print(f"文档 {len(texts):,}，类别 {len(set(labels))}，分布 {dict(sorted(Counter(labels).items()))}")

    w2v = Word2Vec.load(os.path.join(MODEL_DIR, "w2v_cbow_d300.model"))
    rows = []

    for k in (2, 4, 6, 12):
        # 2 类：选主题差异最大的"01 要闻 vs 04 副刊"；之后按版次前缀扩展到 4/6/12 类
        keep = {"01", "04"} if k == 2 else {f"{i:02d}" for i in range(1, k + 1)}
        idx = np.array([i for i, l in enumerate(labels) if l in keep])
        yk = [labels[i] for i in idx]
        itr, ite = train_test_split(np.arange(len(idx)), test_size=0.2, random_state=42, stratify=yk)
        ytr, yte = [yk[i] for i in itr], [yk[i] for i in ite]
        print(f"\n== {k} 类：版次 {sorted(set(yk))} ==")

        # ---- 特征 A：CBOW d300 均值 ----
        Xa_tr = mean_vec_features(texts, w2v.wv, 300, idx[itr])
        Xa_te = mean_vec_features(texts, w2v.wv, 300, idx[ite])
        for clf_name, clf in (("LogReg", LogisticRegression(max_iter=2000, random_state=42)),
                              ("LinearSVC", LinearSVC(random_state=42))):
            clf.fit(Xa_tr, ytr)
            pred = clf.predict(Xa_te)
            rows.append(("meanVec", k, clf_name,
                         round(accuracy_score(yte, pred), 4),
                         round(f1_score(yte, pred, average="macro"), 4)))
            print(f"  meanVec     {clf_name:9s} acc={accuracy_score(yte, pred):.4f} "
                  f"macroF1={f1_score(yte, pred, average='macro'):.4f}")

        # ---- 特征 B：FastText 监督模式原生分类器（相同划分） ----
        # 注意 gensim 监督语料格式：每条是"平铺"列表 [标签, 词1, 词2, ...]
        t0 = time.time()
        m = FastText(vector_size=100, window=5, min_count=5, sg=1, epochs=20, seed=42)
        train_corpus = [[yk[i]] + texts[idx[i]] for i in itr]
        m.build_vocab(train_corpus)
        m.train(train_corpus, total_examples=len(train_corpus), epochs=m.epochs)
        lab_set = sorted(set(ytr))
        lab_vec = {l: m.wv[l] for l in lab_set}
        preds = []
        for i in ite:
            toks = [t for t in texts[idx[i]] if t in m.wv]
            v = np.mean([m.wv[t] for t in toks], axis=0) if toks else np.zeros(100)
            preds.append(min(lab_set, key=lambda l: cosine(lab_vec[l], v)))
        rows.append(("FastTextClf", k, "native",
                     round(accuracy_score(yte, preds), 4),
                     round(f1_score(yte, preds, average="macro"), 4)))
        print(f"  FastTextClf native     acc={accuracy_score(yte, preds):.4f} "
              f"macroF1={f1_score(yte, preds, average='macro'):.4f} (训练 {time.time()-t0:.0f}s)")

    # ---- 12 类混淆分析（特征A + LogReg） ----
    print("\n== 12 类混淆分析（meanVec + LogReg）==")
    Xtr, Xte, ytr, yte = train_test_split(
        mean_vec_features(texts, w2v.wv, 300), labels,
        test_size=0.2, random_state=42, stratify=labels)
    clf = LogisticRegression(max_iter=2000, random_state=42).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    cm = confusion_matrix(yte, pred, labels=[f"{i:02d}" for i in range(1, 13)])
    np.fill_diagonal(cm, 0)
    top = np.argsort(cm, axis=None)[-6:]
    pairs = sorted([(np.unravel_index(t, cm.shape), int(cm.reshape(-1)[t])) for t in top],
                   key=lambda x: -x[1])
    print("混淆前 6 对（真实版次→预测版次，次数）:",
          [(f"{a+1:02d}→{b+1:02d}", c) for (a, b), c in pairs])

    with open(os.path.join(RESULT_DIR, "classify_eval.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print("\n分类结果已写入 results/classify_eval.json")

if __name__ == "__main__":
    main()
