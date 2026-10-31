# LLM-course

《大语言模型与应用实践》（2026 秋，北大）课程作业仓库。

## 目录

- `hw2/` — **Gensim 表示学习 + FastText 文本分类**（L2 part1 第 137 张）
  - `L2作业报告.md`：完整实验报告（数据清洗、7 组词向量训练对比、2/4/6/12 类版次分类、论文与 Gensim 源码对照）
  - `4_1_hf/`：预备学习——Hugging Face NLP Course 第 5 节（Datasets）、第 6 节（Tokenizers）练习脚本与输出，含从零训练的 BPE/WordPiece/Unigram 分词器
  - `4_2/`：语料清洗、词向量训练、文本分类实验的脚本、数据与结果（含训练日志与评测 JSON）

## 复现

```bash
pip install gensim scikit-learn jieba datasets tokenizers transformers
cd hw2/4_2
python clean_corpus.py      # 需要课程群分享的 1998-上半年-带音(1).rar 原文
python train_vectors.py     # 7 组 word2vec/FastText，全量约 10 分钟
python classify.py          # 版次分类 2/4/6/12 类
```

`data/` 下已含清洗后的语料（人民日报 1998 01–06，北大分词），可直接训练。
训练出的模型文件较大（FastText 子词向量约 1.6GB），未入库，用 `train_vectors.py` 可一键重建。

## 环境

Windows 11 · Python 3.10.6 · gensim 4.4.0 · scikit-learn 1.7.2 · jieba 0.42.1 · datasets 5.1.0 · tokenizers 0.23.3
