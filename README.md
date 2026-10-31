# LLM-course

《大语言模型与应用实践》（2026 秋，北大）课程作业仓库。

## 目录

- `hw1/` — **n-gram 训练与文本生成**（L1 第 125 张）
  - `L1作业报告.md`：完整实验报告（tag 清洗、KenLM 1–10 阶与自写 Stupid Backoff 对照、训练时间与资源、PPL/受限候选集排序/续写评测）
  - 脚本：`clean_corpus.py`（清洗）→ `count_ngrams.py`（SQLite 计数）→ `train_kenlm.py`（WSL 内 KenLM 训练）→ `eval_ppl.py` / `eval_rank.py` / `generate.py`（评测与续写），另有 `model_sb.py`（自写回退模型）、`make_report.py`（出图）
  - `data/`：清洗后的 train/valid/test 语料（人民日报 1998 01–06，含 clean_stats/count_times），可直接计数训练
  - `results/`：评测 JSON（ppl/ranking/generation/kenlm_train）与 5 张附图
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

### hw1（n-gram）

```bash
cd hw1
python clean_corpus.py      # 需要课程群分享的 1998-上半年-带音(1).rar 原文；data/ 下已含清洗结果，可跳过
python count_ngrams.py --db <你的路径>/ngrams.db      # 自写 Stupid Backoff 计数（SQLite）
python train_kenlm.py       # KenLM lmplz/build_binary，需在 WSL 自编译（-DKENLM_MAX_ORDER=10）
python eval_ppl.py --kenlm-dir <模型目录> --sb-db <你的路径>/ngrams.db
python eval_rank.py --kenlm-dir <模型目录> --sb-db <你的路径>/ngrams.db
python generate.py --kenlm-dir <模型目录> --sb-db <你的路径>/ngrams.db
```

自写 SB 的 SQLite 计数库（约 1.1GB）与 KenLM 二值模型（1–10 阶共约 3.8GB）体积较大，未入库，用脚本可重建；脚本默认路径指向训练机（`F:/ngram_work/`），运行时按需修改。详见 `hw1/L1作业报告.md` 第七节"环境与复现"。

## 环境

Windows 11 · Python 3.10.6 · gensim 4.4.0 · scikit-learn 1.7.2 · jieba 0.42.1 · datasets 5.1.0 · tokenizers 0.23.3
