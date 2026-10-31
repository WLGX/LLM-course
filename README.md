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

- `hw3/` — **BERT 古文断句标点 + 设计下一代 BERT**（L2 part2 第 68、90–91 张）
  - `4_4/4_4报告.md`：完整实验报告（人民日报 9801 字级标点标注、bert-base-chinese 微调、长文 batch 解码实测与朴素滑窗对照、古文句读定性演示）
  - `4_4/` 脚本：`data_prep.py`（清洗+标注）→ `train_bert.py`（微调训练/评估）→ `decode_long.py`（任务二两种解码方案实测）→ `demo_gushi.py` / `make_figs.py`
  - `4_5/下一代BERT设计.md`：3 亿参数 / 8K 上下文设计文档（架构题/任务题/证据题 + 资源账本）
  - `4_5/mini_ablation.py`：小规模验证实验（微型编码器四变体消融，MLM 损失 vs 断句探针双指标）

- `hw4/` — **唐诗宋词英译提示实验 + 系统提示词泄露/攻击调研**（L3 part1 第 89–90、48–49 张）
  - `4_1_诗词英译提示实验/`：三提示版本（直接翻译/任务标准/边界示例）× 16 首诗 × 2 重复的真实 API 实验（deepseek-chat），预注册评分协议、盲评打分、改对/新增错误/成本对比，`实验报告.md` + `results/` 原始输出
  - `4_2_提示词泄露与攻击/`：15 条攻击用例（注入/提取/越狱/对照）实测 + OWASP/指令层级文献综述，`调研报告.md`
  - `piazza/`：两份提交帖草稿；API key 放本地 `.env`（已在 .gitignore），不入库

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

自写 SB 的 SQLite 计数库（约 1.1GB）与 KenLM 二值模型（1–10 阶共约 3.8GB）体积较大，未入库，用脚本可重建；脚本默认把它们放在 `hw1/models/`（`ngrams.db` 与 `kenlm/`），训练完成后评测与续写脚本即可直接运行。详见 `hw1/L1作业报告.md` 第七节"环境与复现"。

## 环境

Windows 11 · Python 3.10.6 · gensim 4.4.0 · scikit-learn 1.7.2 · jieba 0.42.1 · datasets 5.1.0 · tokenizers 0.23.3
