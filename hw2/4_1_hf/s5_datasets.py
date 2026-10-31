# -*- coding: utf-8 -*-
"""
HF NLP Course 第 5 节（Datasets）练习
要点：语料怎么"读进去"——加载、切片、map 批量处理、动态填充、缓存。
运行：python s5_datasets.py | tee ../outputs 由外层重定向，本脚本只打印。
"""
import os, sys, io, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # 本机直连 huggingface.co 不通，走镜像

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEP = "=" * 70

# ---------- 1. 加载数据集（GLUE / MRPC：句子对是否同义，用于二分类微调） ----------
t0 = time.time()
# 新版 datasets 中 glue 的正规命名空间为 nyu-mll/glue（旧写法 "glue" 会解析失败）
raw_datasets = load_dataset("nyu-mll/glue", "mrpc")
print(SEP)
print(f"[1] load_dataset('glue','mrpc') 用时 {time.time()-t0:.1f}s")
print(raw_datasets)
# 取一小份用于练习（课程里 tokenized_datasets 直接对全部数据 map，这里同样处理但打印只取切片）
splits = ["train", "validation", "test"]
print("各 split 条数:", {k: raw_datasets[k].num_rows for k in splits})

# ---------- 2. 切片与元素访问 ----------
print(SEP)
print("[2] 切片与元素访问")
print("raw_datasets['train'][0] =", raw_datasets["train"][0])
print("raw_datasets['train'][:2]['sentence1'][1] =", raw_datasets["train"][:2]["sentence1"][1])
print("列名 =", raw_datasets["train"].column_names)
print("特征 schema =", raw_datasets["train"].features)

# ---------- 3. map() 批量分词 ----------
print(SEP)
print("[3] map() 批量分词")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
print("加载 AutoTokenizer:", checkpoint, "| 词表大小:", tokenizer.vocab_size)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

t0 = time.time()
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"map(batched=True) 用时 {time.time()-t0:.1f}s")
print("tokenized 列名 =", tokenized_datasets["train"].column_names)

# ---------- 4. 动态填充（DataCollatorWithPadding） ----------
print(SEP)
print("[4] 动态填充（每个 batch 内按最长句填充，而非全数据集统一长度）")
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
samples = [tokenized_datasets["train"][i] for i in range(8)]
for s in samples:
    s.pop("idx", None); s.pop("sentence1", None); s.pop("sentence2", None)
batch = data_collator(samples)
print("batch keys =", list(batch.keys()))
print("各 key 形状 =", {k: v.shape for k, v in batch.items()})
print("attention_mask[0] =", batch["attention_mask"][0].tolist())
print("input_ids[0] =", batch["input_ids"][0].tolist())
print("labels =", batch["labels"].tolist())

# ---------- 5. 缓存：第二次 map 直接复用缓存文件 ----------
print(SEP)
print("[5] 缓存复用（同一参数再 map 一次，速度应明显更快）")
t0 = time.time()
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(f"再次 map 用时 {time.time()-t0:.3f}s（应远小于首次，说明读的是缓存）")

# ---------- 6. 把本地语料"读进去"：datasets 加载本地文本 ----------
print(SEP)
print("[6] 用 datasets 加载本地语料（课程语料如何读进去）")
corpus = r"../../1998-上半年-带音(1)/1998-上半年-带音/1998-01-2003版-带音.txt"
local = load_dataset("text", data_files={"rmrb9801": corpus}, encoding="gbk")
print(local)
print("第一条前 120 字 =", local["rmrb9801"][0]["text"][:120])
print("条数 =", local["rmrb9801"].num_rows)
