# -*- coding: utf-8 -*-
"""
HF NLP Course 第 6 节（Tokenizers）练习
要点：分词流水线（规范化→预切分→模型→后处理）；从零训练 BPE/WordPiece/Unigram；
     快速分词器的批量编码、padding/truncation；中文语料上的表现对比。
"""
import os, sys, io, time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers, decoders
from transformers import AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

SEP = "=" * 70
CORPUS = r"../../1998-上半年-带音(1)/1998-上半年-带音/1998-01-2003版-带音.txt"

def corpus_lines(path, limit=20000):
    """把 GBK 的人民日报语料读成迭代器（每行去掉行首的文章编号前缀）。"""
    with open(path, encoding="gbk") as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            parts = line.split()
            if len(parts) > 1:  # 第一个字段是编号，跳过
                yield " ".join(parts[1:])

# ---------- 1. 加载现成分词器：慢速/快速、子词原理 ----------
print(SEP)
print("[1] 加载现成分词器")
for name in ["bert-base-cased", "gpt2", "xlm-roberta-base"]:
    tok = AutoTokenizer.from_pretrained(name)
    print(f"{name}: vocab={tok.vocab_size} use_fast={tok.is_fast}")
    # tokenizers 后端查看
    t = tok.backend_tokenizer
    print("   model =", t.model.__class__.__name__, "| pre_tokenizer =", t.pre_tokenizer.__class__.__name__ if t.pre_tokenizer else None,
          "| normalizer =", t.normalizer.__class__.__name__ if t.normalizer else None)
    print("   编码示例:", tok.tokenize("Am I annoying? Lowercase the word Huggingface!"))
    print("   ids 示例:", tok("Am I annoying?")["input_ids"])

# ---------- 2. 从零训练 BPE（GPT-2 风格的字节级 BPE） ----------
print(SEP)
print("[2] 从零在人民日报语料上训练 BPE（字节级，vocab=5000）")

def build_bpe():
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()          # 全角→半角等规范化
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tr = trainers.BpeTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tok.train_from_iterator(corpus_lines(CORPUS), trainer=tr, length=20000)
    return tok

t0 = time.time()
bpe = build_bpe()
bpe.decoder = decoders.ByteLevel()   # 把字节级 token 还原成可见文本
print(f"训练用时 {time.time()-t0:.1f}s, vocab={bpe.get_vocab_size()}")
enc = bpe.encode("在阳光明媚的五月，我们学校胜利召开了运动会。")
print("编码示例(每片一个 token):", [bpe.decode([i]) for i in enc.ids])
print("OOV 字符（🤗）:", bpe.encode("🤗").tokens, "-> ids", bpe.encode("🤗").ids)

# ---------- 3. 训练 WordPiece ----------
print(SEP)
print("[3] 训练 WordPiece（whitespace 预切分）")
# 说明：人民日报语料的全角字符集就有 8000+ 个不同字符，
# 第一次用 vocab=5000 时目标词表小于初始字符表，一个合并都做不了（退化成逐字切分）。
# 这里把词表提到 12000，观察真正的子词合并。
def build_wp(vocab_size=12000):
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()   # 语料已按词切好
    tr = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tok.train_from_iterator(corpus_lines(CORPUS), trainer=tr, length=20000)
    return tok

t0 = time.time()
wp = build_wp()
print(f"训练用时 {time.time()-t0:.1f}s, vocab={wp.get_vocab_size()}")
print("编码示例:", wp.encode("在阳光明媚的五月，我们学校胜利召开了运动会。").tokens)
print("OOV 字符（🤗）:", wp.encode("🤗").tokens, "-> ids", wp.encode("🤗").ids)

# ---------- 4. 训练 Unigram ----------
print(SEP)
print("[4] 训练 Unigram（vocab=5000）")

def build_uni():
    tok = Tokenizer(models.Unigram())
    tok.normalizer = normalizers.NFKC()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tr = trainers.UnigramTrainer(vocab_size=5000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
                                 unk_token="[UNK]")
    tok.train_from_iterator(corpus_lines(CORPUS), trainer=tr, length=20000)
    return tok

t0 = time.time()
uni = build_uni()
print(f"训练用时 {time.time()-t0:.1f}s, vocab={uni.get_vocab_size()}")
print("编码示例:", uni.encode("在阳光明媚的五月，我们学校胜利召开了运动会。").tokens)
print("OOV 字符（🤗）:", uni.encode("🤗").tokens, "-> ids", uni.encode("🤗").ids)

# ---------- 5. 三种分词器对比：同一句话的切分结果 ----------
print(SEP)
print("[5] 同一句话的三种切分对比")
sent = "我们学校胜利召开了运动会"
for nm, t in [("BPE", bpe), ("WordPiece", wp), ("Unigram", uni)]:
    enc = t.encode(sent)
    if nm == "BPE":
        print(f"{nm:9s} pieces({len(enc.ids)}) =", [bpe.decode([i]) for i in enc.ids])
    else:
        print(f"{nm:9s} tokens({len(enc.tokens)}) =", enc.tokens)

# ---------- 6. 快速分词器批量编码 + padding/truncation ----------
print(SEP)
print("[6] 快速分词器批量编码与 padding/truncation")
fast = AutoTokenizer.from_pretrained("bert-base-uncased")  # 默认 use_fast=True
sents = ["Hello!", "This is a much longer sentence that goes on and on.", "short"]
enc = fast(sents, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
print("input_ids shape =", enc["input_ids"].shape)
print("attention_mask =")
print(enc["attention_mask"].tolist())
t0 = time.time()
for _ in range(200):
    fast(sents, padding=True, truncation=True)
print(f"批量编码 200 次（3 句/批）用时 {time.time()-t0:.2f}s")

# ---------- 7. 保存训练好的分词器 ----------
print(SEP)
print("[7] 保存分词器")
os.makedirs("trained_tokenizers", exist_ok=True)
bpe.save("trained_tokenizers/rmrb_bpe.json")
wp.save("trained_tokenizers/rmrb_wordpiece.json")
uni.save("trained_tokenizers/rmrb_unigram.json")
print("已保存 trained_tokenizers/rmrb_{bpe,wordpiece,unigram}.json")
