# -*- coding: utf-8 -*-
"""
4.2 表示学习实验：用 Gensim 在人民日报 1998 上半年语料上训练并对比
- word2vec(CBOW/SG) 与 FastText，多组维度/窗口/min_count；
- 记录训练时间、词表规模；用近邻、相似度、类比、OOV 检验向量质量；
- 把评测结果写入 results/vectors_eval.txt，模型存入 models/。
用法：python train_vectors.py [smoke]   # smoke=只跑小数据验证流程
"""
import os, sys, io, time, json, tempfile

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from gensim.models import Word2Vec, FastText

DATA = "data/rmrb_9801_06.seg.txt"
DATA_JIEBA = "data/rmrb_9801_raw_jieba.txt"
MODEL_DIR, RESULT_DIR = "models", "results"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

SMOKE = len(sys.argv) > 1 and sys.argv[1] == "smoke"
LIMIT = 20000 if SMOKE else None

OOV_WORDS = ["深度学习", "神经网络", "大语言模型", "微信", "智能手机"]
ANALOGY = ("中国", "北京", "美国")   # 中国:北京 :: 美国:?

def lines_of(path):
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if LIMIT and i >= LIMIT:
                break
            yield line.split()

def train(cfg):
    t0 = time.time()
    path = DATA if cfg.get("seg", "rmrb") == "rmrb" else DATA_JIEBA
    common = dict(vector_size=cfg["dim"], window=cfg["win"], min_count=cfg["mc"],
                  sg=cfg.get("sg", 0), epochs=cfg.get("ep", 5), workers=4, seed=42)
    if LIMIT:  # smoke：小数据时改用行数限制后的临时文件，保持 corpus_file 流式读取
        tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
        for line in lines_of(path):
            tmp.write(" ".join(line) + "\n")
        tmp.close()
        path = tmp.name
    if cfg["model"] == "fasttext":
        m = FastText(corpus_file=path, **common)
    else:
        m = Word2Vec(corpus_file=path, **common)
    dt = time.time() - t0
    return m, dt

def evaluate(m, is_ft):
    out = {}
    out["vocab"] = len(m.wv)
    out["most_similar_中国"] = [w for w, _ in m.wv.most_similar("中国", topn=5)]
    out["most_similar_经济"] = [w for w, _ in m.wv.most_similar("经济", topn=5)]
    out["sim_北京_上海"] = round(float(m.wv.similarity("北京", "上海")), 4)
    out["sim_中国_美国"] = round(float(m.wv.similarity("中国", "美国")), 4)
    out["sim_足球_体育"] = round(float(m.wv.similarity("足球", "体育")), 4)
    out["doesnt_match"] = m.wv.doesnt_match(["北京", "上海", "天津", "足球"])
    pos = [ANALOGY[1], ANALOGY[2]]
    neg = [ANALOGY[0]]
    out["analogy_中国北京_美国"] = [w for w, _ in m.wv.most_similar(positive=pos, negative=neg, topn=3)]
    oov = {}
    for w in OOV_WORDS:
        if is_ft:
            oov[w] = [x for x, _ in m.wv.most_similar(w, topn=3)]
        else:
            try:
                m.wv[w]
                oov[w] = "in_vocab!"
            except KeyError:
                oov[w] = "KeyError(不在词表)"
    out["oov"] = oov
    return out

def main():
    cfgs = [
        dict(tag="w2v_cbow_d100", model="w2v", dim=100, win=5, mc=5, sg=0),
        dict(tag="w2v_cbow_d300", model="w2v", dim=300, win=5, mc=5, sg=0),
        dict(tag="w2v_cbow_w10",  model="w2v", dim=100, win=10, mc=5, sg=0),
        dict(tag="w2v_cbow_mc20", model="w2v", dim=100, win=5, mc=20, sg=0),
        dict(tag="w2v_sg_d100",   model="w2v", dim=100, win=5, mc=5, sg=1),
        dict(tag="ft_sg_d100",    model="fasttext", dim=100, win=5, mc=5, sg=1),
        dict(tag="ft_sg_d100_jieba", model="fasttext", dim=100, win=5, mc=5, sg=1, seg="jieba"),
    ]
    results = []
    for cfg in cfgs:
        print(f"\n== 训练 {cfg['tag']}  params={ {k: v for k, v in cfg.items() if k not in ('tag',)} }")
        m, dt = train(cfg)
        is_ft = cfg["model"] == "fasttext"
        ev = evaluate(m, is_ft)
        ev["tag"], ev["train_sec"] = cfg["tag"], round(dt, 1)
        print(f"   用时 {dt:.1f}s | 词表 {ev['vocab']:,}")
        print(f"   中国近邻: {ev['most_similar_中国']}")
        print(f"   OOV: {ev['oov']}")
        results.append(ev)
        if not SMOKE:
            m.save(os.path.join(MODEL_DIR, cfg["tag"] + ".model"))
    with open(os.path.join(RESULT_DIR, "vectors_eval.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n评测结果已写入 results/vectors_eval.json（{'smoke 小数据' if SMOKE else '全量数据'}）")

if __name__ == "__main__":
    main()
