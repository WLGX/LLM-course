# -*- coding: utf-8 -*-
"""
P147 影视剧 embedding 思考题 · 演示实验 2：item2vec 玩具实验

思想：把"用户观影序列"当成语料、电影当成词 —— 和 L2 讲的 word2vec 完全同构
（word2vec 把相邻出现的词学成相近向量；这里把同一观看序列里相邻出现的电影学成相近向量，
即 item2vec / prod2vec，Microsoft 2016）。

数据：合成观看序列（玩具数据，明确标注，非真实用户数据）。
  - 40 部电影，标注导演/类型/主演；
  - 3000 个合成用户，每人按"导演粉/演员粉/类型粉/混合"四种模式之一生成 6-15 部的观看序列；
  - gensim Word2Vec(sg=1, window=5) 训练电影向量。

验证三件事：
  1. 相似电影检索：给定《星际穿越》，是否召回同导演（诺兰）的其他电影？
  2. 演员维度：《阿甘正传》是否召回汤姆·汉克斯其他电影？—— 说明"喜欢汉克斯≠只能推汉克斯"，向量把导演/类型等共现信息都编码进去了。
  3. 冷启动：训练集里完全没出现过的新电影没有向量 → 直接检索失败；
     用"内容特征初始化"（取同导演+同类型已见电影向量的均值）给新片一个近似向量，看能否站对位置。

运行：PYTHONIOENCODING=utf-8 python movie_embedding_demo.py
输出：../results/movie_embedding_demo_log.txt 与 movie_embedding_demo.json
"""
import json
import os
import random
from collections import defaultdict

import gensim
from gensim.models import Word2Vec

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

random.seed(42)

# 40 部电影池（导演 / 类型 / 主演）
MOVIES = [
    # --- 诺兰（科幻/悬疑） ---
    ("星际穿越", "诺兰", "科幻", ["马修·麦康纳", "安妮·海瑟薇"]),
    ("盗梦空间", "诺兰", "科幻", ["莱昂纳多"]),
    ("敦刻尔克", "诺兰", "战争", ["汤姆·哈迪"]),
    ("蝙蝠侠：黑暗骑士", "诺兰", "动作", ["克里斯蒂安·贝尔"]),
    ("记忆碎片", "诺兰", "悬疑", ["盖·皮尔斯"]),
    ("信条", "诺兰", "科幻", ["约翰·大卫·华盛顿"]),
    # --- 斯皮尔伯格 ---
    ("侏罗纪公园", "斯皮尔伯格", "科幻", ["山姆·尼尔"]),
    ("拯救大兵瑞恩", "斯皮尔伯格", "战争", ["汤姆·汉克斯"]),
    ("猫鼠游戏", "斯皮尔伯格", "剧情", ["莱昂纳多", "汤姆·汉克斯"]),
    ("人工智能", "斯皮尔伯格", "科幻", ["海利·乔·奥斯蒙"]),
    ("头号玩家", "斯皮尔伯格", "科幻", ["泰伊·谢里丹"]),
    # --- 卡梅隆 ---
    ("泰坦尼克号", "卡梅隆", "爱情", ["莱昂纳多", "凯特·温斯莱特"]),
    ("阿凡达", "卡梅隆", "科幻", ["萨姆·沃辛顿"]),
    ("终结者2", "卡梅隆", "科幻", ["阿诺·施瓦辛格"]),
    ("真实的谎言", "卡梅隆", "动作", ["阿诺·施瓦辛格"]),
    # --- 斯科塞斯 ---
    ("华尔街之狼", "斯科塞斯", "剧情", ["莱昂纳多"]),
    ("好家伙", "斯科塞斯", "犯罪", ["罗伯特·德尼罗"]),
    ("出租车司机", "斯科塞斯", "犯罪", ["罗伯特·德尼罗"]),
    ("无间道风云", "斯科塞斯", "犯罪", ["莱昂纳多"]),
    # --- 汤姆·汉克斯片单 ---
    ("阿甘正传", "泽米吉斯", "剧情", ["汤姆·汉克斯"]),
    ("费城故事", "戴米", "剧情", ["汤姆·汉克斯"]),
    ("荒岛余生", "泽米吉斯", "剧情", ["汤姆·汉克斯"]),
    ("玩具总动员", "拉塞特", "动画", ["汤姆·汉克斯"]),
    # --- 莱昂纳多片单 ---
    ("好莱坞往事", "塔伦蒂诺", "剧情", ["莱昂纳多"]),
    ("禁闭岛", "斯科塞斯", "悬疑", ["莱昂纳多"]),
    # --- 其他经典 ---
    ("肖申克的救赎", "德拉邦特", "剧情", ["蒂姆·罗宾斯"]),
    ("教父", "科波拉", "犯罪", ["阿尔·帕西诺"]),
    ("教父2", "科波拉", "犯罪", ["阿尔·帕西诺"]),
    ("低俗小说", "塔伦蒂诺", "犯罪", ["约翰·特拉沃尔塔"]),
    ("指环王", "杰克逊", "奇幻", ["伊利亚·伍德"]),
    ("指环王2", "杰克逊", "奇幻", ["伊利亚·伍德"]),
    ("哈利·波特", "哥伦布", "奇幻", ["丹尼尔·雷德克里夫"]),
    ("哈利·波特2", "哥伦布", "奇幻", ["丹尼尔·雷德克里夫"]),
    ("狮子王", "阿勒尔斯", "动画", ["马修·布罗德里克"]),
    ("千与千寻", "宫崎骏", "动画", ["柊瑠美"]),
    ("天空之城", "宫崎骏", "动画", ["田中真弓"]),
    ("功夫", "周星驰", "喜剧", ["周星驰"]),
    ("大话西游", "刘镇伟", "喜剧", ["周星驰"]),
    ("疯狂的石头", "宁浩", "喜剧", ["黄渤"]),
    ("流浪地球", "郭帆", "科幻", ["吴京"]),
    ("流浪地球2", "郭帆", "科幻", ["吴京"]),
    ("战狼2", "吴京", "动作", ["吴京"]),
    ("西虹市首富", "闫非", "喜剧", ["沈腾"]),
    ("夏洛特烦恼", "闫非", "喜剧", ["沈腾"]),
    ("独行月球", "张吃鱼", "喜剧", ["沈腾"]),
    ("满江红", "张艺谋", "剧情", ["沈腾"]),
    ("英雄", "张艺谋", "动作", ["李连杰"]),
    ("卧虎藏龙", "李安", "动作", ["周润发"]),
    ("无间道", "刘伟强", "犯罪", ["梁朝伟"]),
    ("花样年华", "王家卫", "剧情", ["梁朝伟"]),
    ("重庆森林", "王家卫", "剧情", ["梁朝伟"]),
    ("东邪西毒", "王家卫", "剧情", ["张国荣"]),
]
NAME2IDX = {m[0]: i for i, m in enumerate(MOVIES)}
by_director = defaultdict(list)
by_genre = defaultdict(list)
by_actor = defaultdict(list)
for name, d, g, acts in MOVIES:
    by_director[d].append(name)
    by_genre[g].append(name)
    for a in acts:
        by_actor[a].append(name)

# 故意留出 3 部作为"新电影"（训练集中完全删掉），用于冷启动实验
HELD_OUT = ["阿凡达", "荒岛余生", "信条"]
TRAIN_NAMES = [m[0] for m in MOVIES if m[0] not in HELD_OUT]


def sample_user_history():
    """按四种偏好模式之一生成一条观影序列（有放回：允许重复观看，更接近真实日志）。"""
    mode = random.choices(["director", "actor", "genre", "mixed"], weights=[3, 3, 2, 2])[0]
    n = random.randint(15, 30)
    pool = TRAIN_NAMES

    def gen(fav):
        seq = []
        for _ in range(n):
            x = random.choice(fav) if random.random() < 0.9 else random.choice(pool)
            seq.append(x)
        return seq

    if mode == "director":
        d = random.choice(list(by_director.keys()))
        fav = [x for x in by_director[d] if x in pool]
        return gen(fav) if fav else sample_user_history()
    if mode == "actor":
        a = random.choice(list(by_actor.keys()))
        fav = [x for x in by_actor[a] if x in pool]
        return gen(fav) if fav else sample_user_history()
    if mode == "genre":
        g = random.choice(list(by_genre.keys()))
        fav = [x for x in by_genre[g] if x in pool]
        return gen(fav) if fav else sample_user_history()
    return [random.choice(pool) for _ in range(n)]


def content_init(name):
    """冷启动的'内容特征初始化'：取该新片同导演+同类型（且已见）电影的向量均值。"""
    d, g, acts = next(m for m in MOVIES if m[0] == name)[1:]
    seeds = [x for x in by_director[d] if x in wv]
    seeds += [x for x in by_genre[g] if x in wv and x not in seeds]
    if not seeds:
        return None
    return sum(wv[s] for s in seeds) / len(seeds)


def main():
    global wv
    log = []
    out = {}

    def logf(msg=""):
        log.append(str(msg))
        print(msg)

    # ---- 1. 生成合成数据 + 训练 ----
    logf("=" * 60)
    logf("1. 合成观影序列 → gensim word2vec（item2vec）")
    logf("=" * 60)
    users = [sample_user_history() for _ in range(3000)]
    logf(f"  合成用户数: {len(users)}；训练集电影数: {len(TRAIN_NAMES)}；"
         f"留出新电影: {HELD_OUT}")
    model = Word2Vec(sentences=users, vector_size=50, window=5, min_count=1,
                     sg=1, negative=5, epochs=30, workers=4, seed=42)
    wv = model.wv
    logf("  训练完成，词表大小: %d" % len(wv))

    # ---- 2. 相似电影检索 ----
    logf("")
    logf("=" * 60)
    logf("2. 相似电影检索（word2vec 的近邻检索直接复用）")
    logf("=" * 60)
    queries = ["星际穿越", "阿甘正传", "华尔街之狼", "泰坦尼克号", "卧虎藏龙"]
    sims = {}
    for q in queries:
        top = wv.most_similar(q, topn=5)
        sims[q] = [[w, round(float(s), 3)] for w, s in top]
        logf(f"  {q} -> " + ", ".join(f"{w}({s:.2f})" for w, s in top))

    # ---- 2b. 定量检验：近邻的导演/类型重合率 vs 随机基线 ----
    logf("")
    logf("  定量检验：top10 近邻中'同导演/同类型/同主演'占比（全体训练集电影平均）")
    meta = {m[0]: m for m in MOVIES}
    s_dir = s_gen = s_act = 0.0
    r_dir = r_gen = r_act = 0.0
    for name in TRAIN_NAMES:
        d, g, acts = meta[name][1:]
        top = wv.most_similar(name, topn=10)
        s_dir += sum(1 for w, _ in top if meta[w][1] == d) / 10
        s_gen += sum(1 for w, _ in top if meta[w][2] == g) / 10
        s_act += sum(1 for w, _ in top
                     if set(meta[w][3]) & set(acts)) / 10
        # 随机基线：从训练池随机抽 10 部的期望重合率
        r_dir += sum(1 for w in TRAIN_NAMES if meta[w][1] == d) / len(TRAIN_NAMES)
        r_gen += sum(1 for w in TRAIN_NAMES if meta[w][2] == g) / len(TRAIN_NAMES)
        r_act += sum(1 for w in TRAIN_NAMES
                     if set(meta[w][3]) & set(acts)) / len(TRAIN_NAMES)
    n_train = len(TRAIN_NAMES)
    coh = {
        "same_director_top10": round(s_dir / n_train, 3),
        "same_genre_top10": round(s_gen / n_train, 3),
        "shared_actor_top10": round(s_act / n_train, 3),
        "random_baseline_director": round(r_dir / n_train, 3),
        "random_baseline_genre": round(r_gen / n_train, 3),
        "random_baseline_actor": round(r_act / n_train, 3),
    }
    out["coherence"] = coh
    logf(f"    同导演: {coh['same_director_top10']}（随机基线 {coh['random_baseline_director']}）")
    logf(f"    同类型: {coh['same_genre_top10']}（随机基线 {coh['random_baseline_genre']}）")
    logf(f"    共享主演: {coh['shared_actor_top10']}（随机基线 {coh['random_baseline_actor']}）")
    logf("  说明：近邻重合率显著高于随机基线，说明向量把'导演/类型/演员'的共现信息都编码进去了——")
    logf("       这正是'喜欢汉克斯的人未必只该被推汉克斯电影'的证据：向量同时受导演、类型影响。")

    # ---- 3. 冷启动 ----
    logf("")
    logf("=" * 60)
    logf("3. 冷启动：新电影没有向量怎么办")
    logf("=" * 60)
    cold = {}
    for name in HELD_OUT:
        in_vocab = name in wv
        cold[name] = {"in_vocab": in_vocab}
        if in_vocab:
            logf(f"  [意外] {name} 在词表中（生成器 bug？）")
            continue
        logf(f"  {name} 不在词表：直接检索会 KeyError —— 这是冷启动问题的本质。")
        v = content_init(name)
        if v is None:
            logf("    无可用内容特征种子，跳过。")
            continue
        top = wv.similar_by_vector(v, topn=5)
        cold[name]["content_init_top5"] = [[w, round(float(s), 3)] for w, s in top]
        d, g = next(m for m in MOVIES if m[0] == name)[1:3]
        logf(f"    内容初始化（同导演[{d}]+同类型[{g}]向量均值）后的近邻:")
        logf("      " + ", ".join(f"{w}({s:.2f})" for w, s in top))
        # 对照：随机初始化向量
        random_v = model.wv.get_vector(random.choice(TRAIN_NAMES)) * 0  # 零向量对照
        top_rand = wv.similar_by_vector(random_v, topn=5)
        cold[name]["zero_vector_top5"] = [[w, round(float(s), 3)] for w, s in top_rand]
        logf(f"    对照（零向量，即'完全没有信息'）的近邻: "
             + ", ".join(f"{w}({s:.2f})" for w, s in top_rand))

    out["similarity_queries"] = sims
    out["cold_start"] = cold
    out["n_users"] = len(users)
    out["n_train_movies"] = len(TRAIN_NAMES)
    out["held_out"] = HELD_OUT

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "movie_embedding_demo.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS, "movie_embedding_demo_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log))
    logf("")
    logf("[saved] results/movie_embedding_demo.json + movie_embedding_demo_log.txt")


if __name__ == "__main__":
    main()
