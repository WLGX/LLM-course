# -*- coding: utf-8 -*-
"""
合并 KenLM 训练结果 JSON：
- results/kenlm_train.json（刚跑完的 1、9、10 阶）
- results/kenlm_train_old.json（首轮 2-8 阶的时间/体积）
- 5-8 阶完整计数来自 order7/order8 ARPA 头部（grep '^ngram'）：
    5: 4895726, 6: 4841416, 7: 4689615, 8: 4505167
- order7/8 build_binary 是重编译 max-order=10 后手动完成的（40.4s / 70.8s）
输出：results/kenlm_train.json（1-10 阶完整、按阶排序、无 error 条目）
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
MODELS = r"F:/ngram_work/models_kenlm"

BASE = {
    1: 123434, 2: 1546501, 3: 3646846, 4: 4664812,
    5: 4895726, 6: 4841416, 7: 4689615, 8: 4505167,
}


def counts_upto(order):
    return {k: v for k, v in BASE.items() if k <= order}


# 旧 JSON：拿到 2-8 阶的时间与体积字段
with open(os.path.join(RESULTS, "kenlm_train_old.json"), encoding="utf-8") as f:
    old = {r["order"]: r for r in json.load(f)}

# 新 JSON：1、9、10 阶
with open(os.path.join(RESULTS, "kenlm_train.json"), encoding="utf-8") as f:
    new = {r["order"]: r for r in json.load(f)}

entries = {}

for o in (2, 3, 4):
    r = old[o]
    entries[o] = {**r, "counts": counts_upto(o)}

# 5、6：时间体积用旧值，计数补全
for o, extra in ((5, 4895726), (6, 4841416)):
    r = old[o]
    c = counts_upto(o)
    c[o] = extra
    entries[o] = {**r, "counts": c}

# 7、8：lmplz 时间用旧值；build_binary 为手动重跑
entries[7] = {
    "order": 7,
    "lmplz_seconds": old[7]["lmplz_seconds"],
    "build_binary_seconds": 40.42,
    "arpa_mib": old[7]["arpa_mib"],
    "binary_mib": round(os.path.getsize(os.path.join(MODELS, "order7.bin")) / 2**20, 1),
    "counts": counts_upto(7),
    "log": old[7]["log"],
}
entries[8] = {
    "order": 8,
    "lmplz_seconds": old[8]["lmplz_seconds"],
    "build_binary_seconds": 70.79,
    "arpa_mib": old[8]["arpa_mib"],
    "binary_mib": round(os.path.getsize(os.path.join(MODELS, "order8.bin")) / 2**20, 1),
    "counts": counts_upto(8),
    "log": old[8]["log"],
}

# 1、9 用新/旧 JSON；10 阶来自第二次运行的任务输出（JSON 被 9 阶重跑覆盖）
if 1 in old:
    entries[1] = {**old[1], "counts": counts_upto(1),
                  "binary_mib": old[1]["arpa_mib"]}   # 1 阶保留 ARPA（build_binary 不支持 unigram）
if 9 in new and not new[9].get("error"):
    entries[9] = {**new[9],
                  "counts": {int(k): v for k, v in new[9]["counts"].items()}}
entries[10] = {
    "order": 10,
    "lmplz_seconds": 427.9,
    "build_binary_seconds": 89.6,
    "arpa_mib": 2132.0,
    "binary_mib": round(os.path.getsize(os.path.join(MODELS, "order10.bin")) / 2**20, 1),
    "counts": {**counts_upto(10), 9: 4306785, 10: 4104906},
    "log": "/root/lmplz_tmp/order10.log",
}

final = [entries[o] for o in sorted(entries)]
with open(os.path.join(RESULTS, "kenlm_train.json"), "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=2)

print("orders merged:", sorted(entries))
for r in final:
    print(r["order"], "lmplz", r.get("lmplz_seconds"), "s | bin",
          r.get("binary_mib"), "MiB | counts", r.get("counts"))
