# -*- coding: utf-8 -*-
"""P123 断点续跑：recipe_exp.py 的续跑入口。

用法：python scripts/recipe_resume.py
- 读 results/recipe_results.json（若存在），跳过已完成的配方（按 name 匹配）；
- 补跑剩余配方，每个配方完成后立即把合并结果写回 results/recipe_results.json。
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from recipe_exp import RECIPES, BUDGET, OUT, load_corpus, run_recipe  # noqa: E402


def main():
    torch.manual_seed(0)
    out_path = os.path.join(OUT, "recipe_results.json")
    done = []
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            prev = json.load(f)
        done = prev["results"]
    done_names = {r["name"] for r in done}
    print(f"resume: already done {len(done)}/{len(RECIPES)}: "
          + ", ".join(sorted(done_names)) if done_names else "resume: nothing done")

    train_text, valid_text, c2i = load_corpus()
    V = len(c2i)
    print(f"corpus: train={len(train_text)} chars, valid={len(valid_text)}, vocab={V}")
    results = list(done)
    for r in RECIPES:
        if r["name"] in done_names:
            print(f"[skip] {r['name']}")
            continue
        rec = run_recipe(r["name"], train_text, valid_text, c2i,
                         r["B"], r["steps"], r["lr"], r["warmup"], r["schedule"],
                         r["dropout"], r["label_smoothing"])
        results.append(rec)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"vocab": V, "budget_tokens": BUDGET, "results": results},
                      f, ensure_ascii=False, indent=1)
    print("saved ->", out_path)


if __name__ == "__main__":
    main()
