# -*- coding: utf-8 -*-
"""kenlm 模型加载小工具：优先 .bin，其次 .arpa。"""
import os


def model_path(models_dir, order):
    for ext in (".bin", ".arpa"):
        p = os.path.join(models_dir, f"order{order}{ext}")
        if os.path.exists(p):
            return p
    return None


def load(kenlm_mod, models_dir, order):
    p = model_path(models_dir, order)
    if p is None:
        return None
    return kenlm_mod.Model(p)


def available_orders(models_dir):
    return sorted({int(os.path.basename(f).replace("order", "").split(".")[0])
                   for f in os.listdir(models_dir)
                   if f.startswith("order") and f.endswith((".bin", ".arpa"))})
