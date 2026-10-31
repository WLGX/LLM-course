# -*- coding: utf-8 -*-
"""hw4 诗词英译实验：盲评脚本（LLM 评审）。

按 protocol_评分协议.md：
- 每份译文一次评审调用，输入不含版本标识（盲评）。
- 逐单元判定 完整/部分/缺失/错误 + evidence + explanation，并给整体可读性 overall(1-5)。
- 评审温度 0；评审响应原文存档；格式检查在 analyze.py 中程序化完成。
"""
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "results", "raw")
SCORE_DIR = os.path.join(HERE, "results", "scores")

API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_RETRIES = 3

JUDGE_SYSTEM = (
    "你是严格的翻译质量评审员。你收到的每条材料包含：中文原诗（source_poem_zh）、"
    "关键语义单元列表（critical_units，每项含 unit 文本、category 类别与 note 期望说明）、"
    "一份匿名英文译文（anonymous_translation，含 translation 与 notes 两个字段）。\n"
    "你的任务：\n"
    "1. 对每个关键单元判定传达状态：status 取 '完整'（语义准确传达）、'部分'（只传达一部分）、"
    "'缺失'（未传达）或 '错误'（明显曲解）。\n"
    "2. evidence：从译文中逐字引用的证据（必须与译文原文逐字一致；找不到可留空字符串）。\n"
    "3. explanation：一句中文说明判断依据。\n"
    "4. overall：译文整体可读性与诗意，1 到 5 的整数（5 最好）。\n"
    "5. 不得推测译文来自哪个提示版本或模型；只依据给定材料。\n"
    '只输出 JSON：{"units": [{"unit": "...", "status": "...", "evidence": "...", "explanation": "..."}], "overall": 整数}'
)


def load_api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key
    # 依次检查 F:/LLM-course/.env 与 F:/LLM-course/hw4/.env
    env_files = [
        os.path.join(os.path.dirname(os.path.dirname(HERE)), ".env"),
        os.path.join(os.path.dirname(HERE), ".env"),
    ]
    for env_file in env_files:
        if os.path.exists(env_file):
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEEPSEEK_API_KEY="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("未找到 DEEPSEEK_API_KEY")


def call_judge(source_poem, units, translation, api_key, retries_left=MAX_RETRIES):
    user = json.dumps(
        {
            "source_poem_zh": source_poem,
            "critical_units": units,
            "anonymous_translation": translation,
        },
        ensure_ascii=False,
    )
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": 2000,
                "stream": False,
            },
            timeout=180,
        )
        if resp.status_code != 200:
            if retries_left > 0:
                time.sleep(2)
                return call_judge(source_poem, units, translation, api_key, retries_left - 1)
            return {"error": f"http_{resp.status_code}"}, None
        content = resp.json()["choices"][0]["message"]["content"].strip()
        c = content
        if c.startswith("```"):
            c = c.split("\n", 1)[-1]
            if c.rstrip().endswith("```"):
                c = c.rstrip()[:-3]
        try:
            return None, json.loads(c)
        except json.JSONDecodeError:
            if retries_left > 0:
                time.sleep(2)
                return call_judge(source_poem, units, translation, api_key, retries_left - 1)
            return {"error": "invalid_json"}, None
    except requests.exceptions.RequestException as e:
        if retries_left > 0:
            time.sleep(2)
            return call_judge(source_poem, units, translation, api_key, retries_left - 1)
        return {"error": str(e)}, None


def main():
    split = os.environ.get("HW4_SPLIT", "test")
    os.makedirs(SCORE_DIR, exist_ok=True)
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    poem_map = {p["id"]: p for p in corpus["poems"] if p["split"] == split}
    api_key = load_api_key()

    raw_path = os.path.join(RAW_DIR, f"raw_{split}.jsonl")
    out_path = os.path.join(SCORE_DIR, f"scores_{split}.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                done.add((e["poem_id"], e["version"], e["rep"]))

    n = 0
    with open(raw_path, encoding="utf-8") as fin, open(out_path, "a", encoding="utf-8") as fout:
        for line in fin:
            entry = json.loads(line)
            key = (entry["poem_id"], entry["version"], entry["rep"])
            if key in done:
                continue
            if not entry.get("parsed"):
                rec = {"poem_id": entry["poem_id"], "version": entry["version"],
                       "rep": entry["rep"], "score": None, "error": "generation_failed"}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            poem = poem_map[entry["poem_id"]]
            source_poem = "\n".join(poem["lines"])
            units = [{"unit": u["text"], "category": u["category"], "note": u["note"]}
                     for u in poem["units"]]
            err, score = call_judge(source_poem, units, entry["parsed"], api_key)
            rec = {
                "poem_id": entry["poem_id"],
                "version": entry["version"],
                "rep": entry["rep"],
                "score": score,
                "error": err,
                "usage_note": "see raw generation entries for usage",
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
            print(f"[{n:02d}] {entry['poem_id']} {entry['version']}-{entry['rep']}: "
                  f"{'ok' if score else ('err ' + str(err))}", flush=True)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
