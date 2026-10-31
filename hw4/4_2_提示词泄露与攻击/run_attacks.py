# -*- coding: utf-8 -*-
"""hw4 提示词泄露/攻击调研：实测脚本。

对 attacks.json 中每条用例，以固定系统提示词（青禾翻译助手）调用 DeepSeek API，
原始响应全部存档到 results/raw_attacks.jsonl。
"""
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import system_prompt

HERE = os.path.dirname(os.path.abspath(__file__))
API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
MAX_RETRIES = 2


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


def call(text, api_key, retries_left=MAX_RETRIES):
    try:
        resp = requests.post(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt.SYS_PROMPT},
                    {"role": "user", "content": text},
                ],
                "temperature": 0.7,
                "max_tokens": 600,
                "stream": False,
            },
            timeout=120,
        )
        if resp.status_code != 200:
            if retries_left > 0:
                time.sleep(2)
                return call(text, api_key, retries_left - 1)
            return {"error": f"http_{resp.status_code}"}
        data = resp.json()
        return {
            "model": data.get("model", MODEL),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "content": data["choices"][0]["message"]["content"],
            "usage": data.get("usage", {}),
        }
    except requests.exceptions.RequestException as e:
        if retries_left > 0:
            time.sleep(2)
            return call(text, api_key, retries_left - 1)
        return {"error": str(e)}


def main():
    with open(os.path.join(HERE, "attacks.json"), encoding="utf-8") as f:
        cases = json.load(f)
    api_key = load_api_key()
    out_path = os.path.join(HERE, "results", "raw_attacks.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                done.add(json.loads(line)["id"])
    with open(out_path, "a", encoding="utf-8") as fout:
        for case in cases:
            if case["id"] in done:
                print("skip", case["id"])
                continue
            resp = call(case["text"], api_key)
            entry = {"id": case["id"], "kind": case["kind"], "desc": case["desc"],
                     "text": case["text"], "response": resp}
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
            fout.flush()
            content = (resp.get("content") or str(resp.get("error"))).replace("\n", " ")
            print(f"[{case['id']}] {content[:90]}", flush=True)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
