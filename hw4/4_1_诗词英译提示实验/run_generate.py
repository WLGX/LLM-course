# -*- coding: utf-8 -*-
"""hw4 诗词英译实验：生成脚本。

对每（诗 × 版本 × 重复）调用 DeepSeek API，原始请求/响应全部存档。
重试仅限 HTTP 错误、超时或返回内容无法解析为 JSON，且计入成本。
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prompts

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "results", "raw")

API_URL = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
TEMPERATURE = 0.7
MAX_TOKENS = 400
MAX_RETRIES = 3
REPEATS = 2
VERSIONS = ["A", "B", "C"]
SPLIT = os.environ.get("HW4_SPLIT", "test")  # dev / test


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
    raise SystemExit("未找到 DEEPSEEK_API_KEY：请设置环境变量或写入 F:/LLM-course/.env")


def call_api(messages, api_key, retries_left=MAX_RETRIES):
    """返回 (payload_dict, raw_text)。payload_dict 为解析后的 JSON 或 None。"""
    start = time.time()
    try:
        resp = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        if resp.status_code != 200:
            err = {
                "error": "http_error",
                "status": resp.status_code,
                "body": resp.text[:500],
                "retry_left": retries_left,
            }
            if retries_left > 0:
                time.sleep(2)
                return call_api(messages, api_key, retries_left - 1)
            return err, None
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        record = {
            "model": data.get("model", MODEL),
            "finish_reason": data["choices"][0].get("finish_reason"),
            "usage": data.get("usage", {}),
            "elapsed_s": round(elapsed, 2),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        # 解析 JSON（容忍代码围栏）
        parsed = None
        c = content.strip()
        if c.startswith("```"):
            c = c.split("\n", 1)[-1]
            if c.rstrip().endswith("```"):
                c = c.rstrip()[:-3]
        try:
            parsed = json.loads(c)
        except json.JSONDecodeError:
            if retries_left > 0:
                time.sleep(2)
                return call_api(messages, api_key, retries_left - 1)
            record["error"] = "invalid_json"
        record["content"] = content
        return record, parsed
    except requests.exceptions.RequestException as e:
        if retries_left > 0:
            time.sleep(2)
            return call_api(messages, api_key, retries_left - 1)
        return {"error": "request_exception", "detail": str(e)}, None


def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    poems = [p for p in corpus["poems"] if p["split"] == SPLIT]
    api_key = load_api_key()

    out_path = os.path.join(RAW_DIR, f"raw_{SPLIT}.jsonl")
    n = 0
    with open(out_path, "a", encoding="utf-8") as fout:
        for poem in poems:
            for ver in VERSIONS:
                for rep in range(1, REPEATS + 1):
                    user = prompts.build_user_prompt(ver, poem)
                    record, parsed = call_api(
                        [
                            {"role": "system", "content": prompts.SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        api_key,
                    )
                    entry = {
                        "poem_id": poem["id"],
                        "title": poem["title"],
                        "version": ver,
                        "rep": rep,
                        "user_prompt": user,
                        "response": record,
                        "parsed": parsed,
                    }
                    fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fout.flush()
                    n += 1
                    status = "ok" if parsed else ("err:" + str(record.get("error")))
                    print(f"[{n:02d}] {poem['id']} {ver}-{rep}: {status}", flush=True)
    print("done ->", out_path)


if __name__ == "__main__":
    main()
