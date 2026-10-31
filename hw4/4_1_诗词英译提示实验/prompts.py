# -*- coding: utf-8 -*-
"""hw4 唐诗宋词英译提示实验：三个提示版本与公共系统消息。

版本设计对应 L3 part1 第 89 张：
  A 直接翻译（简单提示）
  B 加入任务标准（术语与关键关系保持要求）
  C 只加入边界示例（诊断性输入输出对，不陈述规则）
所有版本共用同一系统消息、同一输出格式要求、同一模型与采样配置。
"""

SYSTEM = (
    "你是诗词翻译模型。你的任务是把用户给出的中文诗词翻译成英文。"
    "你只能输出一个 JSON 对象，键为 translation（英文译文字符串，可在字符串内用 \\n 分行）"
    "与 notes（字符串数组，0 到 2 条，为英文读者理解所必需的背景注释，每条不超过 30 个英文单词）。"
    "译文不超过 120 个英文单词，不含标题，只包含诗词译文本身。"
    "不得输出 JSON 以外的任何内容。"
)

# 版本 B 的任务标准（在 A 的指令后追加）
CRITERIA = (
    "翻译标准：\n"
    "1. 时间与时序：保留原文的时间先后与时态变化（如\u201c未\u201d\u201c莫\u201d\u201c何时\u201d等否定与疑问）。\n"
    "2. 否定与例外：原文的否定、转折、例外（如\u201c不\u201d\u201c无\u201d\u201c却\u201d\u201c竟\u201d）必须如实表达，不得改成肯定。\n"
    "3. 术语与文化词：专名用拼音音译（如 龙城 Longcheng、巴山 Bashan）；节日、器物、典故保留原文意象，"
    "可在 notes 中简要解释，不得替换成西方对应物。\n"
    "4. 关系：保持原文的主客体、条件、因果与问答关系，不改变人称与动作发出者。\n"
    "5. 格式：译文每行对应原诗一行，行数一致；只翻译原文，不添加原文没有的事实；不省略任何一句。"
)

# 版本 C 的边界示例（诊断性输入输出对；示例诗不在实验语料中）
EXAMPLE_1_IN = "春蚕到死丝方尽，蜡炬成灰泪始干。"
EXAMPLE_1_OUT = (
    '{"translation": "Silkworms spin silk only till death;\\n'
    'Candles shed tears only when burnt to ash.", "notes": []}'
)
EXAMPLE_2_IN = "遥知兄弟登高处，遍插茱萸少一人。"
EXAMPLE_2_OUT = (
    '{"translation": "Far off I know my brothers climb the height,\\n'
    'All wearing zhuyu, one face missing from the sight.", '
    '"notes": ["zhuyu: cornel sprigs worn on the Double Ninth Festival."]}'
)

TAIL = "\n\n按系统要求以 JSON 输出。"


def poem_text(poem: dict) -> str:
    return "\n".join(poem["lines"])


def build_user_prompt(version: str, poem: dict) -> str:
    """version: A / B / C"""
    text = poem_text(poem)
    if version == "A":
        return "将下面这首中文诗词翻译成英文。\n\n" + text + TAIL
    if version == "B":
        return (
            "将下面这首中文诗词翻译成英文。\n\n"
            + CRITERIA
            + "\n\n"
            + text
            + TAIL
        )
    if version == "C":
        return (
            "将下面这首中文诗词翻译成英文。下面是两个翻译示例，请参照同样的处理方式完成翻译任务。\n\n"
            "示例 1\n原文："
            + EXAMPLE_1_IN
            + "\n译文（JSON）：\n"
            + EXAMPLE_1_OUT
            + "\n\n示例 2\n原文："
            + EXAMPLE_2_IN
            + "\n译文（JSON）：\n"
            + EXAMPLE_2_OUT
            + "\n\n现在翻译：\n"
            + text
            + TAIL
        )
    raise ValueError(f"unknown version: {version}")
