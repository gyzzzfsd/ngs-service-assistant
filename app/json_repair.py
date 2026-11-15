"""
json_repair.py —— 模型输出的 JSON 容错解析

职责：
    把模型返回的一段文本，尽量稳妥地变成 dict。要处理三种常见畸形：
      1. 被 ```json 代码块包着
      2. 前面有"好的，结果如下："这种废话
      3. 字符串正文里的引号没转义（最高频，见 escape_inner_quotes 的说明）

不负责：
    字段是否符合 AgentOutput 契约 —— 那是 Pydantic 的活，上层 agent.py 负责。

为什么单独一个文件：
    这套逻辑已经不是一个"小工具函数"了：要分候选、要两轮尝试、要状态机修引号，
    加起来一百多行，而且有自己独立的测试（tests/test_json_repair.py）。
    塞回 agent.py 会让职责边界变模糊，
    也会让"编排流程"和"文本修复"两件事混在一起看。
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> dict[str, Any] | None:
    """
    从模型输出里抠出 JSON 对象。

    Args:
        text: 模型原始输出，可能是纯 JSON，也可能是代码块或带前缀的废话。

    Returns:
        解析出的 dict；实在解析不出来返回 None（上层据此走兜底）。

    为什么不能直接 json.loads：
        即使开了 json_mode，仍有小概率出现 ```json 包裹或"好的，如下："这种前缀。
        多一层容错比让整个请求失败划算。

    # 模型输出的 JSON 里，正文中的引号经常不转义。
    # 它把 content 写成 "订单1目前是"待发货"状态，可以帮您取消。" ——
    # 这段在 JSON 语法上直接非法（两个引号把字符串提前闭合了），
    # json.loads 必然失败。而模型自己完全不觉得有问题，重试一次还是同样的写法。
    # 这不是"模型偶尔抽风"，是高频现象：只要回答里带引号就会复现。
    # 所以除了"多试几种包裹形式"，还必须专门修一次"字符串内部的裸引号"。
    """
    text = text.strip()
    candidates = _json_candidates(text)

    # 第一轮：原样解析
    for candidate in candidates:
        obj = _loads_object(candidate)
        if obj is not None:
            return obj

    # 第二轮：修掉正文里的裸引号再解析
    for candidate in candidates:
        repaired = escape_inner_quotes(candidate)
        if repaired == candidate:
            continue
        obj = _loads_object(repaired)
        if obj is not None:
            logger.warning("模型输出的 JSON 含有未转义引号，已修复后解析 | 原文片段=%s", text[:120])
            return obj

    return None


def _json_candidates(text: str) -> list[str]:
    """
    列出模型输出里"可能是 JSON 的那几段"，按可信度从高到低。

    三种形态分别是：本身就是 JSON、被 ```json 包裹、前后有废话。
    去重是因为三者经常完全一样，没必要重复解析三次。
    """
    candidates: list[str] = []

    def _add(value: str) -> None:
        value = value.strip()
        if value and value not in candidates:
            candidates.append(value)

    _add(text)

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        _add(fenced.group(1))

    # 多余的花括号也会命中：JSON 字符串里出现 } 时这里会切歪，
    # 所以它排在最后 —— 前面的形态都失败才轮到它。
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        _add(text[start : end + 1])

    return candidates


def _loads_object(candidate: str) -> dict[str, Any] | None:
    """解析成 dict；不是 JSON 或不是对象（比如数组）都返回 None。"""
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def escape_inner_quotes(raw: str) -> str:
    """
    把"字符串值内部没转义的双引号"补上反斜杠。

    Args:
        raw: 一段疑似 JSON 的文本。

    Returns:
        修复后的文本（可能和输入一样，表示没找到需要修的地方）。

    判断规则（只靠上下文，不需要懂业务字段）：
        遇到字符串内部的 `"` 时，看它后面第一个非空白字符。
        如果是 , : } ] 之一，说明这个引号是"值的结尾"，本来就该闭合；
        否则它只可能是正文里的引号（比如中文引号写成了英文引号），补一个反斜杠。

    # 为什么不用正则一次性替换？
    # 因为"哪些引号是结构性的、哪些是正文里的"完全取决于位置，
    # 正则没有状态，分不清第 3 个引号是开还是闭。
    # 必须逐字符扫描并记录"当前是否在字符串里"。

    副作用说明：
        如果模型输出的是"格式正确但语义奇怪"的 JSON，这里不会改动它（原样返回），
        所以修复只在解析失败后作为补救手段，不会污染正常路径。
    """
    out: list[str] = []
    in_string = False
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]

        # ---------- 字符串外部：只关心"是不是要进入字符串" ----------
        if not in_string:
            if ch == '"':
                in_string = True
            out.append(ch)
            i += 1
            continue

        # ---------- 字符串内部 ----------
        # 反斜杠开头的转义序列整体跳过，避免把 \" 误判成裸引号
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(raw[i + 1])
                i += 2
            else:
                i += 1
            continue

        if ch == '"':
            j = i + 1
            while j < n and raw[j] in " \t\r\n":
                j += 1
            if j >= n or raw[j] in ",:}]":
                in_string = False       # 正常闭合
                out.append(ch)
            else:
                out.append('\\"')       # 正文里的裸引号
            i += 1
            continue

        out.append(ch)
        i += 1

    return "".join(out)
