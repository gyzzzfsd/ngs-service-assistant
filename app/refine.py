"""
refine.py —— 收口：把模型草稿加工成"可信"的 AgentOutput

职责（V5 纠错与兜底的完整流程）：
    0. 转人工意图：wants_human() 判断"用户是不是点名要人工"（agent.py 据此走快速通道）
    1. 结构化：让模型把自然语言草稿转成 AgentOutput JSON（V1 的入口）
    2. 校验：交给 harness.validate 判五关卡
    3. 补查：harness 说"证据不足"时，强制补一次 get_user_orders 再复验
    4. 纠错：校验不过 → 把具体错误喂回模型重写一次（只重写一次）
    5. 兜底：重写仍不过 → 固定话术 + REQUEST_HANDOFF

不负责：
    工具循环和上下文组装（agent.py）、判断校验规则本身（harness.py）、
    话术文案（prompts.py）。

为什么单独一个文件：
    这条链路有自己的完整状态机（最多 1 次补查 + 1 次重写 + 1 次兜底），
    抽出来后，agent.py 只管“跑”，本文件只管“收口”。

状态机（最多消耗 2 次模型调用）：
    结构化(1) → [校验失败且证据不足 → 补查(0 次模型) → 复验]
              → [仍失败 → 重写(1) → 复验] → [仍失败 → 兜底(0)]
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from app import harness
from app.json_repair import extract_json
from app.llm import LLMError, chat
from app.prompts import FALLBACK_CONTENT, STRUCTURE_PROMPT
from app.schemas import AgentOutput, ReplyType, ToolCallRecord
from app.tools import execute_tool

logger = logging.getLogger(__name__)


# 确定性意图识别（转人工 / 全量盘点）

# 用户"点名要真人"的说法。只收不会被日常提问用到的短语，
# 刻意不收"客服"两个字 —— "你们客服"这种提法太常见，收了必然误拦。
# 生信域补充：坐席是"技术支持/生信工程师"，用户会说"找工程师""人工支持"。
HANDOFF_CUES = (
    "转人工",
    "转接人工",
    "接人工",
    "人工客服",
    "人工服务",
    "人工支持",
    "找人工",
    "要人工",
    "换人工",
    "真人客服",
    "真人",
    "找工程师",
    "转工程师",
    "找技术支持",
)

# 否定线索：紧贴在短语前面时，这句话的真实意思是"别转"。
# 例："先不用转人工""别给我转人工""我不需要人工客服"。
# 中间允许夹"给我/帮我/再"这类虚词 —— 用户就是这么说话的：
# 只认"别转人工"会把"别给我转人工"漏成肯定，白白弹一个转接按钮出去。
_HANDOFF_NEGATION_RE = re.compile(
    r"(?:不用|不要|不需要|不必|先不|不|别)(?:给我|帮我|再|就|马上|立刻|现在|用|需要)*$"
)

# 例外一：问人工/工程师的作息/联系方式，是在打听信息，不是要转接。
_ASKING_ABOUT_HUMAN_RE = re.compile(
    r"(?:人工客服|人工服务|人工支持|真人客服|转人工|找人工|找工程师|工程师|技术支持)"
    r"(?:要|会|需要|得)?(?:几点|电话|上班|下班|时间|多久|收费|花钱|在吗|联系方式)"
)


def wants_human(text: str) -> bool:
    """
    判断用户这句话是不是"明确要求转人工"。

    Args:
        text: 用户本轮输入的原始文本。

    Returns:
        True = 用户点名要人工；False = 不是（含否定说法和为打听信息而提到人工客服）。

    # 为什么这个判断不交给模型？
    # 实测同一句"转人工"，模型这次答"好的为您转接"（REQUEST_HANDOFF），
    # 下次答"抱歉我无法为您转接人工客服"（DECLINE）—— 采样有随机性，
    # 而这种"能不能找到人"的能力不该掷骰子决定。
    # 用户点名要人工是一条明确的指令，用词表判比用模型判更稳、更快（省两次模型调用），
    # 也让"转人工按钮"稳定出现。
    """
    if not text:
        return False
    if _ASKING_ABOUT_HUMAN_RE.search(text):
        return False

    for cue in HANDOFF_CUES:
        start = 0
        while (idx := text.find(cue, start)) != -1:
            start = idx + len(cue)

            # 例外二："需要人工"里的"要人工"不是独立短语，不能当请求。
            # 没有这一条，"我不需要人工客服"会被判成要转人工。
            if cue == "要人工" and idx > 0 and text[idx - 1] == "需":
                continue

            prefix = text[max(0, idx - 6) : idx]
            # 用"前缀以否定词结尾"判断，而不是"前缀里出现过否定词"：
            # 后者会把"没有问题，转人工吧"误判成否定。
            if not _HANDOFF_NEGATION_RE.search(prefix):
                return True
    return False


# 全量盘点类问法：问"一共几个 / 多少张 / 全部清单"时，答案必须覆盖本轮
# 工具查回的每一张委托单、每一条样本。这类问法最容易被历史答案带偏。
INVENTORY_CUES = (
    "一共几个",
    "一共有几个",
    "总共有几个",
    "总共几个",
    "一共有多少",
    "一共多少",
    "总共有多少",
    "总共多少",
    "几个样本",
    "多少个样本",
    "几条样本",
    "几张委托单",
    "多少张委托单",
    "几张单",
    "全部样本",
    "所有样本",
    "全部委托单",
    "所有委托单",
    "全部订单",
    "所有订单",
)


def wants_inventory(text: str) -> bool:
    """
    判断用户这句话是不是在问"全量盘点"（数量或完整清单）。

    Args:
        text: 用户本轮输入的原始文本。

    Returns:
        True = 必须先把本轮真实数据摆到模型面前；False = 普通问法，照常走工具循环。

    # 为什么这类问题要单独判，而不是靠 system prompt 让模型每轮都查？
    # 全量盘点问法下，模型可能不调工具，直接复述历史里
    # 那句"6 个样本"；prompt 第三节"历史一律视为过期"挡不住这种惯性。
    # 更麻烦的是 harness 只校验"说出来的事实有没有证据"，refine 补查后
    # 旧答案照样拿到背书、判 passed —— 漏掉新样本属于漏报，规则拦不住，
    # 而且补查只补证据、不改写已经生成的正文（见 _validate_with_refill）。
    # 唯一可靠的拦截点是在调模型之前，用确定性方式把本轮数据塞进上下文。
    #
    # 词表刻意不含"我有哪些委托单"这类泛问法：那些问法模型本来就会主动
    # 调工具（test_agent.py 的主流程用例就是这么写的），收进来只会让每次
    # 问答多一次无谓 SQL，还会把"模型自主调工具"这条路径的测试价值冲掉。
    """
    if not text:
        return False
    return any(cue in text for cue in INVENTORY_CUES)



class AgentOutputError(RuntimeError):
    """模型给出的结构化输出无法解析成 AgentOutput。"""


@dataclass
class RefineResult:
    """一次收口的结果，由 agent.py 装进 AgentRun。"""

    output: AgentOutput                     # 最终给用户的输出（可能被降级替换）
    validation: harness.HarnessResult | None  # 校验结论；None = 没走到校验（结构化就失败了）
    corrected: bool = False                 # True = 重写后通过的
    error: str | None = None                # 非空 = 走了兜底，格式 "错误码: 原因"


def refine_draft(
    messages: list[dict[str, Any]],
    draft: str,
    trace: list[ToolCallRecord],
    user_id: int,
) -> RefineResult:
    """
    收口主流程：结构化 → 校验 → 补查 → 重写 → 兜底。

    Args:
        messages: 本轮完整对话（会被就地追加补查的 tool 消息，供重写时复用证据）。
        draft: 模型的自然语言草稿。
        trace: 工具轨迹（补查的记录也会追加进来，供复验取证据）。
        user_id: 当前登录用户。

    Returns:
        RefineResult。任何失败都不抛异常，统一收敛成兜底输出。

    # 补查和重写为什么都只允许一次？
    # 每次重试都要多一次模型调用（约 1~2 秒）和多一份出错的可能。
    # 补查解决的是"没证据"，一次 get_user_orders 就够；
    # 重写解决的是"模型嘴瓢"，把错误说清楚后一次改对的成功率已经很高。
    # 再往上堆次数，用户等的时长线性增加、收益却指数下降 —— 不如转人工。
    """
    # ---------- 第一轮：结构化 + 校验 ----------
    output, structure_error = _try_structure(messages, draft)

    if structure_error is None:
        result = _validate_with_refill(output, messages, trace, user_id)
        if result.passed:
            return RefineResult(output=result.output, validation=result)
        feedback_error = result.error_message
    else:
        feedback_error = structure_error

    # ---------- 第二轮：带着具体错误重写一次（V5）----------
    logger.warning("harness 拦截，反馈错误后重写一次 | user=%s 错误=%s", user_id, feedback_error)
    try:
        output, structure_error = _try_structure(messages, draft, feedback=feedback_error)
    except LLMError as exc:
        # 第一轮模型还活着、重写这轮才挂：属于模型故障，不是输出问题，
        # 错误码必须是 LLM_ERROR，agent_runs 统计里才不会把锅扣给模型输出。
        return _fallback(None, f"LLM_ERROR: {exc}")

    if structure_error is None:
        result = _validate_with_refill(output, messages, trace, user_id)
        if result.passed:
            return RefineResult(output=result.output, validation=result, corrected=True)
        return _fallback(result, f"{result.error_codes[0]}: {result.error_message}")

    return _fallback(None, f"MODEL_OUTPUT_INVALID: {structure_error}")


def _try_structure(
    messages: list[dict[str, Any]],
    draft: str,
    feedback: str | None = None,
) -> tuple[AgentOutput | None, str | None]:
    """
    调一次模型做结构化（V1 的入口），并解析。

    Args:
        messages: 完整对话。
        draft: 自然语言草稿。
        feedback: 非空 = 上一轮的校验错误，拼进请求让模型改。

    Returns:
        (AgentOutput, None) 成功；或 (None, 错误描述) 失败。
        输出问题（JSON 坏、字段错）以字符串返回，由调用方决定重写还是兜底；
        唯一会抛的异常是 LLMError（模型本身挂了，重写没有意义），
        交给 agent.run 统一转成兜底。
    """
    ask = STRUCTURE_PROMPT
    if feedback:
        ask += "\n\n" + harness.correction_feedback(feedback)

    payload = [
        *messages,
        {"role": "assistant", "content": draft},
        {"role": "user", "content": ask},
    ]

    reply = chat(payload, tools=None, json_mode=True)
    raw = reply.content or ""
    # 原文只在 DEBUG 打：它包含完整的用户对话上下文，量很大。
    # 真出问题时把 LOG_LEVEL 调成 DEBUG 再复现一次就能看到。
    logger.debug("结构化输出原文（前 500 字）：%s", raw[:500])

    data = extract_json(raw)
    if data is None:
        return None, f"输出不是合法 JSON：{raw[:200]}"

    try:
        return AgentOutput.model_validate(data), None
    except Exception as exc:  # pydantic.ValidationError
        return None, f"字段不符合 AgentOutput 契约：{exc}"


def _validate_with_refill(
    output: AgentOutput,
    messages: list[dict[str, Any]],
    trace: list[ToolCallRecord],
    user_id: int,
) -> harness.HarnessResult:
    """
    校验一轮；若 harness 提示"证据不足"，先强制补查再复验一次。

    # 为什么这一步不能省？
    # 追问类问题中，模型可能不调工具、直接用历史作答。
    # 此时 V4 失败原因是"没证据"而非"证据矛盾"——直接打回，模型手里还是没证据，
    # 重写出来的仍是没依据的话，最后只能兜底转人工，等于用户被无端拒绝。
    # 先补一次查询，复验大概率直接通过，回复延迟只多约 20ms（一次 SQL）。
    """
    result = harness.validate(output, trace)
    if result.needs_order_query:
        inject_order_evidence(messages, trace, user_id)
        result = harness.validate(output, trace)
    return result


def inject_order_evidence(
    messages: list[dict[str, Any]],
    trace: list[ToolCallRecord],
    user_id: int,
    call_id: str = "forced_orders",
) -> None:
    """
    强制执行一次 get_user_orders，结果回填进对话和工具轨迹。

    Args:
        messages: 就地追加 assistant(tool_calls) + tool 两条消息。
        trace: 就地追加一条 ToolCallRecord，harness 后续从这里取证据。
        user_id: 由 JWT 解出后传入，模型拿不到，也不参与参数协议。
        call_id: 伪造的 tool_call_id。预取用 prefetch_orders、补查用 forced_orders，
                 两者不同是为了避免同一轮里出现两条 id 相同的 tool 消息。

    Raises:
        无。execute_tool 内部把工具异常收敛成 ToolResult，不向上抛。

    消息格式说明：OpenAI 协议要求 role=tool 消息必须挂在一条带 tool_calls 的
    assistant 消息后面，否则下一轮请求直接 400。所以这里要先伪造一条
    assistant 的工具调用意图，再挂 tool 结果 —— 不是给模型看的戏，
    是协议闭合的硬要求。

    # 为什么这个函数对 agent.py 也公开？
    # "查一次库 + 按协议回填上下文 + 记一条轨迹"这件事有两个调用方：
    # refine 的证据不足补查，和 agent 的全量盘点预取。两处各写一遍，
    # 协议闭合那段代码必然走样（最容易漏的是 assistant 的 tool_calls 前缀）。
    # 抽成一个函数、放在 refine 这个"证据收口"功能单元里，双方共用一份实现。
    """
    result = execute_tool("get_user_orders", "{}", user_id)
    trace.append(
        ToolCallRecord(
            tool_name="get_user_orders",
            arguments={},
            result=result.data,
            success=result.success,
        )
    )
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_user_orders", "arguments": "{}"},
                }
            ],
        }
    )
    # 和工具循环里保持一致：indent=2 展开，避免模型复述时跨订单串数据
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
        }
    )
    logger.info(
        "强制查库委托单 | user=%s 来源=%s success=%s", user_id, call_id, result.success
    )


def _fallback(result: harness.HarnessResult | None, error: str) -> RefineResult:
    """
    兜底收口：固定话术 + REQUEST_HANDOFF。

    # 为什么兜底选"转人工"而不是"请稍后重试"？
    # 走到这里说明系统拿不出任何可信信息（模型两次都没改对），
    # "稍后重试"是让用户自己猜，"转人工"至少有人能接手。
    # 同时保留 validation（含 error_codes），让 agent_runs 能记录这次拒绝。
    """
    logger.error("收口失败，走兜底 | codes=%s 错误=%s",
                 result.error_codes if result else [], error)
    return RefineResult(
        output=AgentOutput(
            reply_type=ReplyType.REQUEST_HANDOFF,
            content=FALLBACK_CONTENT,
            action=None,
        ),
        validation=result,
        error=error,
    )
