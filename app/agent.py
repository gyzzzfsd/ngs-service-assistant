"""
agent.py —— 一次问答的编排

职责：
    把"用户这一句话"变成"一份 AgentRun"：
      1. 拼 system prompt + 历史 + 用户消息
      2. 确定性快速通道：转人工直接转接；全量盘点先预取本轮委托单证据
      3. 多轮工具循环（模型要工具 → 执行 → 结果回填 → 再问模型）
      4. 把草稿交给 refine 收口（结构化 → 五关卡校验 → 纠错 → 兜底）
      5. 全程记录工具轨迹，harness 从这里取证据

不负责：
    调模型的细节（llm.py）、工具怎么实现（tools.py）、校验规则（harness.py）、
    收口状态机（refine.py）、提示词文案（prompts.py）、JSON 容错解析（json_repair.py）。

为什么只做两阶段（先答、再结构化）而不是一次就让模型出 JSON：
    带着工具循环时，模型每轮都要决定"调工具还是回答"，此时再要求它严格 JSON，
    两个约束会互相干扰，实测容易输出半个 JSON 或把工具参数塞进正文。
    拆成"先把话说明白，再照着 schema 转一遍"两步，每步只有一个目标，稳定得多。
    代价是多一次模型调用（约 1 秒），换来的是解析失败率接近零。
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.config import settings
from app.harness import HarnessResult, validate
from app.llm import LLMError, chat
from app.prompts import FALLBACK_CONTENT, HANDOFF_ACK_CONTENT, SYSTEM_PROMPT
from app.refine import RefineResult, inject_order_evidence, refine_draft, wants_human, wants_inventory
from app.schemas import AgentOutput, ReplyType, ToolCallRecord
from app.tools import TOOL_SCHEMAS, execute_tool

logger = logging.getLogger(__name__)


@dataclass
class AgentRun:
    """一次 agent 运行的完整产出。"""

    output: AgentOutput
    tool_trace: list[ToolCallRecord] = field(default_factory=list)
    draft: str = ""                          # 模型整理成 JSON 之前的自然语言回答
    messages: list[dict[str, Any]] = field(default_factory=list)  # 完整对话（含补查的 tool 消息）
    error: str | None = None                 # 非空表示走了兜底路径，内容为失败原因
    validation: HarnessResult | None = None  # harness 校验结论（error 非空时也可能有值）
    corrected: bool = False                  # True = 首次未过、重写后通过


# 对外入口


def run(
    user_message: str,
    history: list[dict[str, Any]],
    user_id: int,
    nickname: str = "",
) -> AgentRun:
    """
    跑一次完整的问答。

    Args:
        user_message: 用户这一轮说的话。
        history: 之前的历史消息，OpenAI 格式（role/content），已按时间正序。
        user_id: 当前登录用户 ID，用于工具查询时注入 SQL（模型拿不到它）。
        nickname: 用户昵称，仅用于让称呼更自然，不参与任何鉴权。

    Returns:
        AgentRun。正常路径 error 为 None；模型或校验彻底失败时 output 为兜底话术、
        error 里带原因、validation 里带校验轨迹（api_chat 据此写 agent_runs 表）。

    异常处理策略：
        任何 LLMError 都不向上抛，统一转成兜底 AgentOutput。
        理由：聊天接口返回 500 对用户没有任何帮助，而"转人工"至少能让问题被跟进。
    """
    trace: list[ToolCallRecord] = []
    messages = _build_messages(user_message, history, nickname)
    started = time.perf_counter()

    # ---------- 第零步：用户点名要人工，直接转接 ----------
    # 见 refine.wants_human 的说明：这是明确指令，用词表判比让模型判更稳，
    # 而且省掉两次模型调用（约省 2~5 秒）。
    if wants_human(user_message):
        return _handoff_run(user_message, user_id, messages, started)

    logger.info(
        "agent 开始 | user=%s 历史=%d条 输入=%d字 | %s",
        user_id,
        len(messages) - 2,
        len(user_message),
        _brief(user_message),
    )

    # ---------- 第零步之二：全量盘点，先把本轮真实数据摆到模型面前 ----------
    # 与 wants_human 同构：确定性意图不让模型自己判。区别是这里不返回固定话术，
    # 而是先查库、把结果连同"历史作废"的提示注入上下文，后面照常走工具循环与收口。
    if wants_inventory(user_message):
        inject_order_evidence(messages, trace, user_id, call_id="prefetch_orders")
        messages.append(
            {
                "role": "user",
                "content": (
                    "（系统提示：上面是本轮刚查到的最新委托单与样本数据。"
                    "回答数量时必须按样本编号逐条计数（一张委托单可能含多条样本），"
                    "并逐条列出全部样本；问『一共』时不要自行按状态筛选。"
                    "历史消息里提到的数量与样本清单一律作废，不要沿用、也不要省略任何一条样本。）"
                ),
            }
        )
        logger.info(
            "全量盘点快速通道：已预取本轮委托单 | user=%s 输入=%s",
            user_id,
            _brief(user_message),
        )

    # ---------- 第一步：工具循环，拿到自然语言草稿 ----------
    try:
        draft = _run_tool_loop(messages, user_id, trace)
    except LLMError as exc:
        logger.error(
            "agent 降级（模型失败）| user=%s 已耗时=%.0fms 原因=%s",
            user_id,
            (time.perf_counter() - started) * 1000,
            exc,
        )
        return AgentRun(
            output=_fallback_output(),
            tool_trace=trace,
            messages=messages,
            error=f"LLM_ERROR: {exc}",
        )

    # ---------- 第二步：收口（结构化 → 五关卡 → 纠错 → 兜底） ----------
    try:
        refined: RefineResult = refine_draft(messages, draft, trace, user_id)
    except LLMError as exc:
        # 收口阶段的模型调用也可能整体失败（限流、超时），同样只降级不抛
        logger.error(
            "agent 降级（收口阶段模型失败）| user=%s 已耗时=%.0fms 原因=%s",
            user_id,
            (time.perf_counter() - started) * 1000,
            exc,
        )
        return AgentRun(
            output=_fallback_output(),
            tool_trace=trace,
            draft=draft,
            messages=messages,
            error=f"LLM_ERROR: {exc}",
        )

    _log_finish(user_id, started, refined, len(trace))
    return AgentRun(
        output=refined.output,
        tool_trace=trace,
        draft=draft,
        messages=messages,
        error=refined.error,
        validation=refined.validation,
        corrected=refined.corrected,
    )


def _handoff_run(
    user_message: str,
    user_id: int,
    messages: list[dict[str, Any]],
    started: float,
) -> AgentRun:
    """
    用户点名要人工时的快速通道：不调模型，直接给 REQUEST_HANDOFF。

    Args:
        user_message: 用户本轮输入（只用于打日志）。
        user_id: 当前登录用户。
        messages: 已组装好的上下文（原样带出，方便排查时看到模型本该看到什么）。
        started: 计时起点，用于算耗时。

    Returns:
        AgentRun。validation 也照常跑一遍 —— 这句话不含任何业务事实，
        过 V1 一定通过，agent_runs 里记成 passed，统计口径不会歪。

    # 为什么不直接构造 AgentOutput 就返回，还要过一遍 harness？
    # 因为 agent_runs 的 validation_status 是从 validation 里算出来的。
    # 跳过校验 → validation 为 None → 会被记成 degraded_model，
    # 统计页面就会莫名其妙多出一堆"模型故障"，而其实什么都没坏。
    # 走一遍成本为零（纯查表），换来的是观测数据和真实情况一致。
    """
    output = AgentOutput(
        reply_type=ReplyType.REQUEST_HANDOFF,
        content=HANDOFF_ACK_CONTENT,
        action=None,
    )
    logger.info(
        "用户点名要人工，跳过模型直接转接 | user=%s 输入=%s 耗时=%.0fms",
        user_id,
        _brief(user_message),
        (time.perf_counter() - started) * 1000,
    )
    return AgentRun(
        output=output,
        messages=messages,
        validation=validate(output, []),
    )


def _log_finish(user_id: int, started: float, refined: RefineResult, tool_count: int) -> None:
    """打一条收尾日志：类型、校验结论、纠错次数、耗时，一行看全。"""
    validation = refined.validation
    if validation is None:
        verdict = "未校验（结构化失败）"
    elif validation.degraded:
        verdict = f"V2降级({validation.degraded})"
    elif validation.passed:
        verdict = "校验通过" + ("（重写后）" if refined.corrected else "")
    else:
        verdict = f"校验拒绝 codes={validation.error_codes}"

    logger.info(
        "agent 完成 | user=%s 类型=%s %s 工具=%d次 耗时=%.0fms | 回复=%s",
        user_id,
        refined.output.reply_type.value,
        verdict,
        tool_count,
        (time.perf_counter() - started) * 1000,
        _brief(refined.output.content, 60),
    )


# 内部步骤


def _build_messages(
    user_message: str,
    history: list[dict[str, Any]],
    nickname: str,
) -> list[dict[str, Any]]:
    """
    组装本轮发给模型的完整消息列表。

    Args:
        user_message: 本轮用户输入。
        history: 历史消息（已转成 OpenAI 格式）。
        nickname: 用户昵称。

    Returns:
        [system, *history, user] 的消息列表。

    为什么只取最近 N 条历史：
        agent_history_limit 默认 20。多轮聊天很容易把上下文堆到几千 token，
        而客服场景只需要最近几轮 —— 用户不会在第 30 轮还引用第 1 轮的内容。
    """
    system = SYSTEM_PROMPT
    if nickname:
        system += f"\n## 七、当前用户\n这位用户叫「{nickname}」，可以自然地称呼对方，但不要反复叫。"
    # 明确告诉模型"你拿不到 user_id"，避免它试图在参数里构造用户标识
    system += "\n你无法访问用户的内部 ID，所有按用户过滤的工作由工具自动完成。"

    trimmed = history[-settings.agent_history_limit :] if settings.agent_history_limit > 0 else []
    return [
        {"role": "system", "content": system},
        *trimmed,
        {"role": "user", "content": user_message},
    ]


def _run_tool_loop(
    messages: list[dict[str, Any]],
    user_id: int,
    trace: list[ToolCallRecord],
) -> str:
    """
    多轮工具循环，返回模型最后的自然语言回答。

    Args:
        messages: 会被就地追加 assistant / tool 消息（调用方后续还要用它做结构化）。
        user_id: 注入给工具的用户 ID。
        trace: 就地追加工具调用记录，harness 后续从这里取证据。

    Returns:
        模型最终的文本回答（非空）。

    Raises:
        LLMError: 模型调用失败（已含备用模型尝试）。

    # 轮数用完时不能报错，要逼模型就现有证据作答。
    # 模型可能在第 5 轮还在要工具（说明它没找对方向）。此时直接失败会让用户
    # 白等十几秒还拿不到任何回复；正确做法是撤掉工具、加一句"请基于已查到的信息回答"，
    # 让它在已有证据上给一个（可能不完美但可信的）答复，剩下的交给 harness 拦截。
    """
    max_rounds = max(1, settings.agent_max_tool_rounds)

    for round_no in range(1, max_rounds + 1):
        reply = chat(messages, tools=TOOL_SCHEMAS)

        # 没有工具调用 → 这一轮就是最终回答
        if not reply.tool_calls:
            logger.info("工具循环 | 第 %d 轮直接作答，未再调工具", round_no)
            return reply.content or ""

        logger.info(
            "工具循环 | 第 %d/%d 轮，模型请求工具：%s",
            round_no,
            max_rounds,
            [tc.name for tc in reply.tool_calls],
        )

        # 有工具调用：先把 assistant 的"我要调这些工具"写回上下文，
        # 再把每个工具的执行结果以 role=tool 追加进去。
        # 这两步的消息格式必须严格符合 OpenAI 协议，否则下一轮请求会 400。
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for tc in reply.tool_calls
                ],
            }
        )

        for tc in reply.tool_calls:
            result = execute_tool(tc.name, tc.arguments, user_id)
            trace.append(
                ToolCallRecord(
                    tool_name=tc.name,
                    arguments=_safe_json(tc.arguments),
                    result=result.data,
                    success=result.success,
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    # 用 indent=2 展开而不是压成一行：
                    # 压成一行的 JSON 让模型很难看清"哪个金额属于哪个委托单"，
                    # 会导致跨单串数据（把委托单 2 的金额说成委托单 1 的）。
                    # 多花几十个 token，换来的是复述准确率明显提升。
                    "content": json.dumps(result.model_dump(), ensure_ascii=False, indent=2),
                }
            )

    # ---------- 轮数耗尽：撤掉工具，强制收口 ----------
    logger.warning("工具轮数已达上限 %d 轮，撤掉工具强制收口", max_rounds)
    messages.append(
        {
            "role": "user",
            "content": "（系统提示：工具调用次数已用完，请只基于上面已经查到的信息回答，不要再说要查询。）",
        }
    )
    final = chat(messages, tools=None)
    return final.content or ""


def _safe_json(text: str) -> dict[str, Any]:
    """把模型给的参数 JSON 解析成 dict，解析不了就原样塞进 _raw 保留证据。"""
    try:
        obj = json.loads(text or "{}")
        return obj if isinstance(obj, dict) else {"_raw": text}
    except json.JSONDecodeError:
        return {"_raw": text}


def _fallback_output() -> AgentOutput:
    """
    兜底输出：模型彻底失败时给用户的话术。

    选 REQUEST_HANDOFF 而不是 ANSWER，是因为此时系统不知道任何真实信息，
    唯一负责任的做法是把人接过来。这也是 harness 降级矩阵里的一条。
    """
    return AgentOutput(
        reply_type=ReplyType.REQUEST_HANDOFF,
        content=FALLBACK_CONTENT,
        action=None,
    )


def _brief(text: str, limit: int = 60) -> str:
    """把长文本压成一行用于日志，避免把整段回复灌进日志文件。"""
    one_line = (text or "").replace("\n", " ").strip()
    return one_line if len(one_line) <= limit else one_line[:limit] + "..."
