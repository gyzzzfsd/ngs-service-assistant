"""
llm.py —— 大模型调用

职责：
    把智谱 GLM（OpenAI 兼容协议）的调用封装成一个 chat() 函数：
      - 关掉思考模式（省 3~5 秒/轮）
      - 传工具定义，把模型返回的 tool_calls 提取成干净的 dataclass
      - 限流/超时重试一次，主力模型失败时自动切备用模型

不负责：
    组装 prompt（agent.py）、执行工具（tools.py）、校验输出（harness.py）。

为什么单独一个文件：
    换模型厂商只需要改这一个文件；agent.py 只认 chat() 这个函数，不认 SDK。
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """模型调用失败的统一异常。上层（agent）据此决定降级话术。"""


@dataclass
class ToolCallRequest:
    """模型发起的一次工具调用请求（已从 SDK 对象里摘出来，agent 不必认识 SDK）。"""

    id: str        # SDK 给的回调 id，回填工具结果时必须原样带回
    name: str      # 工具名
    arguments: str  # 参数 JSON 字符串（模型给的是文本，可能不合法，由 tools.py 解析）


@dataclass
class LLMReply:
    """一次模型调用的结果。"""

    content: str | None                       # 文本内容；纯工具调用时为 None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    model: str = ""                            # 实际使用的模型（可能是备用模型）
    finish_reason: str = ""
    # token 用量，用于日志里观察成本；接口不一定每次都返回，所以给默认 0
    prompt_tokens: int = 0
    completion_tokens: int = 0


_client: OpenAI | None = None


def get_client() -> OpenAI:
    """
    返回全局唯一的 OpenAI 兼容客户端（懒加载，首次调用时创建）。

    max_retries=0：关掉 SDK 自带的重试。
    否则 SDK 内部重试 2 次 × 我们外层再重试，最坏情况要等 6 个超时周期，
    前端早就断开了。重试策略统一由本文件的 _call_once / chat 控制。
    """
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout,
            max_retries=0,
        )
    return _client


def _thinking_body() -> dict[str, Any]:
    """
    构造"思考模式"开关参数。

    Returns:
        要传给 extra_body 的字典；不需要时返回空字典。

    # glm-4.5 系列默认会先输出一段思考（reasoning_content），
    # 单轮多花 3~5 秒，而客服场景基本不需要。
    # 传 thinking={"type": "disabled"} 可以关掉。
    # 但 glm-5.3-flash 这类模型不支持关闭，传了会直接报错 1210，
    # 所以下面 _call_once 里必须能在收到 1210 时去掉该参数重试一次。
    """
    if settings.llm_disable_thinking:
        return {"thinking": {"type": "disabled"}}
    if settings.llm_reasoning_effort:
        return {"thinking": {"type": settings.llm_reasoning_effort}}
    return {}


def _call_once(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    json_mode: bool = False,
) -> LLMReply:
    """
    真正发一次请求（不含重试），把 SDK 的响应转成 LLMReply。

    Args:
        model: 模型名。
        messages: OpenAI 格式的消息列表。
        tools: 工具定义列表（OpenAI function calling 格式），None 表示不带工具。
        json_mode: True 时要求模型输出合法 JSON 对象。
            用于 agent.py 的最后一步"把回答整理成 AgentOutput"。

    Returns:
        LLMReply。

    Raises:
        LLMError: 返回了空内容且没有工具调用（模型异常，不是业务失败）。
        Exception: 网络/鉴权/限流等 SDK 原始异常，由上层决定重试还是降级。
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": settings.llm_temperature,
        "max_tokens": settings.llm_max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if json_mode:
        # 注意：json_object 模式要求 prompt 里出现 "json" 字样，否则部分模型会报错。
        # agent.py 的格式指令里已经显式写了 JSON，满足这个前提。
        kwargs["response_format"] = {"type": "json_object"}

    extra = _thinking_body()
    if extra:
        kwargs["extra_body"] = extra

    client = get_client()
    started = time.perf_counter()

    # 完整消息只在 DEBUG 打：里面有用户原话和工具结果，量大且涉及隐私，
    # INFO 级别只记"调了谁、多大、多久"。
    logger.debug(
        "请求模型 | model=%s messages=%d tools=%d json_mode=%s",
        model,
        len(messages),
        len(tools or []),
        json_mode,
    )

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:
        # 关思考不被支持时（错误码 1210），去掉该参数再试一次。
        # 这一步是必需的：运维改了 LLM_MODEL 却忘了把 LLM_DISABLE_THINKING 改成 false，
        # 报错信息只会说"参数不合法"，很难定位到是 thinking 引起的。
        if extra and "1210" in str(exc):
            logger.warning(
                "模型 %s 不接受 thinking 参数（1210），已去掉该参数重试；"
                "建议把 .env 的 LLM_DISABLE_THINKING 改成 false",
                model,
            )
            kwargs.pop("extra_body", None)
            resp = client.chat.completions.create(**kwargs)
        else:
            logger.error("模型 %s 调用失败：%s: %s", model, type(exc).__name__, exc)
            raise

    elapsed_ms = (time.perf_counter() - started) * 1000

    choice = resp.choices[0]
    msg = choice.message
    content = (msg.content or "").strip()

    tool_calls = [
        ToolCallRequest(
            id=tc.id,
            name=tc.function.name,
            arguments=tc.function.arguments or "{}",
        )
        for tc in (msg.tool_calls or [])
    ]

    # glm-5.3-flash 在 max_tokens 给得太小时会返回空 content（实测），
    # 表现为"成功响应但没有内容"。必须当成错误抛出去，让上层切备用模型或降级，
    # 否则用户会收到一条空白消息，还以为客服掉线了。
    if not content and not tool_calls:
        logger.warning(
            "模型 %s 返回空内容 | finish=%s max_tokens=%s",
            model,
            choice.finish_reason,
            settings.llm_max_tokens,
        )
        raise LLMError(
            f"模型 {model} 返回空内容（finish_reason={choice.finish_reason}），"
            f"可能是 max_tokens={settings.llm_max_tokens} 过小或被安全策略拦截"
        )

    usage = getattr(resp, "usage", None)
    reply = LLMReply(
        content=content or None,
        tool_calls=tool_calls,
        model=resp.model or model,
        finish_reason=choice.finish_reason or "",
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
    )
    logger.info(
        "模型返回 | model=%s 耗时=%.0fms 工具=%d tokens=%d+%d finish=%s",
        reply.model,
        elapsed_ms,
        len(reply.tool_calls),
        reply.prompt_tokens,
        reply.completion_tokens,
        reply.finish_reason,
    )
    return reply


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    json_mode: bool = False,
) -> LLMReply:
    """
    调一次模型。内置两次机会：主模型失败 → 备用模型。

    Args:
        messages: OpenAI 格式的消息列表，最后一条决定模型这一轮的产出。
        tools: 工具定义；传 None 用于"不带工具的纯文本生成"（例如让模型按 schema 出结构）。
        json_mode: 要求输出合法 JSON 对象（用于生成 AgentOutput）。

    Returns:
        LLMReply。

    Raises:
        LLMError: 主模型与备用模型都失败。

    重试策略（有意做得简单）：
        1. 主模型抛任何异常 → 立刻切备用模型，不做同模型重试。
           理由：限流（1305/1302）和超时的恢复时间都在秒级以上，
           立刻重试同一个模型大概率还是失败，反而多等一个超时。
        2. 备用模型也失败 → 抛 LLMError，由 agent 走固定降级话术。
    """
    try:
        return _call_once(settings.llm_model, messages, tools, json_mode)
    except Exception as primary_exc:
        if not settings.llm_fallback_model:
            logger.error("模型调用失败且未配置备用模型：%s", primary_exc)
            raise LLMError(f"模型调用失败：{primary_exc}") from primary_exc

        logger.warning(
            "主模型 %s 失败，0.5s 后切备用 %s | 原因：%s",
            settings.llm_model,
            settings.llm_fallback_model,
            primary_exc,
        )
        # 稍等一下再切备用，避开同一瞬间的限流窗口
        time.sleep(0.5)
        try:
            reply = _call_once(settings.llm_fallback_model, messages, tools, json_mode)
            logger.info("备用模型 %s 接管成功", settings.llm_fallback_model)
            return reply
        except Exception as fallback_exc:
            logger.error(
                "主备模型均失败 | 主：%s | 备：%s", primary_exc, fallback_exc
            )
            raise LLMError(
                f"主模型 {settings.llm_model} 失败（{primary_exc}）；"
                f"备用模型 {settings.llm_fallback_model} 也失败（{fallback_exc}）"
            ) from fallback_exc
