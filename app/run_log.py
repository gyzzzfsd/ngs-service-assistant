"""
run_log.py —— agent_runs 观测表的写入与摘要

职责：
    把一次问答的校验结果变成"能看的东西"：
      - summarize()：AgentRun → 前端可用的字典（ChatResponse.validation 的来源）
      - record_run()：同样的结论落进 agent_runs 表，用于追溯"哪次被拦、为什么"

不负责：
    校验本身（harness.py）、怎么跑出这次结果（agent.py/refine.py）。

为什么单独一个文件：
    观测是旁路功能 —— 它挂了不能影响主流程（record_run 里兜了异常），
    两类改动放在同一文件里容易互相干扰，
    而且"改日志格式"和"改校验规则"的人不该在同一份文件里打架。

status 取值（与 script/agent_runs.sql 的注释一一对应）：
    passed / corrected / rejected / degraded_tools / degraded_model
"""

import logging
from typing import Any

from app.db import get_conn

logger = logging.getLogger(__name__)


def summarize(run: Any) -> dict[str, Any]:
    """
    把一次 AgentRun 压成前端/观测表要的字典。

    Args:
        run: agent.AgentRun（鸭子类型：只取属性，不 import agent，避免循环依赖）。

    Returns:
        {"status", "passed", "corrected", "error_codes", "degraded"}。
    """
    validation = getattr(run, "validation", None)
    if validation is None:
        # 模型调用或结构化直接失败，没走到校验
        code = _error_code_of(getattr(run, "error", None))
        return {
            "status": "degraded_model",
            "passed": False,
            "corrected": False,
            "error_codes": [code] if code else [],
            "degraded": "model",
        }

    if validation.degraded == "tool_failure":
        status = "degraded_tools"
    elif validation.passed and getattr(run, "corrected", False):
        status = "corrected"
    elif validation.passed:
        status = "passed"
    else:
        status = "rejected"
    return {
        "status": status,
        "passed": validation.passed,
        "corrected": bool(getattr(run, "corrected", False)),
        "error_codes": list(validation.error_codes),
        "degraded": validation.degraded,
    }


def record_run(
    conversation_id: int,
    user_id: int,
    question: str,
    reply_type: str,
    status: str,
    error_codes: list[str],
    tool_call_count: int,
    duration_ms: int,
) -> None:
    """
    把一次问答的校验结果写入 agent_runs 观测表。

    Raises:
        Exception: 落库失败只打 ERROR 日志不上抛 —— 观测数据写不进去
        不能影响用户拿到回复（观测是旁路，不是主流程）。
    """
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_runs
                    (conversation_id, user_id, question, reply_type,
                     validation_status, error_codes, tool_call_count, duration_ms)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (conversation_id, user_id, question, reply_type, status, error_codes, tool_call_count, duration_ms),
            )
        logger.info(
            "agent_runs 落库 | conv=%s user=%s status=%s codes=%s 工具=%d次 耗时=%dms",
            conversation_id, user_id, status, error_codes or "[]", tool_call_count, duration_ms,
        )
    except Exception:  # noqa: BLE001 —— 观测旁路，失败不上抛
        logger.exception("agent_runs 落库失败（不影响回复）| conv=%s", conversation_id)


def _error_code_of(error: str | None) -> str | None:
    """从 AgentRun.error 字符串里抠出错误码前缀（格式约定：'CODE: 详情'）。"""
    if not error:
        return None
    return error.split(":", 1)[0].strip()
