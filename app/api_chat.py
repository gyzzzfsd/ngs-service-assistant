"""
api_chat.py —— 聊天接口

职责：
    会话的创建、历史消息读取、发消息并同步拿到 AI 回复。
    你在前端点"发送"后发生的所有事情，入口都在这个文件。

不负责：
    AI 怎么想（agent.py）、消息是否可信（harness.py）、
    写操作确认（api_actions.py）、坐席端接口与转人工流转（api_handoff.py）。

阶段④ 说明：
    人工接待中（QUEUED/HUMAN）的用户消息在本文件分流到 _relay_to_agent：
    只入库并推给坐席，绝不调 AI —— 人工在场时 AI 抢答是接线事故。
"""

import logging
import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.rows import dict_row

from app import agent, run_log
from app.db import get_conn
from app.prompts import FALLBACK_CONTENT
from app.schemas import (
    ChatRequest,
    ChatResponse,
    ConversationInfo,
    MessageItem,
    UserMessageAck,
)
from app.security import CurrentUser, get_current_user
from app.ws import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["聊天"])


# 会话


@router.post("", response_model=ConversationInfo)
def create_conversation(current: CurrentUser = Depends(get_current_user)) -> ConversationInfo:
    """
    进入客服页时调用：有进行中的 AI 会话就复用，没有就新建。

    Args:
        current: 当前登录用户。

    Returns:
        会话信息（含 id 和状态）。

    为什么要复用而不是每次都新建：
        如果每次刷新页面都建一个新会话，聊天记录会被切成很多碎片，
        agent 的 history 也就断了（它只看当前会话的消息）。
        复用"最近一个仍在 AI 接待中的会话"，多轮上下文才能连续。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, status, created_at, last_active_at
            FROM conversations
            WHERE user_id = %s AND status = 'AI'
            ORDER BY last_active_at DESC
            LIMIT 1
            """,
            (current.user_id,),
        )
        row = cur.fetchone()

        if row is None:
            cur.execute(
                """
                INSERT INTO conversations (user_id, status)
                VALUES (%s, 'AI')
                RETURNING id, status, created_at, last_active_at
                """,
                (current.user_id,),
            )
            row = cur.fetchone()
            logger.info("新建会话 | conv=%s user=%s", row["id"], current.user_id)
        else:
            logger.info("复用会话 | conv=%s user=%s", row["id"], current.user_id)

    return ConversationInfo(**row)


@router.get("/mine", response_model=ConversationInfo | None)
def my_conversation(current: CurrentUser = Depends(get_current_user)) -> ConversationInfo | None:
    """
    返回当前用户最近一个会话（任意状态），没有则返回 null。

    前端刷新页面后用它决定"要不要显示聊天气泡、当前是不是人工接待中"。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, status, created_at, last_active_at
            FROM conversations
            WHERE user_id = %s
            ORDER BY last_active_at DESC
            LIMIT 1
            """,
            (current.user_id,),
        )
        row = cur.fetchone()

    return ConversationInfo(**row) if row else None


@router.get("/{conversation_id}/messages", response_model=list[MessageItem])
def list_messages(
    conversation_id: int,
    limit: int = 50,
    current: CurrentUser = Depends(get_current_user),
) -> list[MessageItem]:
    """
    按时间正序返回一个会话的历史消息。

    Args:
        conversation_id: 会话 ID。
        limit: 最多返回多少条（默认 50，最多 200）。

    Raises:
        HTTPException 404: 会话不存在，或不属于当前用户。
    """
    _get_owned_conversation(conversation_id, current.user_id, allow_agent=current.role == "agent")

    limit = max(1, min(limit, 200))
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        # 先按 id 倒序取最近 N 条，再翻转成正序，这样拿到的是"最近 N 条"而不是"最早 N 条"
        cur.execute(
            """
            SELECT id, role, content, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (conversation_id, limit),
        )
        rows = cur.fetchall()

    return [MessageItem(**r) for r in reversed(rows)]


# 发消息（核心）


@router.post("/{conversation_id}/messages", response_model=ChatResponse | UserMessageAck)
def send_message(
    conversation_id: int,
    body: ChatRequest,
    current: CurrentUser = Depends(get_current_user),
) -> ChatResponse:
    """
    用户发一条消息，同步返回 AI 的结构化回复。

    Args:
        conversation_id: 会话 ID。
        body: 消息内容。
        current: 当前登录用户（user_id 只从这里来，请求体里不接受用户 ID）。

    Returns:
        ChatResponse：落库后的两条消息 ID + 模型输出的 AgentOutput。

    Raises:
        HTTPException 404: 会话不存在或不属于当前用户。
        HTTPException 409: 会话已关闭（终态）。

    执行顺序（顺序本身很重要）：
        1. 校验会话归属
        2. 状态分流：QUEUED/HUMAN → 只入库并推给坐席（不调 AI），直接返回
        3. 先读历史，再插入本条 —— 如果先插入再读历史，这条消息会在 history 里出现一次，
           agent 又把它当本轮输入加一次，模型会看到同一句话两遍
        4. 调 agent（内含五关卡校验）
        5. 落库 AI 回复（用 output.content，而不是模型整理 JSON 前的草稿）
        6. 写 agent_runs 观测表
        7. 更新 last_active_at
    """
    started = time.perf_counter()
    conv = _get_owned_conversation(conversation_id, current.user_id)

    # ---------- 状态分流（阶段④）----------
    if conv["status"] in ("QUEUED", "HUMAN"):
        # 人工接待中：AI 不抢答。消息入库后推给所有在线坐席，
        # 由坐席在工单页看到并回复（role=human，见 api_handoff.agent_reply）。
        return _relay_to_agent(conversation_id, current.user_id, body.content, conv["status"])

    if conv["status"] == "CLOSED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="会话已关闭，请开始新的会话",
        )

    logger.info(
        "收到消息 | conv=%s user=%s 长度=%d 内容=%s",
        conversation_id,
        current.user_id,
        len(body.content),
        body.content[:50].replace("\n", " "),
    )

    history = _load_history(conversation_id)

    nickname = _load_nickname(current.user_id)
    result = agent.run(
        user_message=body.content,
        history=history,
        user_id=current.user_id,
        nickname=nickname,
    )

    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s) RETURNING id",
            (conversation_id, body.content),
        )
        user_message_id = cur.fetchone()["id"]

        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'ai', %s) RETURNING id",
            (conversation_id, result.output.content),
        )
        ai_message_id = cur.fetchone()["id"]

        cur.execute("UPDATE conversations SET last_active_at = now() WHERE id = %s", (conversation_id,))

    logger.info(
        "回复完成 | conv=%s user=%s 类型=%s 耗时=%.0fms msg=%s/%s%s",
        conversation_id,
        current.user_id,
        result.output.reply_type.value,
        (time.perf_counter() - started) * 1000,
        user_message_id,
        ai_message_id,
        f" | 降级原因={result.error}" if result.error else "",
    )

    # 校验结果摘要：run_log.summarize 产出 status / passed / corrected / error_codes，
    # 前端按 status 决定提示样式，agent_runs 落库也用它，两边永远一致。
    validation = run_log.summarize(result)

    # 观测落库（旁路）：写失败不影响回复，record_run 内部已兜住异常
    run_log.record_run(
        conversation_id=conversation_id,
        user_id=current.user_id,
        question=body.content,
        reply_type=result.output.reply_type.value,
        status=validation["status"],
        error_codes=validation["error_codes"],
        tool_call_count=len(result.tool_trace),
        duration_ms=int((time.perf_counter() - started) * 1000),
    )

    return ChatResponse(
        conversation_id=conversation_id,
        user_message_id=user_message_id,
        ai_message_id=ai_message_id,
        reply=result.output,
        validation=validation,
    )


# 内部工具函数


def _relay_to_agent(
    conversation_id: int, user_id: int, content: str, conv_status: str
) -> UserMessageAck:
    """
    人工接待期间的用户消息：入库（role=user）+ 推给在线坐席，不调 AI。

    Args:
        conversation_id: 会话 ID。
        user_id: 发消息的用户。
        content: 消息正文（已经过 ChatRequest 的空白清洗）。
        conv_status: 当前状态（QUEUED / HUMAN），只用于回显。

    Returns:
        UserMessageAck。注意没有 AI 回复字段 —— 前端在人工模式下
        也不该等待回复气泡，坐席的回答会通过 WebSocket 推过来。

    # 为什么推给"所有坐席"而不是"接管的那个"？
    # 本项目只有一个坐席（agent001）且无坐席-会话绑定表；
    # 广播给全部在线坐席，前端各自过滤"是我正在看的会话"即可。
    # 真实系统会多一层分配（坐席表 + 认领会话记录），那是二期的事。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'user', %s) RETURNING id, created_at",
            (conversation_id, content),
        )
        msg = cur.fetchone()
        cur.execute("UPDATE conversations SET last_active_at = now() WHERE id = %s", (conversation_id,))

    manager.notify_agents(
        {
            "type": "message",
            "conversation_id": conversation_id,
            "message": {
                "id": msg["id"],
                "role": "user",
                "content": content,
                "created_at": msg["created_at"].isoformat(),
            },
        }
    )
    logger.info(
        "人工模式消息入库 | conv=%s user=%s status=%s msg=%s",
        conversation_id, user_id, conv_status, msg["id"],
    )
    return UserMessageAck(
        conversation_id=conversation_id,
        user_message_id=msg["id"],
        status=conv_status,
    )


def _get_owned_conversation(
    conversation_id: int, user_id: int, allow_agent: bool = False
) -> dict[str, Any]:
    """
    取出会话并校验归属。

    Args:
        conversation_id: 会话 ID。
        user_id: 当前登录用户。
        allow_agent: 坐席豁免归属校验（坐席需要读任意会话的历史与消息，
            见 list_messages / send_message 的调用方）。

    Returns:
        会话行（id / status / created_at / last_active_at）。

    Raises:
        HTTPException 404: 不存在，或存在但不属于当前用户。

    # 两种情况都返回 404，绝不返回 403。
    # 403 等于告诉攻击者"这个 ID 确实存在，只是你没权限"，可以被用来枚举别人的会话。
    # 404 让"不存在"和"不是你的"在响应上完全一致，试探不出任何信息。
    # （坐席豁免走 role 判断，role 来自 JWT 签发，伪造不了。）
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, user_id, status, created_at, last_active_at FROM conversations WHERE id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()

    if row is None or (row["user_id"] != user_id and not allow_agent):
        # 日志里区分两种情况（安全审计需要），但响应对外完全一致（防会话枚举）。
        if row is not None:
            logger.warning(
                "越权访问拦截 | conv=%s 归属用户=%s 请求用户=%s",
                conversation_id,
                row["user_id"],
                user_id,
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return row


def _load_history(conversation_id: int) -> list[dict[str, Any]]:
    """
    把库里的历史消息转成 OpenAI 消息格式（不含本轮这条，它还没入库）。

    Returns:
        [{"role": "user"|"assistant", "content": "..."}]，按时间正序。

    角色映射：
        user  → user       用户说的话
        ai    → assistant  AI 说的话
        human → assistant  人工客服说的话

    # 人工客服的话为什么也映射成 assistant 而不是 user？
    # 对模型来说，"自己这边（客服侧）说过的话"都是 assistant，
    # 映射成 user 会让模型以为用户在自言自语，下一轮回答会混乱。
    # 但必须加 [人工客服] 前缀，否则模型会以为自己说过那些话，
    # 进而把人工的承诺当成自己的承诺继续往下说（甩锅和重复承诺都由此而来）。

    # 为什么要把"兜底话术"从历史里剔除？
    # 兜底话术是"系统出故障"时说的话，不是 AI 的真实回答。但它一旦落库，
    # 下一轮就会作为 assistant 的历史重新喂给模型 —— 模型看到"我上次就是这么答的"，
    # 于是继续照抄同一句兜底话术。实测这会形成自我强化的死循环：
    # 一次失败之后，这个会话再也回答不了任何问题。
    # 所以：兜底内容对用户可见（要落库、要能查历史），但绝不进模型上下文。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = %s
            ORDER BY id DESC
            LIMIT 40
            """,
            (conversation_id,),
        )
        rows = list(reversed(cur.fetchall()))

    messages: list[dict[str, Any]] = []
    for row in rows:
        if row["role"] == "user":
            messages.append({"role": "user", "content": row["content"]})
        elif row["role"] == "ai":
            # 兜底话术不进上下文，见下面的说明。
            if row["content"] == FALLBACK_CONTENT:
                continue
            messages.append({"role": "assistant", "content": row["content"]})
        else:  # human
            messages.append({"role": "assistant", "content": f"[人工客服] {row['content']}"})
    return messages


def _load_nickname(user_id: int) -> str:
    """取用户昵称，仅用于让 AI 的称呼更自然；取不到就返回空串，不影响主流程。"""
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT nickname FROM users WHERE id = %s", (user_id,))
        row = cur.fetchone()
    return (row["nickname"] or "") if row else ""
