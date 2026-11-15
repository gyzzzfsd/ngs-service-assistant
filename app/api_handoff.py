"""
api_handoff.py —— 转人工与坐席端接口

职责：
    状态机后半段的全部流转：
        顾客侧：POST /api/conversations/{id}/handoff    AI → QUEUED
        坐席侧：GET  /api/agent/queued                  待接管列表（QUEUED）
                GET  /api/agent/active                  接待中列表（HUMAN）
                POST /api/agent/conversations/{id}/accept   QUEUED → HUMAN
                POST /api/agent/conversations/{id}/messages 人工回复（role=human）
                POST /api/agent/conversations/{id}/close    HUMAN → AI / CLOSED

不负责：
    用户在 HUMAN 模式下发的消息（走 api_chat 的 send_message，那里按状态分流）、
    推送通道本身（ws.py）、坐席登录（api_auth，agent001 就是普通登录）。

状态流转安全：
    所有状态修改都写成"UPDATE ... WHERE status=期望旧值"的条件更新，
    影响行数 0 视为流转失败（已被别人抢先/状态已变），返回 409。
    这与写操作（加急/复检/退单）的三重条件是同一条纪律：判断和修改必须在一条语句里。

# 坐席回复为什么不复用用户的发消息接口？
# 两者的职责完全不同：用户接口要"存消息 + 调 AI + 校验 + 落观测表"，
# 坐席接口只需要"存消息 + 推给用户"。硬塞进同一个函数会变成一堆
# if role == 'agent' 的分支泥潭。分开两个端点，各走各的直线逻辑。
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.rows import dict_row

from app.db import get_conn
from app.schemas import (
    ActiveConversation,
    AgentMessageRequest,
    CloseConversationRequest,
    HandoffRequest,
    HandoffResponse,
    MessageItem,
    QueuedConversation,
)
from app.security import CurrentUser, get_current_user
from app.ws import manager

logger = logging.getLogger(__name__)

# 顾客侧路由（与 api_chat 同前缀，不同 tag；FastAPI 允许多个 router 共用前缀）
router = APIRouter(prefix="/api", tags=["转人工与坐席"])

# 坐席工作台列表里"最近一条消息"的截断长度：够坐席判断主题即可，
# 太长会让队列页面变成消息墙
_QUEUE_PREVIEW_CHARS = 60


# 坐席权限依赖


def get_current_agent(current: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """
    坐席专用依赖：已登录且 role == 'agent'。

    # 为什么顾客访问坐席接口返回 403 而不是 404？
    # "坐席接口存在"不是秘密（写在公开文档里），需要藏的是数据（谁排在队列里）。
    # 所以权限层级用 403 明确说"身份不对"，数据层级（会话归属）才用 404 防枚举。
    """
    if current.role != "agent":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="该接口仅限人工坐席使用",
        )
    return current


# 顾客侧：请求转人工


@router.post("/conversations/{conversation_id}/handoff", response_model=HandoffResponse)
def request_handoff(
    conversation_id: int,
    body: HandoffRequest | None = None,
    current: CurrentUser = Depends(get_current_user),
) -> HandoffResponse:
    """
    用户点【转人工】按钮：AI → QUEUED，等待坐席接管。

    幂等设计：
        已经 QUEUED → 原样返回（重复点击不报错）；
        已经 HUMAN  → 原样返回（坐席已经在场，无需排队）；
        CLOSED      → 409（终态，需要重新开会话）。
    """
    _get_owned_conversation(conversation_id, current.user_id)

    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        # 条件更新：只有 AI 状态才会真正流转，其他状态原样读出
        cur.execute(
            """
            UPDATE conversations SET status = 'QUEUED', last_active_at = now()
            WHERE id = %s AND user_id = %s AND status = 'AI'
            RETURNING status
            """,
            (conversation_id, current.user_id),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT status FROM conversations WHERE id = %s AND user_id = %s",
                (conversation_id, current.user_id),
            )
            row = cur.fetchone()

    final_status = row["status"]
    if final_status == "CLOSED":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="会话已关闭，无法转人工"
        )

    if final_status == "QUEUED":
        # 通知所有在线坐席刷新队列（新用户进来了）
        manager.notify_agents({"type": "queue_update"})
        logger.info(
            "转人工请求 | conv=%s user=%s reason=%s → QUEUED",
            conversation_id, current.user_id, (body.reason if body else None) or "用户主动",
        )
    return HandoffResponse(conversation_id=conversation_id, status=final_status)


# 坐席侧：队列 / 接管 / 回复 / 结束


@router.get("/agent/queued", response_model=list[QueuedConversation])
def list_queued(_: CurrentUser = Depends(get_current_agent)) -> list[QueuedConversation]:
    """
    待接管会话列表（QUEUED 状态），按排队先后排序。

    附带用户昵称和最近一条消息预览，坐席不用点进去就能判断先接谁。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.id AS conversation_id,
                   coalesce(u.nickname, u.username, '未知用户') AS user_nickname,
                   c.last_active_at AS queued_since,
                   (
                       SELECT m.content FROM messages m
                       WHERE m.conversation_id = c.id
                       ORDER BY m.id DESC LIMIT 1
                   ) AS last_message
            FROM conversations c
            JOIN users u ON u.id = c.user_id
            WHERE c.status = 'QUEUED'
            ORDER BY c.last_active_at ASC
            """
        )
        rows = cur.fetchall()

    return [
        QueuedConversation(
            conversation_id=r["conversation_id"],
            user_nickname=r["user_nickname"],
            last_message=(r["last_message"] or "")[:_QUEUE_PREVIEW_CHARS] or None,
            queued_since=r["queued_since"],
        )
        for r in rows
    ]


@router.get("/agent/active", response_model=list[ActiveConversation])
def list_active(current: CurrentUser = Depends(get_current_agent)) -> list[ActiveConversation]:
    """
    接待中会话列表（HUMAN 状态），按最近活跃时间倒序。

    为什么必须有这个接口：
        接管动作会让会话从 QUEUED 队列消失，坐席切去接待别人或刷新页面后，
        原会话就彻底"失联" —— 用户在等，坐席看不到。有了这份列表，
        HUMAN 的会话始终留在工作台上，随时能点回来继续回复。

    claimed_by_me 用于多坐席场景区分归属：本项目只有 agent001，
    字段恒为 True；但前端据此渲染归属标记，将来加坐席不用改接口。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT c.id AS conversation_id,
                   coalesce(u.nickname, u.username, '未知用户') AS user_nickname,
                   c.last_active_at,
                   (
                       SELECT m.content FROM messages m
                       WHERE m.conversation_id = c.id
                       ORDER BY m.id DESC LIMIT 1
                   ) AS last_message,
                   -- NULL 用 IS NOT DISTINCT FROM 比较，不能用 =，
                   -- 否则 claimed_by 为空的历史会话会得到 NULL 而不是 false
                   (c.claimed_by IS NOT DISTINCT FROM %s) AS claimed_by_me
            FROM conversations c
            JOIN users u ON u.id = c.user_id
            WHERE c.status = 'HUMAN'
            ORDER BY c.last_active_at DESC
            """,
            (current.user_id,),
        )
        rows = cur.fetchall()

    return [
        ActiveConversation(
            conversation_id=r["conversation_id"],
            user_nickname=r["user_nickname"],
            last_message=(r["last_message"] or "")[:_QUEUE_PREVIEW_CHARS] or None,
            last_active_at=r["last_active_at"],
            claimed_by_me=r["claimed_by_me"],
        )
        for r in rows
    ]


@router.post("/agent/conversations/{conversation_id}/accept", response_model=HandoffResponse)
def accept_conversation(
    conversation_id: int,
    current: CurrentUser = Depends(get_current_agent),
) -> HandoffResponse:
    """
    坐席接管：QUEUED → HUMAN，同时记录 claimed_by（谁在服务这个用户）。

    条件更新保证两个坐席同时点"接管"时只有一个成功（另一个收到 409），
    不需要分布式锁 —— 单条 UPDATE 的原子性就够了。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE conversations SET status = 'HUMAN', claimed_by = %s, last_active_at = now()
            WHERE id = %s AND status = 'QUEUED'
            RETURNING id, user_id, status
            """,
            (current.user_id, conversation_id),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="会话不在排队中（可能已被接管或已关闭）",
        )

    # 用户侧立刻从"转接中"切到"人工客服为您服务"
    manager.notify_user(
        row["user_id"],
        {"type": "status", "conversation_id": conversation_id, "status": "HUMAN"},
    )
    logger.info("坐席接管 | conv=%s agent=%s", conversation_id, current.username)
    return HandoffResponse(conversation_id=conversation_id, status=row["status"])


@router.post("/agent/conversations/{conversation_id}/messages", response_model=MessageItem)
def agent_reply(
    conversation_id: int,
    body: AgentMessageRequest,
    current: CurrentUser = Depends(get_current_agent),
) -> MessageItem:
    """
    坐席回复一条消息（role='human'），并实时推给用户。

    前置条件：会话必须处于 HUMAN（QUEUED 还没接管时坐席不该抢答）。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, user_id, status FROM conversations WHERE id = %s", (conversation_id,)
        )
        conv = cur.fetchone()
        if conv is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
        if conv["status"] != "HUMAN":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"会话状态为 {conv['status']}，仅 HUMAN 状态可回复",
            )

        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (%s, 'human', %s)
            RETURNING id, role, content, created_at
            """,
            (conversation_id, body.content),
        )
        msg = cur.fetchone()
        cur.execute("UPDATE conversations SET last_active_at = now() WHERE id = %s", (conversation_id,))

    # 推送 JSON 的字段名与 MessageItem 完全一致，前端用同一套渲染函数
    manager.notify_user(
        conv["user_id"],
        {
            "type": "message",
            "conversation_id": conversation_id,
            "message": {
                "id": msg["id"],
                "role": msg["role"],
                "content": msg["content"],
                "created_at": msg["created_at"].isoformat(),
            },
        },
    )
    logger.info(
        "坐席回复 | conv=%s agent=%s 长度=%d",
        conversation_id, current.username, len(body.content),
    )
    return MessageItem(**msg)


@router.post("/agent/conversations/{conversation_id}/close", response_model=HandoffResponse)
def close_conversation(
    conversation_id: int,
    body: CloseConversationRequest,
    current: CurrentUser = Depends(get_current_agent),
) -> HandoffResponse:
    """
    坐席结束服务：HUMAN → AI（回到 AI 接待）或 CLOSED（终态）。

    back='ai' 是默认路径：问题解决后顾客继续留在会话里，
    下一句"我还有个问题"由 AI 接着答 —— 历史消息都在，上下文无缝衔接
    （人工的话在 _load_history 里带 [人工客服] 前缀，AI 知道那是同事说的）。
    """
    target = "AI" if body.back == "ai" else "CLOSED"
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        # claimed_by 一并置空：会话离开人工阶段后不再属于任何坐席，
        # 否则下次被接管时"接待中"列表会短暂显示成上一位坐席的会话
        cur.execute(
            """
            UPDATE conversations SET status = %s, claimed_by = NULL, last_active_at = now()
            WHERE id = %s AND status = 'HUMAN'
            RETURNING id, user_id, status
            """,
            (target, conversation_id),
        )
        row = cur.fetchone()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="会话不在人工服务中，无法结束",
        )

    manager.notify_user(
        row["user_id"],
        {"type": "status", "conversation_id": conversation_id, "status": target},
    )
    logger.info("坐席结束服务 | conv=%s agent=%s → %s", conversation_id, current.username, target)
    return HandoffResponse(conversation_id=conversation_id, status=target)


# 内部工具


def _get_owned_conversation(conversation_id: int, user_id: int) -> dict[str, Any]:
    """
    校验会话归属（顾客侧专用）。404 策略与 api_chat 相同：不存在和不是
    你的对外表现一致，防枚举。逻辑与 api_chat._get_owned_conversation 一致，
    刻意复制而不是共享 —— 两个文件各自独立可读，抽公共"会话校验模块"
    反而增加阅读时的跳转成本。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, user_id, status FROM conversations WHERE id = %s",
            (conversation_id,),
        )
        row = cur.fetchone()

    if row is None or row["user_id"] != user_id:
        if row is not None:
            logger.warning(
                "越权访问拦截 | conv=%s 归属用户=%s 请求用户=%s",
                conversation_id, row["user_id"], user_id,
            )
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    return row
