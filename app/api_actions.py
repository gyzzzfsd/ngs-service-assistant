"""
api_actions.py —— 写操作的确认执行（四段式的最后一段）

职责：
    把"AI 建议 + 用户点击确认"变成一次真实的数据库写操作。共三个写操作：
        POST /api/orders/{order_id}/urgent   申请加急（未上机 → is_urgent=true）
        POST /api/orders/{order_id}/retest   申请复检（质检不合格 / 发报告 7 日内 → 计数+1）
        POST /api/orders/{order_id}/cancel   申请退单（未建库 → status='cancelled'）

四段式回顾（为什么这个接口是最后一段）：
    ① AI 只出建议      ACTION_GUIDE + action=APPLY_*（模型没有执行权）
    ② harness 校验建议 V4：委托单必须被本轮工具查过、节点必须允许该操作
    ③ 用户前端确认     页面上出现确认按钮，点了才发请求
    ④ 后端复核执行     本文件：UPDATE 的 WHERE 里再验一次归属和状态

# ②已经校验过状态，④为什么还要再验一次？
# 因为②校验的是"AI 说话那一刻"的世界，④执行的是"用户点按钮那一刻"的世界。
# 两次之间可能隔了几十秒，实验室并发推进、另一个浏览器 tab、甚至攻击者直接
# curl 这个接口，都可能让世界变了样。防线必须在执行现场再设一道 ——
# UPDATE ... WHERE id=? AND user_id=? AND <状态条件>
# 三个条件写进同一条语句，靠数据库的原子性保证"检查和修改之间没有缝隙"。
# 这也是为什么不先 SELECT 检查再 UPDATE：两步之间同样有缝隙。

# 为什么加急/复检改标志位，退单改节点状态？
# 加急、复检都不能让样本"倒退"回早期节点，否则会人为制造
# "上一轮说已发报告、这轮又显示已接收"的事实漂移（harness V3 会被迫二选一）；
# 退单例外 —— cancelled 是终止态，节点只前进到终局，不允许再发起其他动作。

不负责：
    AI 怎么给出建议（agent.py）、建议是否可信（harness.py）、前端按钮（web/chat.html）。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.rows import dict_row

from app.db import get_conn
from app.schemas import OrderActionRequest, OrderActionResponse
from app.security import CurrentUser, get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/orders", tags=["写操作"])


@router.post("/{order_id}/urgent", response_model=OrderActionResponse)
def apply_urgent(
    order_id: int,
    body: OrderActionRequest | None = None,
    current: CurrentUser = Depends(get_current_user),
) -> OrderActionResponse:
    """
    用户确认后为一张自己的未上机委托单登记加急。

    Args:
        order_id: 目标委托单（orders.id，路径参数）。
        body: 可选的 conversation_id，用于把确认消息写进聊天记录。
        current: 当前登录用户（user_id 从 JWT 来，请求体里不接受）。

    Returns:
        OrderActionResponse。业务失败（节点不允许/已加急）也是 200；
        404 只留给"委托单不存在或不是你的"（响应对外统一 404，防枚举）。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE orders
            SET is_urgent = true
            WHERE id = %s AND user_id = %s
              AND status IN ('received', 'qc_passed', 'library_built')
              AND is_urgent = false
            RETURNING id, total_amount, is_urgent
            """,
            (order_id, current.user_id),
        )
        row = cur.fetchone()

    if row is None:
        mine = _load_my_order(order_id, current.user_id)
        if mine is None:
            _raise_not_found(order_id, current.user_id)
        reason = _urgent_fail_reason(mine)
        _log_rejected("urgent", order_id, current.user_id, mine, reason)
        return OrderActionResponse(order_id=order_id, success=False, message=reason)

    confirm = (
        f"已为委托单 {order_id} 登记加急，加急样本的检测周期缩短至 3 个工作日"
        f"（自收样日起算），实验室会优先安排实验。"
    )
    _append_confirm_message(body, current, confirm)
    logger.info("委托单已加急 | order=%s user=%s conv=%s", order_id, current.user_id, _conv(body))
    return OrderActionResponse(order_id=order_id, success=True, message=confirm)


def _urgent_fail_reason(mine: dict) -> str:
    """加急失败的具体原因（先查后说，用户需要知道为什么不能加急）。"""
    if mine["is_urgent"]:
        return "该委托单已申请过加急，无需重复提交"
    return (
        f"委托单当前节点为「{mine['status_text']}」，已进入上机及后续流程，"
        "无法再加急；预计出报告时间以检测周期为准，请耐心等待。"
    )


@router.post("/{order_id}/retest", response_model=OrderActionResponse)
def apply_retest(
    order_id: int,
    body: OrderActionRequest | None = None,
    current: CurrentUser = Depends(get_current_user),
) -> OrderActionResponse:
    """
    用户确认后为一张自己的委托单登记复检。

    放行条件：质检不合格（qc_failed），或报告已签发且在 7 日内。
    落库动作：retest_count + 1（不改节点，防止"已发报告又变回早期节点"的漂移）。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE orders
            SET retest_count = retest_count + 1
            WHERE id = %s AND user_id = %s
              AND (status = 'qc_failed'
                   OR (status = 'report_issued'
                       AND EXISTS (SELECT 1 FROM reports r
                                   WHERE r.order_id = orders.id
                                     AND r.issued_at >= now() - interval '7 day')))
            RETURNING id, retest_count
            """,
            (order_id, current.user_id),
        )
        row = cur.fetchone()

    if row is None:
        mine, last_report = _load_my_order_with_report(order_id, current.user_id)
        if mine is None:
            _raise_not_found(order_id, current.user_id)
        reason = _retest_fail_reason(mine, last_report)
        _log_rejected("retest", order_id, current.user_id, mine, reason)
        return OrderActionResponse(order_id=order_id, success=False, message=reason)

    count = row["retest_count"]
    confirm = (
        f"已为委托单 {order_id} 登记复检（累计第 {count} 次），"
        "实验室将重新安排检测，结果出来后会第一时间同步您。"
    )
    _append_confirm_message(body, current, confirm)
    logger.info("委托单已登记复检 | order=%s user=%s 第%s次 conv=%s",
                order_id, current.user_id, count, _conv(body))
    return OrderActionResponse(order_id=order_id, success=True, message=confirm)


def _retest_fail_reason(mine: dict, last_report: dict | None) -> str:
    """复检失败的具体原因（7 日窗口是否过期要结合报告签发时间判断）。"""
    if mine["status"] == "report_issued" and last_report is not None:
        return "报告签发已超过 7 日的免费复检窗口，如需复检请转人工协商处理。"
    return (
        f"委托单当前节点为「{mine['status_text']}」，不支持申请复检；"
        "质检不合格的样本、或报告签发 7 日内才可申请复检。"
    )


@router.post("/{order_id}/cancel", response_model=OrderActionResponse)
def apply_cancel(
    order_id: int,
    body: OrderActionRequest | None = None,
    current: CurrentUser = Depends(get_current_user),
) -> OrderActionResponse:
    """
    用户确认后退回一张自己的未建库委托单。

    落库动作：status='cancelled'（终止态，允许改节点 —— 与加急/复检的
    标志位策略不同，理由见模块 docstring 的第二个难点）。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            UPDATE orders
            SET status = 'cancelled'
            WHERE id = %s AND user_id = %s
              AND status IN ('received', 'qc_passed', 'qc_failed')
            RETURNING id, total_amount
            """,
            (order_id, current.user_id),
        )
        row = cur.fetchone()

    if row is None:
        mine = _load_my_order(order_id, current.user_id)
        if mine is None:
            _raise_not_found(order_id, current.user_id)
        reason = _cancel_fail_reason(mine)
        _log_rejected("cancel", order_id, current.user_id, mine, reason)
        return OrderActionResponse(order_id=order_id, success=False, message=reason)

    confirm = (
        f"已为您退回委托单 {order_id}（合计 {row['total_amount']} 元），"
        "相关费用将按退单政策处理，请注意查收通知。"
    )
    _append_confirm_message(body, current, confirm)
    logger.info("委托单已退单 | order=%s user=%s 金额=%s conv=%s",
                order_id, current.user_id, row["total_amount"], _conv(body))
    return OrderActionResponse(order_id=order_id, success=True, message=confirm)


def _cancel_fail_reason(mine: dict) -> str:
    """退单失败的具体原因（建库后只能人工协商费用，不在线退）。"""
    if mine["status"] == "cancelled":
        return "该委托单已是退单状态，请勿重复提交。"
    return (
        f"委托单当前节点为「{mine['status_text']}」，样本已进入实验流程，"
        "无法在线退单；涉及费用的问题请转人工协商处理。"
    )


# 共用小工具（同属"写操作执行"这个功能单元，不外拆文件）


def _load_my_order(order_id: int, user_id: int) -> dict | None:
    """查"我的这张委托单"当前状态。只用于失败后解释原因；查不到返回 None（404 口径）。"""
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT id, status, is_urgent, total_amount,
                   CASE status WHEN 'received' THEN '已接收'
                       WHEN 'qc_passed' THEN '质检合格'
                       WHEN 'qc_failed' THEN '质检不合格'
                       WHEN 'library_built' THEN '建库完成'
                       WHEN 'sequencing' THEN '上机测序中'
                       WHEN 'analyzing' THEN '生信分析中'
                       WHEN 'report_review' THEN '报告审核中'
                       WHEN 'report_issued' THEN '已发报告'
                       WHEN 'cancelled' THEN '已退单' END AS status_text
            FROM orders WHERE id = %s AND user_id = %s
            """,
            (order_id, user_id),
        )
        return cur.fetchone()


def _load_my_order_with_report(order_id: int, user_id: int) -> tuple[dict | None, dict | None]:
    """查"我的这张委托单" + 最近一份报告的签发时间（复检失败原因需要判断 7 日窗口）。"""
    mine = _load_my_order(order_id, user_id)
    if mine is None:
        return None, None
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT report_no, issued_at FROM reports
            WHERE order_id = %s ORDER BY issued_at DESC LIMIT 1
            """,
            (order_id,),
        )
        return mine, cur.fetchone()


def _append_confirm_message(
    body: OrderActionRequest | None,
    current: CurrentUser,
    content: str,
) -> None:
    """
    把操作成功的确认消息写进会话（写操作第 ④ 段的用户可见落点）。

    # 这条消息为什么不需要过 harness？
    # 因为它由后端代码在 UPDATE 成功之后生成，本身就是对刚发生事实的陈述 ——
    # 事实的来源就是这条语句，不存在"模型编造"环节。role 用 'ai'：
    # 对用户来说"客服侧说的话"，对后续模型历史也成立（委托单确实加急了）。
    """
    conv_id = body.conversation_id if body else None
    if not conv_id:
        return
    with get_conn() as conn, conn.cursor() as cur:
        # 归属校验：确认消息只能写进"自己的"会话，防止拿别人的 conversation_id 乱写
        cur.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            SELECT %s, 'ai', %s
            WHERE EXISTS (SELECT 1 FROM conversations WHERE id = %s AND user_id = %s)
            """,
            (conv_id, content, conv_id, current.user_id),
        )


def _raise_not_found(order_id: int, user_id: int) -> None:
    """统一 404：查不到或不属于本人一律"委托单不存在"，不给枚举别人单号的机会。"""
    logger.warning("写操作越权/不存在拦截 | order=%s user=%s", order_id, user_id)
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="委托单不存在")


def _log_rejected(action: str, order_id: int, user_id: int, mine: dict, reason: str) -> None:
    """业务失败（节点不允许）统一打 INFO：这是用户可见的正常业务结果，不是错误。"""
    logger.info(
        "写操作被节点条件拦截 | action=%s order=%s user=%s status=%s 原因=%s",
        action, order_id, user_id, mine["status"], reason,
    )


def _conv(body: OrderActionRequest | None) -> int | None:
    """取请求体里的 conversation_id（可能为 None），仅用于日志。"""
    return body.conversation_id if body else None
