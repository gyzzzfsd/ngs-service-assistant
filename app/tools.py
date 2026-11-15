"""
tools.py —— 模型能调用的工具（证据来源）

职责：
    定义 4 个只读工具（查委托单/样本、查检测产品、查报告变异、查知识库）
    + 给模型看的工具说明 + 统一分发入口。
    所有 SQL 都写在这个文件里 —— 因为 SQL 是工具的实现细节，跟着工具走，
    而不是单独抽一层 repository 让人来回跳。

不负责：
    决定调哪个工具（agent.py 里由模型决定）、校验工具结果是否被正确引用（harness.py）。

设计要点（整个项目最关键的三条）：
    1. 工具全部只读。没有"申请加急"这类写工具 —— 从能力层面就不给模型改库的可能；
       加急/复检/退单走四段式（建议 → 校验 → 用户确认 → 后端原子 UPDATE），见 api_actions.py。
    2. user_id 永远不在参数表里，由后端从 JWT 取出后注入 SQL。
       模型只能决定"查不查"，改不了"查谁"，越权在 SQL 条件里就被掐死。
    3. 工具失败不抛异常，返回 ToolResult(success=False, code=..., message=...)。
       把"没查到"如实告诉模型，它才不会编造；异常会把整个请求打断，模型失去纠错机会。
"""

import json
import logging
import time
from datetime import timedelta
from typing import Any

from psycopg.rows import dict_row

from app import rag
from app.config import settings
from app.db import get_conn
from app.schemas import ToolResult

logger = logging.getLogger(__name__)

# 实验节点的中文说明。给模型看的是"已接收"而不是 received，
# 否则模型容易在回答里直接蹦出英文状态码，体验很差。
# 9 档 = 主线 7 档 + 旁支 qc_failed / cancelled，与 orders 的 DB CHECK 一一对应。
ORDER_STATUS_TEXT = {
    "received": "已接收",
    "qc_passed": "质检合格",
    "qc_failed": "质检不合格",
    "library_built": "建库完成",
    "sequencing": "上机测序中",
    "analyzing": "生信分析中",
    "report_review": "报告审核中",
    "report_issued": "已发报告",
    "cancelled": "已退单",
}


# 工具实现


def get_user_orders(user_id: int) -> ToolResult:
    """
    查当前用户自己的全部检测委托单（含样本明细与流程节点）。

    Args:
        user_id: 当前登录用户 ID。**由 agent.py 从 JWT 解出后传入，不接受模型传参**。

    Returns:
        ToolResult，data 形如：
            {"orders": [{"id": 1, "status": "received", "status_text": "已接收",
                         "total_amount": "3600.00", "department": "市一医院呼吸内科",
                         "is_urgent": false, "retest_count": 0, "qc_note": null,
                         "created_at": "收样日期", "expected_report_date": "预计出报告日",
                         "items": [{"product_name": "...", "sample_no": "HP-2026-01001",
                                    "sample_type": "FFPE组织", "quantity": 1,
                                    "unit_price": "3600.00", "turnaround_days": 7}]}]}
        用户没有委托单时返回 success=False, code=ORDERS_EMPTY。

    # 难点（越权防护）：这条 SQL 的 WHERE 里写死了 user_id。
    # 工具的参数定义（见下方 TOOL_SCHEMAS）里根本没有 user_id 这一项，
    # 所以模型就算"想"查别人的委托单也无从下手 —— 它连参数都填不了。
    # 把归属校验放在 SQL 里，而不是放在后面的业务代码里，是因为 SQL 是唯一必经之路，
    # 少一处 if 判断就少一个漏判的机会。
    """
    # 为什么是 30 而不是 10？
    # 10 是按“单客户最多 7 张单”估的，数据量增长后不够用。
    # 客户单据较多时，过小的 LIMIT 会挤掉较早的单据，
    # （超窗口复检、双样本、已退单等），
    # harness 的“先查后引导”就拿不到完整证据。
    # 30 是留足余量的上界：单用户真实送检量到不了这个数量级，
    # 一次查回 30 张单的 JSON 也仍在模型上下文预算内。
    # 刻意不做成配置项 —— 多一个配置就多一处要跳的文件，收益不抵成本。
    sql = """
        SELECT id, status, total_amount, department, is_urgent, retest_count, qc_note, created_at
        FROM orders
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT 30
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (user_id,))
        orders = cur.fetchall()

        if not orders:
            return ToolResult.fail("ORDERS_EMPTY", "当前用户没有任何检测委托单")

        # 一次把所有委托单的样本查回来，避免 N+1 查询（每张单循环查一次）。
        # JOIN products 是为了拿 turnaround_days：预计出报告日 = 收样日 + 周期，
        # 这个数字要下发给模型（用户高频问"报告什么时候出"）。
        order_ids = [o["id"] for o in orders]
        cur.execute(
            """
            SELECT i.order_id, i.product_name, i.sample_no, i.sample_type,
                   i.quantity, i.unit_price, p.turnaround_days
            FROM order_items i
            JOIN products p ON p.id = i.product_id
            WHERE i.order_id = ANY(%s)
            """,
            (order_ids,),
        )
        items_by_order: dict[int, list[dict]] = {}
        for item in cur.fetchall():
            items_by_order.setdefault(item["order_id"], []).append(
                {
                    "product_name": item["product_name"],
                    "sample_no": item["sample_no"],
                    "sample_type": item["sample_type"],
                    "quantity": item["quantity"],
                    # 金额统一转成字符串下发：JSON 里的 float 会出现 3600.0000000001 这种
                    # 浮点误差，而模型和 harness 都要拿它跟"3600"这类字面量比对。
                    "unit_price": str(item["unit_price"]),
                    # 周期跟着样本下发：harness V3 的"周期比对"从这里取证据
                    "turnaround_days": item["turnaround_days"],
                }
            )

    data = [_order_payload(o, items_by_order.get(o["id"], [])) for o in orders]
    return ToolResult.ok({"orders": data}, message=f"共 {len(data)} 张委托单")


def _order_payload(order: dict, items: list[dict]) -> dict:
    """把一张委托单的行拼成下发给模型/前端的形状（与 get_user_orders 的 docstring 一致）。

    # 预计出报告日为什么取各样本周期的最大值？
    # 一张委托单理论上可能含多个产品（本项目种子里双样本单是同产品），
    # 报告要等最慢的那条线跑完才算齐 —— 取 max 与业务直觉一致，
    # 也避免"按第一件样本的周期算出更早的日期"这类事实漂移。
    """
    max_days = max((it["turnaround_days"] for it in items), default=None)
    expected = order["created_at"] + timedelta(days=max_days) if max_days else None
    return {
        "id": order["id"],
        "status": order["status"],
        "status_text": ORDER_STATUS_TEXT.get(order["status"], order["status"]),
        "total_amount": str(order["total_amount"]),
        "department": order["department"],
        "is_urgent": order["is_urgent"],
        "retest_count": order["retest_count"],
        "qc_note": order["qc_note"],
        "created_at": order["created_at"].isoformat(timespec="minutes"),
        "expected_report_date": expected.isoformat(timespec="minutes") if expected else None,
        "items": items,
    }


def search_products(query: str = "") -> ToolResult:
    """
    按关键词搜在售的检测产品（panel）。

    Args:
        query: 关键词，可为空。为空时返回价格最低的 5 款在售产品（用于"你们能做什么检测"）。

    Returns:
        ToolResult，data 形如 {"products": [{"id", "name", "category", "price",
        "turnaround_days", "description"}]}。没搜到时 success=False, code=PRODUCTS_EMPTY。
    """
    keyword = (query or "").strip()
    like = f"%{keyword}%"

    sql = """
        SELECT id, name, category, price, turnaround_days, description
        FROM products
        WHERE status = 'on_sale'
          AND (%s = '' OR name ILIKE %s OR category ILIKE %s OR description ILIKE %s)
        ORDER BY price ASC
        LIMIT 5
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (keyword, like, like, like))
        rows = cur.fetchall()

    if not rows:
        return ToolResult.fail(
            "PRODUCTS_EMPTY",
            f"没有找到与「{keyword}」相关的在售检测产品" if keyword else "当前没有在售的检测产品",
        )

    data = [
        {
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "price": str(r["price"]),
            # 周期是本域的高频事实（"多久出报告"），与价格同级下发
            "turnaround_days": r["turnaround_days"],
            "description": (r["description"] or "")[:200],
        }
        for r in rows
    ]
    return ToolResult.ok({"products": data}, message=f"匹配到 {len(data)} 个检测产品")


def get_report_variants(user_id: int) -> ToolResult:
    """
    查当前用户**已发报告**的变异明细（基因、c./p. 变异、VAF、Tier 分级、解读）。

    Args:
        user_id: 当前登录用户 ID。**由 agent.py 从 JWT 解出后传入，不接受模型传参**。

    Returns:
        ToolResult，data 形如：
            {"reports": [{"report_no": "RPT-2026-0002", "order_id": 8,
                           "issued_at": "...", "conclusion": "...",
                           "variants": [{"gene": "KRAS", "c_change": "c.34G>T",
                                         "p_change": "p.G12C", "vaf": "0.2800",
                                         "tier": 1, "interpretation": "..."}]}]}
        没有已发报告时返回 success=False, code=REPORTS_EMPTY。

    # 为什么按"已发报告"过滤而不是按用户全量报告？
    # reports 表的行天然属于某张委托单（order_id 外键），过滤条件 o.user_id=%s
    # 已经把归属掐死；再加 o.status='report_issued' 是因为"报告已生成但委托单
    # 还在报告审核"属于实验室内部状态 —— 对外说出去就是"报告已出"，
    # 会和委托单节点(报告审核中)自相矛盾，harness V3 会被迫二选一。
    """
    sql = """
        SELECT r.id, r.report_no, r.order_id, r.issued_at, r.conclusion
        FROM reports r
        JOIN orders o ON o.id = r.order_id
        WHERE o.user_id = %s AND o.status = 'report_issued'
        ORDER BY r.issued_at DESC
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, (user_id,))
        reports = cur.fetchall()

        if not reports:
            return ToolResult.fail("REPORTS_EMPTY", "当前用户没有已发报告的委托单")

        report_ids = [r["id"] for r in reports]
        cur.execute(
            """
            SELECT report_id, gene, c_change, p_change, vaf, tier, interpretation
            FROM report_variants
            WHERE report_id = ANY(%s)
            ORDER BY tier, gene
            """,
            (report_ids,),
        )
        variants_by_report: dict[int, list[dict]] = {}
        for v in cur.fetchall():
            variants_by_report.setdefault(v["report_id"], []).append(
                {
                    "gene": v["gene"],
                    "c_change": v["c_change"],
                    "p_change": v["p_change"],
                    # VAF 用字符串下发（同金额的理由：浮点误差 + 比对口径统一）
                    "vaf": str(v["vaf"]),
                    "tier": v["tier"],
                    "interpretation": v["interpretation"],
                }
            )

    data = [
        {
            "report_no": r["report_no"],
            "order_id": r["order_id"],
            "issued_at": r["issued_at"].isoformat(timespec="minutes"),
            "conclusion": r["conclusion"],
            "variants": variants_by_report.get(r["id"], []),
        }
        for r in reports
    ]
    # 阴性报告（variants 为空列表）是合法结果不是失败："未检出"是有效事实
    return ToolResult.ok({"reports": data}, message=f"共 {len(data)} 份已发报告")


def search_knowledge(query: str) -> ToolResult:
    """
    在送检知识库里检索与问题最相关的段落（RAG：SOP/产品说明/报告解读指南/政策）。

    Args:
        query: 用户问题的关键内容。

    Returns:
        ToolResult，data 形如 {"chunks": [{"content": "...", "title": "...", "score": 0.82}]}。
        三种失败各走各的码：
            KNOWLEDGE_NOT_FOUND  检索正常但无相关内容（空召回是业务结果不是错误）
            KNOWLEDGE_ERROR      检索本身失败（模型没加载、数据库异常）
            INVALID_ARGUMENTS    query 为空

    # 阶段③ 起本工具已接真实向量检索（rag.search_chunks）。
    # 混合检索（稠密+稀疏 RRF）在 RAG 升级步骤落地，本工具的签名与失败码不变。
    # "没查到就如实说"的行为约定从未变过 —— 这是防幻觉的关键一环。
    """
    if not (query or "").strip():
        return ToolResult.fail("INVALID_ARGUMENTS", "检索内容不能为空")

    try:
        hits = rag.search_chunks(query)
    except Exception as exc:  # noqa: BLE001 —— 模型未配置/加载失败也不打断问答
        # exc_info=True 把堆栈打进日志：异常类型进了给模型的 message，
        # 但定位"为什么加载失败"需要堆栈（只有日志里有）
        logger.warning("知识库检索异常 | query=%s", query, exc_info=True)
        return ToolResult.fail(
            "KNOWLEDGE_ERROR",
            # 带上原始异常信息（但不带堆栈）：模型不需要，日志里 exception 会补全
            f"知识库检索失败（{type(exc).__name__}: {exc}），请如实告知用户暂时查不到，建议转人工",
        )

    if not hits:
        return ToolResult.fail(
            "KNOWLEDGE_NOT_FOUND",
            "知识库中未找到与问题相关的内容，请如实告知用户，不要编造政策或流程内容，建议转人工",
        )
    return ToolResult.ok({"chunks": hits})


# 给模型看的工具定义（OpenAI function calling 格式）

# description 直接决定模型会不会在正确的时机调这个工具，写得越具体越好。
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_user_orders",
            "description": (
                "查询当前登录用户自己的检测委托单与样本列表（含流程节点、样本编号、"
                "收样日期、预计出报告日、加急/复检标记、金额明细）。"
                "用户问『我的样本到哪一步了』『报告出了吗』『什么时候出结果』"
                "『要加急/复检/退单』时调用。"
                "注意：该工具不需要任何参数，系统会自动按当前登录用户过滤，"
                "你无法也无需指定用户。"
            ),
            # parameters 是空对象，且没有任何 user_id 参数。
            # 这是"物理隔离"而不是"约定"：模型即便幻觉出 user_id，也没有地方可填。
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "按关键词搜索本实验室在售的检测产品（panel），返回名称、适用癌种、"
                "价格、检测周期（工作日）、简介。用户问『有没有 XX 检测』"
                "『大 panel 多少钱』『520 基因覆盖什么』时调用。"
                "关键词可以为空，此时返回几款在售产品。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检测产品关键词，例如『肺癌』『520』『MRD』『遗传』；不确定就留空",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_report_variants",
            "description": (
                "查询当前登录用户已发报告的变异明细：基因、c./p. 变异位点、"
                "VAF、临床分级（Tier）、报告内解读原文。"
                "用户问『报告里有什么变异』『EGFR 是什么分级』『HER2 阳不阳』"
                "『这个位点什么意思』时调用。"
                "注意：该工具不需要任何参数，系统会自动按当前登录用户过滤。"
            ),
            # 同 get_user_orders：参数表里没有 user_id，归属在 SQL 里掐死
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在送检知识库里检索 SOP、产品说明、报告解读指南与政策原文，"
                "返回最相关的段落。用户问『样本怎么采集』『多久能出报告』"
                "『加急政策』『质检不合格怎么办』『Tier 是什么意思』这类"
                "文档里的问题时调用。问委托单进度不要用它，用 get_user_orders；"
                "问报告里的变异不要用它，用 get_report_variants。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要在知识库里检索的问题要点，尽量保留用户原话里的产品名与术语",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# 统一分发入口

# 工具名 → 实现函数。加新工具只需在这里加一行 + 上面写一个函数。
_TOOL_FUNCS = {
    "get_user_orders": lambda args, user_id: get_user_orders(user_id),
    "search_products": lambda args, user_id: search_products(args.get("query", "")),
    "get_report_variants": lambda args, user_id: get_report_variants(user_id),
    "search_knowledge": lambda args, user_id: search_knowledge(args.get("query", "")),
}


def execute_tool(name: str, arguments: str, user_id: int) -> ToolResult:
    """
    执行一个工具调用，**任何情况下都返回 ToolResult，绝不抛异常**。

    Args:
        name: 模型给出的工具名。
        arguments: 模型给出的参数 JSON 字符串（可能是残缺/非法的 JSON）。
        user_id: 当前登录用户，由 agent 从 JWT 注入，用于覆盖模型可能乱传的值。

    Returns:
        ToolResult。失败分为三类：
            INVALID_ARGUMENTS  参数不是合法 JSON
            UNKNOWN_TOOL       模型编了一个不存在的工具名
            TOOL_ERROR         工具内部异常（SQL 报错、连不上库等）

    本函数只做两件事：计时 + 打日志，具体的参数解析和分发在 _dispatch 里。
    这样"每次工具调用都一定有日志"是结构上保证的，而不是靠每个分支记得写。

    # 为什么这里要 catch 住所有异常？
    # 因为工具是"给模型看的"，不是"给人看的"。
    # 一旦异常冒泡到 FastAPI，用户看到 500，模型也被剥夺了自我纠正的机会：
    # 它永远不知道刚才那次查询失败了，下一轮就会用编造的数据回答。
    # 把异常转成 success=False 的结构化结果回填给模型，它才有机会说"查询失败，请稍后重试"。
    """
    started = time.perf_counter()
    result = _dispatch(name, arguments, user_id)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # 成功用 INFO，失败用 WARNING：日志里扫一眼就知道哪些工具经常查不到东西，
    # 这是后续判断"是不是该补数据/改工具描述"的依据。
    log = logger.info if result.success else logger.warning
    log(
        "工具调用 | name=%-20s user=%s 耗时=%5.0fms success=%-5s code=%-20s args=%s",
        name,
        user_id,
        elapsed_ms,
        result.success,
        result.code,
        _brief(arguments),
    )
    return result


def _dispatch(name: str, arguments: str, user_id: int) -> ToolResult:
    """
    解析参数并按工具名分发。所有失败都转成 ToolResult，不抛异常。

    Args:
        name: 工具名。
        arguments: 参数 JSON 字符串。
        user_id: 当前用户 ID。

    Returns:
        ToolResult。
    """
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return ToolResult.fail("INVALID_ARGUMENTS", f"参数不是合法 JSON：{arguments[:100]}")

    if not isinstance(parsed, dict):
        return ToolResult.fail("INVALID_ARGUMENTS", "参数必须是一个 JSON 对象")

    func = _TOOL_FUNCS.get(name)
    if func is None:
        return ToolResult.fail("UNKNOWN_TOOL", f"不存在名为 {name} 的工具")

    try:
        return func(parsed, user_id)
    except Exception as exc:  # noqa: BLE001 —— 这里就是要兜住所有异常
        # 异常堆栈要完整打出来（exc_info），否则"TOOL_ERROR"回到模型那里只是一句话，
        # 排查时完全不知道是 SQL 写错了还是连接断了。
        logger.exception("工具 %s 执行异常", name)
        return ToolResult.fail("TOOL_ERROR", f"工具执行失败：{type(exc).__name__}: {exc}")


def _brief(arguments: str, limit: int = 80) -> str:
    """把参数压成一行短文本用于日志，避免一次搜索把整段 JSON 打进日志。"""
    text = (arguments or "").replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "..."
