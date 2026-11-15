"""
test_tools.py —— 工具层测试

重点不是"SQL 能跑"，而是三件事：
    1. 数据隔离：陈医生看到的委托单里不能有林教授的
    2. 失败不抛异常：查不到东西时返回结构化失败，而不是 500
    3. 日志真的在打
另加本域特有断言：预计出报告日、阴性报告空变异仍是成功、暂停接样过滤。
"""

import logging

from app.tools import execute_tool, get_report_variants, get_user_orders, search_products


# get_user_orders


def test_chen_has_seven_orders():
    """陈医生（user_id=1）在种子数据里有 7 张委托单，节点覆盖见断言。"""
    result = get_user_orders(1)

    assert result.success is True
    orders = {o["id"]: o for o in result.data["orders"]}
    assert set(orders) == {1, 2, 3, 6, 7, 8, 11}

    assert orders[1]["status"] == "received"
    assert orders[1]["status_text"] == "已接收"
    assert orders[2]["status"] == "report_issued"
    assert orders[3]["status"] == "cancelled"
    assert orders[6]["status"] == "analyzing"
    assert orders[7]["status"] == "report_review"
    assert orders[11]["status"] == "qc_passed"


def test_order_amount_is_string_not_float():
    """
    金额必须以字符串返回。

    # 如果用 float，3600 会变成 3600.0，多个样本相加会出现
    # 3600.0000000000005。harness 后面要拿它跟回复里的"3600"做
    # Decimal 比对，浮点误差会让比对莫名失败。
    """
    result = get_user_orders(1)
    order1 = next(o for o in result.data["orders"] if o["id"] == 1)

    assert order1["total_amount"] == "3600.00"
    assert isinstance(order1["total_amount"], str)


def test_order_payload_has_domain_fields():
    """委托单负载必须带本域关键事实：样本号/类型/周期/预计出报告日/加急标记。"""
    result = get_user_orders(1)
    order1 = next(o for o in result.data["orders"] if o["id"] == 1)

    item = order1["items"][0]
    assert item["sample_no"] == "HP-2026-01001"
    assert item["sample_type"] == "FFPE组织"
    assert item["turnaround_days"] == 7
    assert item["unit_price"] == "3600.00"
    assert order1["is_urgent"] is False
    assert order1["retest_count"] == 0
    # 预计出报告日 = 收样日 + 周期（模型回答"什么时候出报告"的事实来源）
    assert order1["expected_report_date"] is not None


def test_qc_failed_order_carries_qc_note():
    """王医生的 qc_failed 单要带质检说明（模型要说得出复检理由）。"""
    result = get_user_orders(3)
    order5 = next(o for o in result.data["orders"] if o["id"] == 5)

    assert order5["status"] == "qc_failed"
    assert order5["qc_note"] is not None
    assert "DNA" in order5["qc_note"]


def test_double_sample_order_attaches_both_items():
    """委托单 8 双样本（FFPE + 胸水）：两条明细都挂在自己身上，不串单。"""
    result = get_user_orders(1)
    order8 = next(o for o in result.data["orders"] if o["id"] == 8)

    sample_nos = {i["sample_no"] for i in order8["items"]}
    assert sample_nos == {"HP-2026-00890", "HP-2026-00891"}
    assert order8["total_amount"] == "7200.00"  # 3600 × 2


def test_user_data_isolation():
    """
    最关键的用例：林教授只能看到自己的委托单 4 / 10，看不到陈医生的 1。

    # 这条断言是防"越权"的最后一道门。
    # 工具内部 SQL 写死了 WHERE user_id = %s，模型没有任何参数可以改它。
    # 一旦有人"优化"成 WHERE user_id = %(user_id)s 并从参数里取，
    # 模型就能通过编造参数查别人的委托单 —— 这条用例会立刻变红。
    """
    result = get_user_orders(2)

    assert result.success is True
    assert {o["id"] for o in result.data["orders"]} == {4, 10}
    assert all(o["id"] != 1 for o in result.data["orders"])


def test_no_orders_returns_structured_failure():
    """没有委托单的用户（赵同学）应返回 ORDERS_EMPTY，而不是抛异常或返回 500。"""
    result = get_user_orders(4)

    assert result.success is False
    assert result.code == "ORDERS_EMPTY"
    assert "没有" in result.message


# search_products


def test_search_products_hit():
    """
    搜"panel"命中含 panel 关键词的在售产品，且带价格与周期。

    # ILIKE 匹配 name/category/description 三字段 —— 肺癌11基因与
    # MRD 的简介里也写了"panel"，所以命中 3 款；而暂停接样的 648 超 panel
    # 不在其中，恰好证明 status='on_sale' 过滤真的生效。
    """
    result = search_products("panel")

    assert result.success is True
    products = result.data["products"]
    assert len(products) == 3
    assert "实体瘤520基因大panel" in {p["name"] for p in products}
    assert all(p["price"] for p in products)
    assert all(p["turnaround_days"] for p in products)


def test_search_products_empty_query_returns_cheapest():
    """不传关键词时返回最便宜的几款，且按价格升序 —— 前端"猜你喜欢"依赖这个顺序。"""
    result = search_products("")

    assert result.success is True
    prices = [float(p["price"]) for p in result.data["products"]]
    assert len(prices) == 5
    assert prices == sorted(prices)


def test_search_products_miss_returns_structured_failure():
    result = search_products("质谱")

    assert result.success is False
    assert result.code == "PRODUCTS_EMPTY"


def test_paused_products_excluded():
    """
    暂停接样的产品不应该出现在搜索结果里（SQL 里有 status='on_sale' 过滤）。

    # 种子里的消化道 26 基因 panel 只在 description 提到"消化道"，
    # 搜"消化道"命中它的条件与 status='paused' 冲突 —— 用它当探针，
    # 过滤被谁悄悄去掉时这条用例立刻变红。
    """
    result = search_products("消化道")

    assert result.success is False
    assert result.code == "PRODUCTS_EMPTY"


# get_report_variants


def test_get_report_variants_for_chen():
    """陈医生有两份已发报告：RPT-0001（6 变异）+ RPT-0002（4 变异）。"""
    result = get_report_variants(1)

    assert result.success is True
    reports = {r["report_no"]: r for r in result.data["reports"]}
    assert set(reports) == {"RPT-2026-0001", "RPT-2026-0002"}

    rep1 = reports["RPT-2026-0001"]
    assert {v["gene"] for v in rep1["variants"]} == {
        "EGFR", "KRAS", "PIK3CA", "TP53", "KEAP1", "RBM10",
    }
    l858 = next(v for v in rep1["variants"] if v["p_change"] == "p.L858R")
    assert l858["tier"] == 1
    assert l858["vaf"] == "0.4200"
    assert l858["c_change"] == "c.2573T>C"


def test_negative_report_returns_empty_variants_as_success():
    """
    林教授的 RPT-2026-0003 是阴性报告（未检出）→ 变异列表为空但仍是成功。

    # 「未检出」是有效事实不是失败 —— 如果这里返回 success=False，
    # 模型会把"没检出"说成"查询失败"，阴性报告被系统性地错误转述。
    """
    result = get_report_variants(2)

    assert result.success is True
    rep3 = next(r for r in result.data["reports"] if r["report_no"] == "RPT-2026-0003")
    assert rep3["variants"] == []
    assert "未检出" in rep3["conclusion"]


def test_get_report_variants_empty_for_user_without_reports():
    """赵同学没有已发报告 → REPORTS_EMPTY（结构化失败）。"""
    result = get_report_variants(4)

    assert result.success is False
    assert result.code == "REPORTS_EMPTY"


# execute_tool 分发与容错


def test_unknown_tool_returns_code():
    """模型编了一个不存在的工具名，必须返回 UNKNOWN_TOOL 而不是崩。"""
    result = execute_tool("cancel_order", "{}", 1)

    assert result.success is False
    assert result.code == "UNKNOWN_TOOL"


def test_invalid_json_arguments():
    """模型给出的参数经常是半截 JSON，必须被兜住。"""
    result = execute_tool("search_products", '{"query": "肺癌', 1)

    assert result.success is False
    assert result.code == "INVALID_ARGUMENTS"


def test_arguments_must_be_object():
    result = execute_tool("search_products", "[1,2,3]", 1)

    assert result.success is False
    assert result.code == "INVALID_ARGUMENTS"


def test_knowledge_tool_with_hits(monkeypatch):
    """知识检索命中：rag 返回段落 → 工具成功，data.chunks 原样透传。"""
    from app import tools

    monkeypatch.setattr(
        tools.rag,
        "search_chunks",
        lambda q, top_k=None: [
            {"title": "样本采集与送检要求", "content": "FFPE 切片不少于 10 张", "score": 0.82},
        ],
    )
    result = execute_tool("search_knowledge", '{"query": "样本怎么送"}', 1)

    assert result.success is True
    assert result.data["chunks"][0]["title"] == "样本采集与送检要求"


def test_knowledge_tool_empty_recall(monkeypatch):
    """空召回是业务结果不是错误：模型收到 KNOWLEDGE_NOT_FOUND 后应如实告知。"""
    from app import tools

    monkeypatch.setattr(tools.rag, "search_chunks", lambda q, top_k=None: [])
    result = execute_tool("search_knowledge", '{"query": "量子力学"}', 1)

    assert result.success is False
    assert result.code == "KNOWLEDGE_NOT_FOUND"


def test_knowledge_tool_error_becomes_tool_fail(monkeypatch):
    """
    检索炸了（模型没加载/数据库异常）→ KNOWLEDGE_ERROR，绝不向上抛异常。

    # 这里必须兜住。如果让异常冒泡，FastAPI 返回 500，
    # 模型失去"知道检索失败"的机会，下一轮就会用编造内容作答 ——
    # 和其他工具的失败处理是同一条纪律。
    """
    from app import tools

    def _boom(q, top_k=None):
        raise RuntimeError("bge-m3 加载失败")

    monkeypatch.setattr(tools.rag, "search_chunks", _boom)
    result = execute_tool("search_knowledge", '{"query": "样本怎么送"}', 1)

    assert result.success is False
    assert result.code == "KNOWLEDGE_ERROR"
    assert "bge-m3" in result.message


def test_knowledge_tool_blank_query():
    """空检索词不触发任何检索（INVALID_ARGUMENTS）。"""
    result = execute_tool("search_knowledge", '{"query": "   "}', 1)

    assert result.success is False
    assert result.code == "INVALID_ARGUMENTS"


def test_dispatch_injects_user_id_not_from_arguments():
    """
    就算模型在参数里伪造一个 user_id，也只能查到它自己（或查不到）。

    # 这是"参数层面的越权防护"。
    # get_user_orders 的实现根本不读 args，只用调用方传的 user_id。
    # 所以模型写 {"user_id": 2} 完全无效 —— 它查到的还是调用方（陈医生）的委托单。
    """
    result = execute_tool("get_user_orders", '{"user_id": 2}', user_id=1)

    assert result.success is True
    ids = {o["id"] for o in result.data["orders"]}
    assert ids == {1, 2, 3, 6, 7, 8, 11}, "模型通过参数伪造 user_id 越权成功了"


def test_tool_call_writes_log(caplog):
    """日志要求：每次工具调用必须留痕（工具名、用户、耗时、结果码）。"""
    caplog.set_level(logging.INFO, logger="app.tools")

    execute_tool("get_user_orders", "{}", 1)

    messages = [r.getMessage() for r in caplog.records]
    assert any("工具调用" in m and "get_user_orders" in m for m in messages), messages
