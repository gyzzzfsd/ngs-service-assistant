"""
test_api_actions.py —— 写操作确认接口的测试（加急 / 复检 / 退单）

覆盖四段式最后一段的全部防线：
    正常落库、节点条件拦截、越权 404、不存在 404、确认消息写入、
    重复提交的幂等行为。
委托单状态在 SAVEPOINT 里改，用例结束自动回滚，种子数据不受影响。

种子的时效性前提（seed_ngs.sql 用 now()-interval 保证永远成立）：
    委托单8 的报告签发在 2 天前 → 复检窗口内（放行演示）；
    委托单2 的报告签发在 30 天前 → 超出 7 日窗口（拦截演示）。
"""

def test_urgent_received_order_success(client, auth_headers, conversation_id, raw_conn):
    """陈医生加急已接收的委托单1：成功，标志位落库，确认消息写进会话。"""
    resp = client.post(
        "/api/orders/1/urgent",
        headers=auth_headers,
        json={"conversation_id": conversation_id},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "3 个工作日" in data["message"]      # 加急政策周期随确认消息下发

    with raw_conn.cursor() as cur:
        cur.execute("SELECT is_urgent, status FROM orders WHERE id = 1")
        is_urgent, status = cur.fetchone()
        assert is_urgent is True
        assert status == "received"             # 加急不改节点，防事实漂移
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        role, content = cur.fetchone()
        assert role == "ai"
        assert "已为委托单 1 登记加急" in content


def test_urgent_after_sequencing_blocked(client, auth_headers):
    """委托单6 已进入生信分析（上机后）：业务失败（200 + success=False），给用户可读的原因。"""
    resp = client.post("/api/orders/6/urgent", headers=auth_headers, json={})

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "无法再加急" in data["message"] or "上机" in data["message"]


def test_urgent_already_applied_is_idempotent(client, auth_headers, conversation_id, raw_conn):
    """
    连点两次加急按钮：第二次是干净的业务失败（已加急），不是 500。

    # 标志位写操作天然幂等 —— is_urgent=false 条件写进 UPDATE 的
    # WHERE，第二次执行时条件不再成立，落库数 0 自动转成"已申请过"话术。
    """
    first = client.post(
        "/api/orders/1/urgent", headers=auth_headers, json={"conversation_id": conversation_id}
    )
    second = client.post("/api/orders/1/urgent", headers=auth_headers, json={})

    assert first.json()["success"] is True
    assert second.status_code == 200
    assert second.json()["success"] is False
    assert "已申请过加急" in second.json()["message"]


def test_retest_qc_failed_order_success(client, wang_headers, conversation_id, raw_conn):
    """王医生复检质检不合格的委托单5：无条件放行，计数落库。"""
    resp = client.post(
        "/api/orders/5/retest",
        headers=wang_headers,
        json={"conversation_id": conversation_id},
    )

    assert resp.status_code == 200
    assert resp.json()["success"] is True
    assert "复检" in resp.json()["message"]

    with raw_conn.cursor() as cur:
        cur.execute("SELECT retest_count, status FROM orders WHERE id = 5")
        retest_count, status = cur.fetchone()
        assert retest_count == 1
        assert status == "qc_failed"            # 复检不改节点，防事实漂移


def test_retest_within_window_success(client, auth_headers):
    """委托单8 的报告签发在 2 天前（7 日窗口内）：复检放行。"""
    resp = client.post("/api/orders/8/retest", headers=auth_headers, json={})

    assert resp.status_code == 200
    assert resp.json()["success"] is True


def test_retest_out_of_window_blocked(client, auth_headers):
    """委托单2 的报告签发在 30 天前：超出 7 日免费复检窗口，业务失败。"""
    resp = client.post("/api/orders/2/retest", headers=auth_headers, json={})

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "7" in data["message"]


def test_retest_wrong_status_blocked(client, auth_headers):
    """委托单1 还在已接收（没质检结果也没报告）：不支持复检。"""
    resp = client.post("/api/orders/1/retest", headers=auth_headers, json={})

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "不支持申请复检" in data["message"]


def test_cancel_qc_passed_order_success(client, auth_headers, conversation_id, raw_conn):
    """陈医生退回质检合格的委托单11：成功，状态落 cancelled，确认消息写进会话。"""
    resp = client.post(
        "/api/orders/11/cancel",
        headers=auth_headers,
        json={"conversation_id": conversation_id},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "5200" in data["message"]            # 退单金额随确认消息下发

    with raw_conn.cursor() as cur:
        cur.execute("SELECT status FROM orders WHERE id = 11")
        assert cur.fetchone()[0] == "cancelled"  # 退单是终止态，允许改节点
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id DESC LIMIT 1",
            (conversation_id,),
        )
        role, content = cur.fetchone()
        assert role == "ai"
        assert "已为您退回委托单 11" in content


def test_cancel_after_library_built_blocked(client, auth_headers):
    """委托单6 已建库上机：样本进入实验流程，无法在线退单。"""
    resp = client.post("/api/orders/6/cancel", headers=auth_headers, json={})

    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert "无法在线退单" in data["message"] or "实验流程" in data["message"]


def test_others_order_returns_404(client, auth_headers):
    """
    委托单4 属于林教授：陈医生来操作 → 404。

    # 断言的是 404 不是 403 —— 与会话越权同理，
    # 403 会暴露"这个单存在但不属于你"，可被用来枚举单号。
    三个写操作共用同一道防线，各打一发。
    """
    assert client.post("/api/orders/4/urgent", headers=auth_headers, json={}).status_code == 404
    assert client.post("/api/orders/4/retest", headers=auth_headers, json={}).status_code == 404
    assert client.post("/api/orders/4/cancel", headers=auth_headers, json={}).status_code == 404


def test_nonexistent_order_returns_404(client, auth_headers):
    assert client.post("/api/orders/999/urgent", headers=auth_headers, json={}).status_code == 404
    assert client.post("/api/orders/999/cancel", headers=auth_headers, json={}).status_code == 404


def test_actions_require_auth(client):
    """没带 token：401，写操作更是必须验明身份。"""
    assert client.post("/api/orders/1/urgent", json={}).status_code == 401
    assert client.post("/api/orders/1/retest", json={}).status_code == 401
    assert client.post("/api/orders/1/cancel", json={}).status_code == 401


def test_confirm_message_not_written_to_others_conversation(client, auth_headers, lin_headers, raw_conn):
    """
    操作成功但 conversation_id 是别人的：确认消息不落库（防串写）。

    # 写操作本身的校验在委托单表（user_id 匹配），确认消息的校验在
    # 会话表（user_id 匹配）—— 两处防线互不代劳，各管各的表。
    """
    # 林教授还没有会话，先给他建一个（SAVEPOINT 内回滚，不留数据）
    lin_conv = client.post("/api/conversations", headers=lin_headers).json()["id"]

    resp = client.post(
        "/api/orders/1/urgent",
        headers=auth_headers,
        json={"conversation_id": lin_conv},
    )

    assert resp.status_code == 200 and resp.json()["success"] is True
    # 林教授的会话一条确认消息都没多（chen 被会话归属校验挡下）
    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM messages WHERE conversation_id = %s AND role = 'ai'",
            (lin_conv,),
        )
        assert cur.fetchone()[0] == 0
