"""
test_api_handoff.py —— 转人工与坐席端接口的测试

完整走一遍状态机：AI → QUEUED → HUMAN → AI/CLOSED，
外加权限（顾客访问坐席接口 403）和越权（404）分支。
"""


def test_handoff_moves_ai_to_queued(client, auth_headers, conversation_id, raw_conn):
    """转人工：AI → QUEUED，重复请求幂等返回 QUEUED。"""
    resp = client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json()["status"] == "QUEUED"

    # 幂等：再点一次不报错
    resp2 = client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    assert resp2.json()["status"] == "QUEUED"


def test_handoff_on_human_returns_current(client, auth_headers, conversation_id, agent_headers, raw_conn):
    """已经是 HUMAN 时再请求转人工：返回当前状态（坐席已在场）。"""
    with raw_conn.cursor() as cur:
        cur.execute("UPDATE conversations SET status = 'HUMAN' WHERE id = %s", (conversation_id,))

    resp = client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    assert resp.json()["status"] == "HUMAN"


def test_handoff_requires_owner(client, auth_headers, conversation_id, lin_headers):
    """林教授请求陈医生的会话转人工 → 404（防枚举）。"""
    resp = client.post(f"/api/conversations/{conversation_id}/handoff", headers=lin_headers)

    assert resp.status_code == 404


def test_queued_list_shows_waiting_conversations(client, auth_headers, conversation_id, agent_headers):
    """排队列表：含会话 ID、用户昵称、最近一条消息预览。"""
    # 排队
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    resp = client.get("/api/agent/queued", headers=agent_headers)

    assert resp.status_code == 200
    items = resp.json()
    assert any(i["conversation_id"] == conversation_id for i in items)
    mine = next(i for i in items if i["conversation_id"] == conversation_id)
    assert mine["user_nickname"] == "陈医生"
    assert mine["last_message"] is not None  # 种子会话里原有消息


def test_accept_moves_queued_to_human(client, auth_headers, conversation_id, agent_headers, raw_conn):
    """坐席接管：QUEUED → HUMAN。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers
    )

    assert resp.status_code == 200
    assert resp.json()["status"] == "HUMAN"


def test_accept_twice_second_gets_409(client, auth_headers, conversation_id, agent_headers):
    """
    双坐席同时点"接管"：第二个收到 409。

    # 这靠 UPDATE ... WHERE status='QUEUED' 的原子性，
    # 而不是应用层先查再改 —— 后者在并发下两个请求都能通过检查。
    """
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    first = client.post(
        f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers
    )
    second = client.post(
        f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers
    )

    assert first.status_code == 200
    assert second.status_code == 409


def test_agent_endpoints_forbid_customers(client, auth_headers, conversation_id):
    """顾客访问坐席接口 → 403（接口存在不是秘密，身份才是门槛）。"""
    assert (
        client.get("/api/agent/queued", headers=auth_headers).status_code == 403
    )
    assert (
        client.post(
            f"/api/agent/conversations/{conversation_id}/accept", headers=auth_headers
        ).status_code
        == 403
    )
    assert (
        client.post(
            f"/api/agent/conversations/{conversation_id}/messages",
            headers=auth_headers,
            json={"content": "hi"},
        ).status_code
        == 403
    )


def test_agent_reply_stores_human_role(client, auth_headers, conversation_id, agent_headers, raw_conn):
    """坐席回复落库为 role=human，用户拉历史能看到。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers)

    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        headers=agent_headers,
        json={"content": "您好，我是技术支持老徐"},
    )

    assert resp.status_code == 200
    assert resp.json()["role"] == "human"
    assert "老徐" in resp.json()["content"]

    # 用户视角拉历史
    msgs = client.get(
        f"/api/conversations/{conversation_id}/messages", headers=auth_headers
    ).json()
    assert any(m["role"] == "human" for m in msgs)


def test_agent_reply_rejected_when_queued(client, auth_headers, conversation_id, agent_headers):
    """没接管（QUEUED）就抢答 → 409。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        headers=agent_headers,
        json={"content": "提前抢答"},
    )

    assert resp.status_code == 409


def test_agent_reply_blank_content_rejected(client, auth_headers, conversation_id, agent_headers):
    """坐席也不能发空白串（与用户同一套清洗）。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers)

    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        headers=agent_headers,
        json={"content": "   "},
    )

    assert resp.status_code == 422


def test_close_back_to_ai(client, auth_headers, conversation_id, agent_headers, stub_agent):
    """
    坐席结束（back=ai）：HUMAN → AI，用户的下一句话重新由 AI 接待。

    人工说的话已在历史里（role=human → assistant + [人工客服] 前缀），
    AI 能接着上下文继续服务。
    """
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers)
    client.post(
        f"/api/agent/conversations/{conversation_id}/messages",
        headers=agent_headers,
        json={"content": "问题已解决，还有其他需要吗"},
    )

    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/close",
        headers=agent_headers,
        json={"back": "ai"},
    )
    assert resp.json()["status"] == "AI"

    # AI 重新接管：这次发消息应拿到 reply（stub 的内容）
    from test_api_chat import _fake_run  # 同目录模块，pytest 已把 tests/ 放进 sys.path

    stub_agent(_fake_run("AI 重新为您服务"))
    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "我还有个问题"},
    )
    assert "reply" in resp.json()


def test_close_rejects_when_not_human(client, auth_headers, conversation_id, agent_headers):
    """会话不在人工服务中，close → 409（防误关 AI 会话）。"""
    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/close",
        headers=agent_headers,
        json={"back": "ai"},
    )
    assert resp.status_code == 409


# 接待中列表（HUMAN 状态）
# 这一组用例守的是"接管后会话不能失联"：会话一旦从 QUEUED 队列消失，
# 就必须出现在 /api/agent/active 里，否则坐席切走后无法回到该会话。


def test_accept_records_claimed_by(client, auth_headers, conversation_id, agent_headers, raw_conn):
    """接管写入 claimed_by：会话从此归属这位坐席。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers)

    with raw_conn.cursor() as cur:
        cur.execute(
            "SELECT status, claimed_by FROM conversations WHERE id = %s", (conversation_id,)
        )
        conv_status, claimed_by = cur.fetchone()

    assert conv_status == "HUMAN"
    assert claimed_by == 6  # tech001 的 user_id，见种子数据


def test_active_list_shows_human_conversation(
    client, auth_headers, conversation_id, agent_headers
):
    """接管后会话出现在"接待中"列表，并标记为当前坐席的会话。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers)

    resp = client.get("/api/agent/active", headers=agent_headers)

    assert resp.status_code == 200
    items = resp.json()
    mine = next((i for i in items if i["conversation_id"] == conversation_id), None)
    assert mine is not None, "接管后的会话必须留在接待中列表里"
    assert mine["user_nickname"] == "陈医生"
    assert mine["claimed_by_me"] is True
    assert mine["last_message"] is not None  # 种子会话里原有消息
    assert mine["last_active_at"] is not None


def test_active_list_excludes_queued(client, auth_headers, conversation_id, agent_headers):
    """只排队未接管的会话不进"接待中"，仍留在"待接管"里。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)

    active_ids = [
        i["conversation_id"]
        for i in client.get("/api/agent/active", headers=agent_headers).json()
    ]
    queued_ids = [
        i["conversation_id"]
        for i in client.get("/api/agent/queued", headers=agent_headers).json()
    ]

    assert conversation_id not in active_ids
    assert conversation_id in queued_ids


def test_active_list_removed_after_close(
    client, auth_headers, conversation_id, agent_headers, raw_conn
):
    """结束服务（back=ai）后会话离开"接待中"，claimed_by 一并清空。"""
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=agent_headers)
    client.post(
        f"/api/agent/conversations/{conversation_id}/close",
        headers=agent_headers,
        json={"back": "ai"},
    )

    active_ids = [
        i["conversation_id"]
        for i in client.get("/api/agent/active", headers=agent_headers).json()
    ]
    assert conversation_id not in active_ids

    with raw_conn.cursor() as cur:
        cur.execute("SELECT claimed_by FROM conversations WHERE id = %s", (conversation_id,))
        assert cur.fetchone()[0] is None


def test_active_endpoint_forbids_customers(client, auth_headers):
    """顾客访问"接待中"列表 → 403。"""
    assert client.get("/api/agent/active", headers=auth_headers).status_code == 403
