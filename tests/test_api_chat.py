"""
test_api_chat.py —— 聊天接口测试

模型调用全部 mock 掉（agent.run 直接返回构造好的 AgentRun），
这样测的是"接口自身的逻辑"：会话复用、越权拦截、状态拦截、落库顺序、历史映射。
这些逻辑跟模型无关，用假模型测反而更准、更快。
"""

import json

import pytest
from psycopg.rows import dict_row

from app import agent, harness
from app.schemas import AgentOutput, ReplyType


def _fake_run(content="好的", reply_type=ReplyType.ANSWER, action=None, error=None, trace=None):
    """
    造一个假的 agent.run 返回值。

    这个函数的存在本身就是一条设计约束：接口只依赖 AgentRun 这个结构，
    不关心 agent 内部怎么跑。所以只要结构对，接口逻辑就能独立验证。

    validation 默认造一个"校验通过"的结论；需要演练降级/拦截的用例
    自己传 error 或换别的 validation。

    注：stub_agent 夹具已提升到 conftest.py（test_api_handoff 也要用）。
    """
    output = AgentOutput(reply_type=reply_type, content=content, action=action)
    validation = None if error else harness.validate(output, trace or [])
    return agent.AgentRun(
        output=output,
        tool_trace=trace or [],
        draft=content,
        messages=[],
        error=error,
        validation=validation,
        corrected=False,
    )


# 会话


def test_create_conversation(client, auth_headers):
    resp = client.post("/api/conversations", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json()["status"] == "AI"


def test_create_conversation_reuses_existing(client, auth_headers):
    """
    连续调两次必须返回同一个会话。

    # 如果每次刷新都新建，聊天记录会被切成碎片，
    # agent 的 history 里看不到上一轮说了什么，多轮上下文直接断掉。
    """
    first = client.post("/api/conversations", headers=auth_headers).json()
    second = client.post("/api/conversations", headers=auth_headers).json()

    assert first["id"] == second["id"]


def test_mine_returns_none_when_no_conversation(client, raw_conn):
    """
    新用户没有会话时返回 null，而不是 404 —— 前端据此决定显示空态。

    # 这条不能靠"删掉林教授的会话"来造场景。
    # 别人的会话里可能已经有消息，直接 DELETE 会撞外键报错，
    # 于是测试结果就取决于"之前有没有人手工跑过聊天"——这种用例迟早会莫名其妙变红。
    # 正确做法是现场造一个干净的临时用户，用完随事务一起回滚。
    """
    from app.security import hash_password

    with raw_conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO users (username, password_hash, nickname, role)
            VALUES ('tmp_no_conv', %s, '临时用户', 'customer')
            RETURNING id
            """,
            (hash_password("123456"),),
        )
        temp_username = "tmp_no_conv"

    token = client.post("/api/auth/login", json={"username": temp_username, "password": "123456"}).json()["access_token"]

    resp = client.get("/api/conversations/mine", headers={"Authorization": f"Bearer {token}"})

    assert resp.status_code == 200
    assert resp.json() is None


def test_endpoints_require_auth(client):
    assert client.post("/api/conversations").status_code == 401
    assert client.get("/api/conversations/mine").status_code == 401


# 发消息


def test_send_message_persists_both_sides(client, auth_headers, conversation_id, stub_agent):
    """一轮问答后，数据库里应该正好多出一条 user 和一条 ai。"""
    stub_agent(_fake_run("您有 7 张委托单。"))

    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "我有哪些委托单"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["reply"]["content"] == "您有 7 张委托单。"
    assert data["reply"]["reply_type"] == "ANSWER"
    # 阶段② 起 validation 必有值：这次应是通过（内容无业务事实，五关卡直接放行）
    assert data["validation"]["status"] == "passed"
    assert data["validation"]["passed"] is True
    assert data["validation"]["error_codes"] == []

    # 历史接口应能读到这两条，且顺序是 user 在前
    msgs = client.get(f"/api/conversations/{conversation_id}/messages", headers=auth_headers).json()
    assert [m["role"] for m in msgs[-2:]] == ["user", "ai"]
    assert msgs[-2]["content"] == "我有哪些委托单"
    assert msgs[-1]["content"] == "您有 7 张委托单。"


def test_send_message_passes_history_before_inserting(client, auth_headers, conversation_id, stub_agent):
    """
    agent 收到的 history 里不能包含"本轮这条"。

    # 如果先插入再读历史，这句"我有哪些委托单"会在 history 里出现一次，
    # agent 又把它当本轮输入追加一次，模型会看到同一句话两遍，
    # 表现为重复回答或把用户的问句当成已知信息。
    """
    calls = stub_agent(_fake_run("好的"))

    client.post(f"/api/conversations/{conversation_id}/messages", headers=auth_headers, json={"content": "第一句"})
    client.post(f"/api/conversations/{conversation_id}/messages", headers=auth_headers, json={"content": "第二句"})

    second_history = calls[1]["history"]
    contents = [m["content"] for m in second_history]
    assert "第一句" in contents          # 上一轮应该在里面
    assert "第二句" not in contents       # 本轮不该在里面


def test_send_message_maps_human_role_to_assistant(client, auth_headers, conversation_id, stub_agent, raw_conn):
    """
    人工客服说的话要映射成 assistant 并加 [人工客服] 前缀。

    # 映射成 user 会让模型以为用户在自言自语；
    # 不加前缀，模型会以为那些承诺是自己许下的，下一轮继续兑现。
    """
    with raw_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'human', %s)",
            (conversation_id, "我是技术支持老徐，已为您登记"),
        )

    calls = stub_agent(_fake_run("好的"))
    client.post(f"/api/conversations/{conversation_id}/messages", headers=auth_headers, json={"content": "谢谢"})

    history = calls[0]["history"]
    human_msg = [m for m in history if "技术支持老徐" in m["content"]][0]
    assert human_msg["role"] == "assistant"
    assert human_msg["content"].startswith("[人工客服]")


def test_fallback_message_is_not_fed_back_to_model(client, auth_headers, conversation_id, stub_agent, raw_conn):
    """
    兜底话术要落库（用户要能看到），但绝不能进模型上下文。

    # 这是实测到的死循环 —— 一旦有过一次兜底，那句话就成了 assistant 历史，
    # 模型看到"我上次这么答的"，于是继续照抄，此后再也答不出正常内容。
    # 用户侧表现是"这个客服突然坏了，怎么问都是同一句话"。
    """
    with raw_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'ai', %s)",
            (conversation_id, agent.FALLBACK_CONTENT),
        )
        cur.execute(
            "INSERT INTO messages (conversation_id, role, content) VALUES (%s, 'ai', %s)",
            (conversation_id, "这是一条正常的回答"),
        )

    calls = stub_agent(_fake_run("好的"))
    client.post(f"/api/conversations/{conversation_id}/messages", headers=auth_headers, json={"content": "继续"})

    contents = [m["content"] for m in calls[0]["history"]]
    assert agent.FALLBACK_CONTENT not in contents       # 兜底话术被剔除
    assert "这是一条正常的回答" in contents              # 正常历史保留


def test_send_message_surfaces_degradation(client, auth_headers, conversation_id, stub_agent):
    """agent 走兜底时，validation 里要带上原因，前端才能提示"已转人工"。"""
    stub_agent(
        _fake_run(
            content=agent.FALLBACK_CONTENT,
            reply_type=ReplyType.REQUEST_HANDOFF,
            error="LLM_ERROR: 主模型失败",
        )
    )

    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "你好"},
    )

    assert resp.status_code == 200
    assert resp.json()["reply"]["reply_type"] == "REQUEST_HANDOFF"
    # 模型级失败 → degraded_model，错误码从 error 前缀提取
    v = resp.json()["validation"]
    assert v["status"] == "degraded_model"
    assert v["passed"] is False
    assert v["error_codes"] == ["LLM_ERROR"]


def test_send_message_returns_action_for_urgent(client, auth_headers, conversation_id, stub_agent):
    """ACTION_GUIDE 必须原样把 action 传给前端 —— 前端靠它渲染"确认加急"按钮。"""
    stub_agent(
        _fake_run(
            content="委托单 1 是已接收状态，可以加急，请确认。",
            reply_type=ReplyType.ACTION_GUIDE,
            action={"code": "APPLY_URGENT", "order_id": 1},
        )
    )

    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "帮我加急委托单1"},
    )

    action = resp.json()["reply"]["action"]
    assert action["code"] == "APPLY_URGENT"
    assert action["order_id"] == 1


def test_empty_content_rejected(client, auth_headers, conversation_id):
    """空消息在 Pydantic 层就被拦下（422），不会白跑一次模型。"""
    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "   "},
    )

    assert resp.status_code == 422


def test_too_long_content_rejected(client, auth_headers, conversation_id):
    """超长输入拦在入口，避免把几万字符塞给模型烧钱。"""
    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "啊" * 3000},
    )

    assert resp.status_code == 422


# 权限与状态


def test_cannot_send_to_others_conversation(client, auth_headers, conversation_id, lin_headers, stub_agent):
    """
    林教授不能往陈医生的会话里发消息，且必须返回 404（而不是 403）。

    # 403 等于承认"这个 ID 存在，只是你没权限"，可被用来枚举他人会话。
    # 404 让"不存在"和"不是你的"在响应上完全一致。
    """
    stub_agent(_fake_run("不该出现"))

    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=lin_headers,
        json={"content": "偷看一下"},
    )

    assert resp.status_code == 404
    assert resp.json()["detail"] == "会话不存在"


def test_cannot_read_others_messages(client, conversation_id, lin_headers):
    resp = client.get(f"/api/conversations/{conversation_id}/messages", headers=lin_headers)

    assert resp.status_code == 404


def test_nonexistent_conversation_returns_404(client, auth_headers, stub_agent):
    stub_agent(_fake_run("x"))

    resp = client.post("/api/conversations/999999/messages", headers=auth_headers, json={"content": "在吗"})

    assert resp.status_code == 404


def test_human_takeover_blocks_ai(client, auth_headers, conversation_id, stub_agent, raw_conn):
    """
    会话转人工后，用户消息不再触发 AI，而是入库转投坐席。

    阶段④ 起行为变了：以前返回 409（当时没有坐席端），现在消息照常入库
    （role=user）并返回 UserMessageAck —— AI 是否"闭嘴"靠
    "响应里没有 reply 字段"来断言，这正是人工模式的语义。

    # 这里不能"继续让 AI 答"。人工已经接手，AI 再插一句话
    # 就是两套说法同时存在，用户会不知道该信谁。
    """
    with raw_conn.cursor() as cur:
        cur.execute("UPDATE conversations SET status = 'HUMAN' WHERE id = %s", (conversation_id,))

    stub_agent(_fake_run("不该出现"))  # 若 AI 被调用，测试会因 reply 字段出现而失败
    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "在吗"},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "reply" not in data          # 没有 AI 回复
    assert data["status"] == "HUMAN"     # 前端据此渲染"人工服务中"
    # 消息确实入库了（role=user），坐席端拉历史能看到
    assert data["user_message_id"] > 0


def test_queued_message_relayed_without_ai(client, auth_headers, conversation_id, stub_agent):
    """QUEUED（排队中）同样不调 AI：等待期内的消息进历史，接管后坐席可见。"""
    # 通过接口触发 QUEUED（而不是直接 UPDATE），顺带验证 handoff 本身
    resp = client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["status"] == "QUEUED"

    stub_agent(_fake_run("不该出现"))
    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "还在吗"},
    )

    assert resp.status_code == 200
    assert "reply" not in resp.json()
    assert resp.json()["status"] == "QUEUED"


def test_closed_conversation_rejects_message(client, auth_headers, conversation_id):
    """CLOSED 是终态：发消息明确 409，让前端引导用户开新会话。"""
    from app.security import CurrentUser, create_token

    tech_headers = {"Authorization": "Bearer " + create_token(CurrentUser(6, "tech001", "agent"))}
    client.post(f"/api/conversations/{conversation_id}/handoff", headers=auth_headers)
    client.post(f"/api/agent/conversations/{conversation_id}/accept", headers=tech_headers)
    resp = client.post(
        f"/api/agent/conversations/{conversation_id}/close",
        headers=tech_headers,
        json={"back": "closed"},
    )
    assert resp.json()["status"] == "CLOSED"

    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "你好"},
    )
    assert resp.status_code == 409


def test_history_limit_clamped(client, auth_headers, conversation_id):
    """limit 超出范围要被夹住，不能让前端一次拉十万条。"""
    resp = client.get(f"/api/conversations/{conversation_id}/messages?limit=99999", headers=auth_headers)

    assert resp.status_code == 200
    assert len(resp.json()) <= 200


# 日志（对应"该写日志的地方必须写日志"这条要求）


def test_send_message_writes_log(client, auth_headers, conversation_id, stub_agent, caplog):
    """收到消息与回复完成都必须留日志，否则线上出问题无法回溯。"""
    import logging

    caplog.set_level(logging.INFO)
    stub_agent(_fake_run("好的"))

    client.post(f"/api/conversations/{conversation_id}/messages", headers=auth_headers, json={"content": "在吗"})

    text = caplog.text
    assert "收到消息" in text
    assert "回复完成" in text


def test_unauthorized_access_is_logged(client, conversation_id, lin_headers, stub_agent, caplog):
    """越权访问必须留下 WARNING 级别的审计日志。"""
    import logging

    caplog.set_level(logging.WARNING)
    stub_agent(_fake_run("x"))

    client.post(f"/api/conversations/{conversation_id}/messages", headers=lin_headers, json={"content": "偷看"})

    assert "越权访问" in caplog.text


# 转人工（用户点名要人工，走 agent 快速通道）


def test_explicit_handoff_returns_handoff_without_llm(client, auth_headers, conversation_id, raw_conn):
    """
    用户发"转人工" → 接口直接返回 REQUEST_HANDOFF，前端据此渲染【转接人工支持】按钮。

    # 这条用例故意不 mock agent.run。
    # 它要验证的正是"这条路径根本不该碰模型"——一旦有人把快速通道挪到模型后面，
    # 测试会撞上 _no_real_llm 的哨兵直接报错，而不是悄悄变慢、偶尔答"无法转接"。
    # 之前线上就是这个症状：同一句"转人工"，陈医生被判 DECLINE、林教授被判 REQUEST_HANDOFF。
    """
    resp = client.post(
        f"/api/conversations/{conversation_id}/messages",
        headers=auth_headers,
        json={"content": "转人工"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"]["reply_type"] == "REQUEST_HANDOFF"
    assert "无法" not in body["reply"]["content"]
    # 没有业务事实，校验应记 passed（而不是被误判成 degraded_model）
    assert body["validation"]["status"] == "passed"
    assert body["validation"]["error_codes"] == []

    # 用户消息和 AI 应答都要落库（角色顺序：user 在 ai 之前）
    with raw_conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT role, content FROM messages WHERE conversation_id = %s ORDER BY id DESC LIMIT 2",
            (conversation_id,),
        )
        rows = list(reversed(cur.fetchall()))
    assert [(r["role"]) for r in rows] == ["user", "ai"]
    assert rows[0]["content"] == "转人工"
