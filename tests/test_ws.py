"""
test_ws.py —— WebSocket 通道的测试

分两层：
    1. ConnectionManager 的登记簿逻辑（假 WebSocket，不真连）
    2. /ws 端点的鉴权行为（TestClient 真握手）

不做"真推送端到端"：TestClient 的 WebSocket 和业务线程各自跑在不同
事件循环里，桥接属于框架行为，测它等于测 Starlette 本身。
推送逻辑由 _schedule_send 的离线分支 + 业务接口测试共同覆盖。
"""

import asyncio
import logging

import jwt as pyjwt
import pytest
from starlette.websockets import WebSocketDisconnect

from app import ws
from app.security import CurrentUser, create_token


class FakeWebSocket:
    """只记录调用的假连接（manager 的登记簿测试用）。"""

    def __init__(self, name: str):
        self.name = name
        self.sent: list[dict] = []
        self.closed = False


@pytest.fixture
def fresh_manager(monkeypatch):
    """每个用例一个干净的 manager（全局单例在测试里会互相串在线状态）。"""
    m = ws.ConnectionManager()
    monkeypatch.setattr(ws, "manager", m)
    return m


# 登记簿逻辑


def test_connect_and_disconnect_bookkeeping(fresh_manager):
    """上线登记、下线移除；用户和坐席分开存。"""
    user = CurrentUser(1, "chen", "customer")
    agent = CurrentUser(6, "tech001", "agent")
    uws, aws = FakeWebSocket("u1"), FakeWebSocket("a1")

    fresh_manager.connect(user, uws)
    fresh_manager.connect(agent, aws)

    assert fresh_manager._user_sockets == {1: uws}
    assert fresh_manager._agent_sockets == [aws]

    fresh_manager.disconnect(user, uws)
    fresh_manager.disconnect(agent, aws)
    assert fresh_manager._user_sockets == {}
    assert fresh_manager._agent_sockets == []


def test_duplicate_user_connection_replaces_old(fresh_manager):
    """同一用户第二个标签页连接：旧的被挤掉，只剩新的。"""
    user = CurrentUser(1, "chen", "customer")
    old, new = FakeWebSocket("old"), FakeWebSocket("new")

    fresh_manager.connect(user, old)
    fresh_manager.connect(user, new)

    assert fresh_manager._user_sockets[1] is new

    # 旧连接断开回调晚到：不能把新连接误删（is 判断的意义）
    fresh_manager.disconnect(user, old)
    assert fresh_manager._user_sockets[1] is new


def test_notify_user_offline_is_silent(fresh_manager, caplog):
    """
    用户不在线时推送：静默丢弃，不报错。

    # 这里"丢"是正确行为而不是缺陷 —— 推送只是提醒，
    # 消息本体已经落库，用户下次打开页面会通过拉历史补上。
    # 为了"不丢消息"引入离线队列会显著增加单机部署的复杂度。
    """
    with caplog.at_level(logging.WARNING):
        fresh_manager.notify_user(999, {"type": "message"})

    assert "推送" not in caplog.text  # 离线不算异常，连警告都不该有


def test_schedule_send_without_loop_is_graceful(fresh_manager, caplog):
    """事件循环还没绑定（如单元测试环境）时推送：警告并跳过，不抛异常。"""
    target = FakeWebSocket("u1")
    fresh_manager.connect(CurrentUser(1, "chen", "customer"), target)
    assert fresh_manager._loop is None

    with caplog.at_level(logging.WARNING):
        fresh_manager.notify_user(1, {"type": "message"})

    assert "事件循环" in caplog.text
    assert target.sent == []


def test_notify_user_delivers_when_loop_available(fresh_manager):
    """绑定事件循环后，业务线程的推送真的送到对端（run_coroutine_threadsafe 路径）。"""
    target = FakeWebSocket("u1")
    fresh_manager.connect(CurrentUser(1, "chen", "customer"), target)

    # 给假连接一个真的 async send_json（manager._send 会 await 它）
    async def _fake_send(payload):
        target.sent.append(payload)

    target.send_json = _fake_send  # type: ignore[method-assign]

    # 事件循环必须在独立线程里真的 run 起来，run_coroutine_threadsafe
    # 投递的协程才会被执行（只是创建 loop 不跑，协程永远不会 await）
    import threading
    import time as _t

    loop = asyncio.new_event_loop()
    fresh_manager._loop = loop
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    try:
        # 模拟真实场景：业务线程（这里就是主线程）发起推送
        fresh_manager.notify_user(1, {"type": "status"})
        for _ in range(50):  # 最多等 2.5 秒
            if target.sent:
                break
            _t.sleep(0.05)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        loop_thread.join(timeout=2)
        loop.close()

    assert target.sent == [{"type": "status"}]


# /ws 端点鉴权


def test_ws_rejects_invalid_token(client):
    """
    token 非法 → 关闭码 1008（策略拒绝）。

    # 必须主动 receive 才能观察到关闭。
    # 服务端"先 accept 再 close"，握手本身是成功的，直接进出 with
    # 什么异常都不会有；只有开始收数据，才会碰到那帧 1008 关闭。
    """
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect("/ws?token=not-a-jwt") as sock:
            sock.receive_text()  # 主动等服务端的关闭帧

    assert exc_info.value.code == 1008


def test_ws_accepts_valid_token(client):
    """合法 token → 握手成功，能收 ping（服务端忽略内容）。"""
    token = create_token(CurrentUser(1, "chen", "customer"))
    with client.websocket_connect(f"/ws?token={token}") as sock:
        sock.send_text("ping")  # 服务端收到但不处理，不应断开

    # 正常退出上下文 = 客户端主动断开，服务端 disconnect 清理不抛错


def test_ws_rejects_expired_token(client):
    """伪造/过期 token 同样 1008：任何 PyJWTError 都走同一个拒绝分支。"""
    bad = pyjwt.encode(
        {"sub": "1", "exp": 0},  # 1970 年就过期了
        "wrong-secret-but-long-enough-0123456789abcdef",  # 错的签名，长度合规避免无谓警告
        algorithm="HS256",
    )
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(f"/ws?token={bad}") as sock:
            sock.receive_text()

    assert exc_info.value.code == 1008
