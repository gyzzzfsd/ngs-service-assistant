"""
ws.py —— WebSocket 连接管理与实时推送

职责：
    1. 维护"谁在线"的两张表：按 user_id 索引的顾客连接 + 坐席连接列表
    2. 提供 sync 环境可调用的推送入口 notify_user / notify_agents
    3. /ws 端点：token 鉴权后挂上连接，等对方断开

不负责：
    决定什么时机该推什么（那是 api_chat / api_handoff 的业务逻辑）。

# 为什么接口函数是同步的（FastAPI 线程池里跑），这里却全是 async？
# WebSocket 的收发是协程，只能在事件循环里 await；而发消息等业务接口
# 跑在线程池线程里（sync def），线程里没有事件循环，不能直接 await。
# 标准解法：启动时抓住主事件循环（main.py 的 lifespan 里 capture_loop），
# 业务线程想推送时用 run_coroutine_threadsafe 把协程"投递"回主循环执行。
# 忘了这套机制直接在业务代码里 send，会报 "no running event loop"。

推送协议（前端按 type 字段分发）：
    {"type": "message",      "conversation_id": 1, "message": {id, role, content, created_at}}
    {"type": "status",       "conversation_id": 1, "status": "QUEUED|HUMAN|AI|CLOSED"}
    {"type": "queue_update"}  —— 通知坐席端刷新待接管列表
"""

import asyncio
import logging
import threading

import jwt
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.security import CurrentUser, decode_token

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """
    进程内的在线连接登记簿（单机够用，不引入 Redis）。

    线程安全说明：
        业务线程（推送方）和事件循环（断开回调）会并发读写两张表，
        所以所有结构性修改（connect/disconnect）都过同一把锁；
        _notify_* 只在锁内快照引用，发送动作放锁外，避免"发消息慢，拖住所有人断开"。
    """

    def __init__(self) -> None:
        self._user_sockets: dict[int, WebSocket] = {}  # 一个用户一个连接（后连的挤掉先连的）
        self._agent_sockets: list[WebSocket] = []
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None

    # ---------- 生命周期 ----------

    def capture_loop(self) -> None:
        """记录主事件循环（在 lifespan 的异步上下文里调用，否则拿不到 loop）。"""
        self._loop = asyncio.get_running_loop()
        logger.info("WebSocket 推送通道就绪（已绑定主事件循环）")

    def connect(self, user: CurrentUser, ws: WebSocket) -> None:
        """登记一条已 accept 的连接。同用户重复连接时旧的被替换（挤掉旧标签页）。"""
        with self._lock:
            if user.role == "agent":
                self._agent_sockets.append(ws)
            else:
                self._user_sockets[user.user_id] = ws
        logger.info("WS 上线 | user=%s role=%s 在线用户=%d 坐席=%d",
                    user.user_id, user.role, len(self._user_sockets), len(self._agent_sockets))

    def disconnect(self, user: CurrentUser, ws: WebSocket) -> None:
        """移除断开的连接。必须兜住 KeyError：重复断开/被挤掉的旧连接都会找不到。"""
        with self._lock:
            if user.role == "agent":
                if ws in self._agent_sockets:
                    self._agent_sockets.remove(ws)
            else:
                if self._user_sockets.get(user.user_id) is ws:
                    # is 而不是 ==：只有"当前登记的就是这条"才删，
                    # 防止新连接已覆盖同 user_id 时把新连接误删
                    del self._user_sockets[user.user_id]
        logger.info("WS 下线 | user=%s role=%s", user.user_id, user.role)

    # ---------- 推送入口（业务线程调用） ----------

    def notify_user(self, user_id: int, payload: dict) -> None:
        """推送给某一个用户（比如坐席回复了他的会话）。不在线就静默丢弃。"""
        with self._lock:
            ws = self._user_sockets.get(user_id)
        if ws is None:
            return
        self._schedule_send(ws, payload)

    def notify_agents(self, payload: dict) -> None:
        """广播给所有在线坐席（新排队会话、用户在人工模式下发消息）。"""
        with self._lock:
            sockets = list(self._agent_sockets)
        for ws in sockets:
            self._schedule_send(ws, payload)

    def notify_conversation(self, conv: dict, payload: dict) -> None:
        """推给会话主人 + 所有坐席（双方都要看到新消息）。conv 需含 user_id。"""
        self.notify_user(conv["user_id"], payload)
        self.notify_agents(payload)

    # ---------- 内部 ----------

    def _schedule_send(self, ws: WebSocket, payload: dict) -> None:
        """
        把一次发送投递回主事件循环。

        # 为什么逐条兜异常而不是让一条失败打断整批？
        # 某个坐席的连接刚好断开时 send 会抛 RuntimeError；
        # 若不兜住，广播循环中断，其他坐席/用户就收不到本该收到的消息。
        # 单条失败只打 DEBUG：断开瞬间的发送失败是常态，不是故障。
        """
        if self._loop is None or self._loop.is_closed():
            logger.warning("WS 推送跳过：事件循环不可用（服务正在关闭？）")
            return

        async def _send() -> None:
            try:
                await ws.send_json(payload)
            except Exception:  # noqa: BLE001 —— 见 docstring
                logger.debug("WS 单条推送失败（连接可能已断开）", exc_info=True)

        asyncio.run_coroutine_threadsafe(_send(), self._loop)


# 全局唯一实例：main.py、api_chat、api_handoff 都 import 这一个
manager = ConnectionManager()


@router.websocket("/ws")
async def ws_endpoint(websocket: WebSocket, token: str = "") -> None:
    """
    WebSocket 端点：?token=<JWT> 鉴权后保持连接。

    Args:
        websocket: FastAPI 注入的连接对象。
        token: 查询参数里的 JWT。浏览器原生 WebSocket API 没法自定义请求头，
            所以 token 走 URL —— 这是 WS 鉴权的通行做法（HTTP 接口仍走 Header）。

    行为：
        token 无效 → 拒绝（1008 策略违例）；合法 → accept 并登记。
        连接期间不期望收到任何业务数据（收到的文本一律忽略，纯当保活），
        推送全部由服务端主动发起。
    """
    try:
        user = decode_token(token)
    except jwt.PyJWTError:
        # 先 accept 再 close，而不是直接 close。Starlette 的实现里
        # 未 accept 的连接直接 close 会发 HTTP 403 握手失败，前端
        # onerror/onclose 都能收到，但拿不到"为什么"。先 accept 再用
        # 1008 关闭码，语义更明确（策略拒绝，不是崩溃）。
        await websocket.accept()
        await websocket.close(code=1008)
        return

    await websocket.accept()
    manager.connect(user, websocket)
    try:
        while True:
            # 收什么内容不重要（前端可能只发 ping）；这里的作用是
            # 阻塞等待，直到对方断开触发 WebSocketDisconnect
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(user, websocket)
