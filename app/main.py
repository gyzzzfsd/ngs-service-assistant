"""
main.py —— 启动入口

职责：
    装配整个后端：建 FastAPI 实例、开/关数据库连接池、挂 CORS、注册路由。

不负责：
    任何业务逻辑。这个文件里不该出现 SQL、不该出现 if 判断业务状态。

启动方式（在项目根目录执行）：
    uvicorn app.main:app --reload --port 8010
    然后浏览器打开 http://127.0.0.1:8010/docs 就能看到所有接口并直接试。
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import api_actions, api_auth, api_chat, api_handoff, db, ws
from app.config import settings
from app.log_setup import setup_logging

# 日志必须在任何业务模块打日志之前初始化。
# 这里调用一次是为了"被 import 时也不会一片空白"（比如 PyCharm 里只跑 main.py）。
# setup_logging 内部幂等，run.py 再调一次也安全。
setup_logging()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    应用生命周期钩子：启动前建连接池，关闭时释放。

    # 为什么连接池要放在 lifespan 里，而不是模块顶层直接 init_pool()？
    # 模块顶层执行会在 import 时就建连 —— 这意味着跑单元测试、或者只 import 一下
    # 某个函数（比如想单独测 harness）都会去连数据库，测试环境没库就直接崩。
    # 放进 lifespan 后，只有真正启动服务才会建连，"import 是安全的"。
    """
    logger.info("正在连接数据库 ...")
    db.init_pool()
    logger.info("数据库连接池就绪（max_size=%s）", settings.db_pool_size)
    # 抓住主事件循环给 WebSocket 推送用：业务接口跑在线程池里，
    # 想推送时靠这个 loop 把协程投递回来（原理见 ws.py 的模块说明）
    ws.manager.capture_loop()
    logger.info("服务已启动：http://127.0.0.1:%s/docs", settings.app_port)
    try:
        yield
    finally:
        db.close_pool()
        logger.info("数据库连接池已关闭")


app = FastAPI(
    title=settings.app_name,
    description="肿瘤 NGS 送检全流程智能助手后端（演示数据，不构成医学建议）",
    version="0.2.0",
    lifespan=lifespan,
)

# CORS：阶段④ 前端页面如果直接由本服务托管，其实用不到 CORS；
# 但如果用 VSCode Live Server 或独立端口打开页面，就必须要。
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_auth.router)
app.include_router(api_chat.router)
app.include_router(api_actions.router)
app.include_router(api_handoff.router)
app.include_router(ws.router)


@app.get("/health", tags=["运维"])
def health() -> dict:
    """
    健康检查：确认服务活着、数据库连得上。

    部署或排查问题时第一个该敲的接口。数据库打不通时返回 500，
    这样负载均衡/监控能立刻发现，而不是等到有用户来提问才暴露。
    """
    with db.get_conn() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.fetchone()
    return {"status": "ok", "env": settings.app_env, "model": settings.llm_model}
