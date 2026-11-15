"""
db.py —— 数据库连接池

职责：
    启动时创建 psycopg 连接池，对外只暴露 get_conn() 这个上下文管理器。

不负责：
    任何业务 SQL。订单 SQL 在 tools.py，向量 SQL 在 rag.py。
    —— 这是刻意的：如果 SQL 都堆在 db.py，改一个查询条件就要在几百行 SQL 里翻，
    所以让 SQL 跟着功能走，谁用谁写。

为什么用连接池而不是每次 connect()：
    建立一条 PG 连接要 TCP 握手 + 认证，约 10ms。一次聊天要查委托单、查产品、查知识库，
    每条都新建连接的话延迟会叠加。池子建好 5 条复用，单次查询降到 1ms 以内。
"""

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

from app.config import settings

logger = logging.getLogger(__name__)

# 模块级单例。None 表示还没初始化 —— get_conn() 会显式报错，
# 而不是安静地返回一条没有注册向量类型的连接，导致报错信息难以理解。
_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    """
    每建立一条新的物理连接时调用一次（不是每次借出时），做连接级的初始化。

    Args:
        conn: 刚从池子里新建出来的连接。

    为什么在这里注册向量类型：
        pgvector 的 vector 类型不是 PG 内置类型，psycopg 默认不知道怎么把
        Python list 转成 vector。register_vector 会查询 pg_type 拿到 OID 并注册适配器。
        注册是"按连接生效"的，所以每条新连接都要注册一次 —— 放在 configure 回调里
        正好每条物理连接只跑一次，不会拖慢每次借还。

    注意：
        这行要求目标库已经装好 vector 扩展（本项目是 0.8.7）。扩展没装会在这里直接
        抛异常，连接池初始化就失败 —— 这是好事，启动即暴露，不用等到第一次向量查询。
    """
    register_vector(conn)


def init_pool() -> None:
    """
    创建并打开连接池。只应该在 main.py 的 lifespan 启动阶段调用一次。

    Raises:
        psycopg_pool.PoolTimeout: 10 秒内连不上数据库（网络不通、密码错、容器没起）。

    为什么用 open=False 再手动 open(wait=True)：
        构造函数里直接开池会在后台异步建连，配置错了要等到第一次查询才发现。
        显式 wait=True 会把建连阻塞在启动阶段 —— 数据库没连上服务器就起不来，
        比"服务起来了但每个请求都 500"容易排查得多。
    """
    global _pool
    if _pool is not None:
        return

    # 日志里不打印带密码的连接串，只打印主机和库名
    safe_target = settings.psycopg_dsn.split("@")[-1] if "@" in settings.psycopg_dsn else "(未知)"
    logger.info("创建数据库连接池 | target=%s max_size=%s", safe_target, settings.db_pool_size)

    _pool = ConnectionPool(
        conninfo=settings.psycopg_dsn,
        min_size=1,
        max_size=settings.db_pool_size,
        configure=_configure,
        open=False,
        timeout=10,
    )
    try:
        _pool.open(wait=True, timeout=10)
    except Exception:
        # 连不上数据库是最常见的启动失败原因，日志里把目标地址和异常类型都打出来
        logger.exception("数据库连接池初始化失败 | target=%s", safe_target)
        _pool = None
        raise
    logger.info("数据库连接池就绪")


def close_pool() -> None:
    """关闭连接池，释放所有物理连接。在 lifespan 退出时调用。"""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
        logger.info("数据库连接池已关闭")


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    """
    从池子里借一条连接，用完自动归还。

    正常退出自动 commit；抛异常自动 rollback —— 这两个行为由
    psycopg_pool 的 connection() 上下文负责，不用手写 try/except。

    用法：
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("select ...")

    Raises:
        RuntimeError: 池子未初始化（忘记调 init_pool，或 import 顺序不对）。
    """
    if _pool is None:
        logger.error("连接池未初始化就尝试取连接，通常是 lifespan 未执行或 import 顺序有误")
        raise RuntimeError(
            "数据库连接池未初始化：请确认 main.py 的 lifespan 启动阶段调用了 init_pool()"
        )
    with _pool.connection() as conn:
        yield conn
