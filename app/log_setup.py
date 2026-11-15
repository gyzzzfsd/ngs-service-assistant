"""
log_setup.py —— 日志配置（唯一初始化日志的地方）

职责：
    配置日志的"输出到哪里、长什么样、什么级别"：
      - 控制台（PyCharm 的 Run 窗口能直接看到）
      - 文件（logs/app.log，5MB 轮转，留 3 份历史）

不负责：
    打日志本身。各模块只需要 `logger = logging.getLogger(__name__)` 然后 logger.info(...)，
    不需要关心 handler、格式、文件路径。

为什么单独一个文件：
    日志是横切关注点 —— 每个模块都要用，但谁都不该负责初始化它。
    分散配置最典型的后果是：每个模块各加一个 handler，
    同一条日志在控制台打印 5 遍，文件里也重复 5 行。

调用时机：
    run.py（PyCharm 启动）、main.py 的模块顶层、ingest.py 的 main() 各调用一次。
    函数内部有幂等保护，重复调用不会重复加 handler。
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import BASE_DIR, settings

LOG_DIR: Path = BASE_DIR / "logs"
LOG_FILE: Path = LOG_DIR / "app.log"

# 日志格式：时间 | 级别 | 模块名 | 内容
# 模块名用 %(name)s 而不是写死，这样能一眼看出是哪个文件打的日志（app.tools / app.agent ...）
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-18s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False

# 这些第三方库在 DEBUG 级别会把每次 HTTP 请求的完整报文打出来，
# 一条聊天能刷几十行，把真正有用的业务日志淹掉，所以统一压到 WARNING。
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "asyncio", "psycopg.pool")


def setup_logging() -> None:
    """
    初始化日志（幂等，重复调用只生效第一次）。

    级别取 .env 的 LOG_LEVEL，默认 INFO：
        INFO  正常的请求、工具调用、模型调用
        DEBUG 额外打印完整 prompt 和工具参数（排查模型"为什么这么答"时用）
        WARNING 只看异常

    Raises:
        无。日志目录创建失败（比如没有写权限）时降级为"只输出到控制台"，
        不让"日志写不了"把整个服务拖崩 —— 日志是辅助，不能成为故障源。
    """
    global _configured
    if _configured:
        return

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # ---------- 控制台 ----------
    # 用 stdout 而不是 stderr：PyCharm 的 Run 窗口对 stdout 着色更正常，
    # 而且 uvicorn 本身也往 stdout 写，混在一起按时间顺序看更连贯。
    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ---------- 文件 ----------
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        # 只提示一次，不抛异常
        root.warning("日志文件初始化失败，仅输出到控制台：%s", exc)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True
    logging.getLogger(__name__).info(
        "日志已就绪 | level=%s | file=%s", settings.log_level.upper(), LOG_FILE
    )
