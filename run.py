"""
run.py —— 后端启动入口（PyCharm 里就是运行这个文件）

用法：
    PyCharm: 右键 run.py → Run 'run'（或 Debug 'run'）
    命令行:  python run.py

为什么需要这个文件而不直接在 PyCharm 里跑 uvicorn 命令：
    PyCharm 的 Python 运行配置只能指定"脚本文件"，不能直接跑 `uvicorn` 这个可执行程序
    （除非配成模块运行并手填一堆参数）。包一层 run.py 之后，只要把 Script path 指向它，
    选好项目解释器，就能一键启动，还能直接下断点调试。

热重载：
    默认关闭 reload。原因是 reload 模式下 uvicorn 会起子进程，
    PyCharm 的断点打在父进程里不生效，调试体验会莫名其妙。
    改代码频繁、不需要断点时，设置环境变量再启动：
        PowerShell:  $env:UVICORN_RELOAD="1"; python run.py
"""

import os
import sys
from pathlib import Path

# 把项目根目录加进 sys.path，保证从任何工作目录启动都能 import app 包。
# （PyCharm 默认工作目录是项目根，但命令行里可能在别处执行本文件。）
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn  # noqa: E402 —— 必须放在 sys.path 设置之后

from app.config import settings  # noqa: E402
from app.log_setup import setup_logging  # noqa: E402


def main() -> None:
    """初始化日志并启动 uvicorn。"""
    setup_logging()

    reload_enabled = os.getenv("UVICORN_RELOAD", "0") == "1"
    kwargs: dict = {
        "host": settings.app_host,
        "port": settings.app_port,
        "reload": reload_enabled,
        "log_level": settings.log_level.lower(),
    }
    if reload_enabled:
        # 只监听 app 目录，避免日志文件一写就触发重启（logs/ 在项目根下）
        kwargs["reload_dirs"] = [str(PROJECT_ROOT / "app")]

    uvicorn.run("app.main:app", **kwargs)


if __name__ == "__main__":
    main()
