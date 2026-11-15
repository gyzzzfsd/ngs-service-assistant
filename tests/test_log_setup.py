"""
test_log_setup.py —— 日志配置测试

日志是这个项目的一条硬性要求（"该写日志的地方要写日志"），
所以它自己也得有测试：配好了没、会不会重复加 handler、写不写得进文件。
"""

import logging

import pytest

from app import log_setup


@pytest.fixture
def fresh_logging(monkeypatch):
    """
    把"已配置"标记重置，并清掉 root 上现有的 handler。

    # 不这么做的话，测试顺序会互相影响 ——
    # setup_logging() 是幂等的，第二次调用直接 return，
    # 于是"验证它加了 handler"的用例在别的用例之后跑就会失败。
    # 测试之间不能有隐藏的先后依赖。
    """
    monkeypatch.setattr(log_setup, "_configured", False)
    root = logging.getLogger()
    originals = list(root.handlers)
    root.handlers.clear()
    yield root
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()
    for h in originals:
        root.addHandler(h)


def test_setup_adds_console_and_file_handler(fresh_logging):
    log_setup.setup_logging()

    kinds = {type(h).__name__ for h in fresh_logging.handlers}
    assert "StreamHandler" in kinds
    assert "RotatingFileHandler" in kinds


def test_setup_is_idempotent(fresh_logging):
    """
    调两次不能加两套 handler。

    # run.py、main.py、ingest.py 都会调一次 setup_logging()。
    # 没有幂等保护的话，一条日志会在控制台打印三遍、文件里写三行，
    # 排查问题时看日志像在看三重影。
    """
    log_setup.setup_logging()
    count_after_first = len(fresh_logging.handlers)

    log_setup.setup_logging()

    assert len(fresh_logging.handlers) == count_after_first


def test_log_level_follows_settings(fresh_logging):
    log_setup.setup_logging()

    # conftest 把 LOG_LEVEL 设成了 WARNING
    assert fresh_logging.level == logging.WARNING


def test_noisy_loggers_are_suppressed(fresh_logging):
    """第三方库的 DEBUG 日志不能把业务日志淹掉。"""
    log_setup.setup_logging()

    for name in log_setup._NOISY_LOGGERS:
        assert logging.getLogger(name).level >= logging.WARNING


def test_file_handler_writes_and_is_utf8(fresh_logging):
    """真写一条中文日志进文件，验证编码没配错（Windows 默认 GBK 会乱码）。"""
    log_setup.setup_logging()

    logger = logging.getLogger("test.log_setup")
    logger.warning("中文日志写入验证：委托单 1 已接收")
    for h in fresh_logging.handlers:
        h.flush()

    content = log_setup.LOG_FILE.read_text(encoding="utf-8")
    assert "中文日志写入验证" in content


def test_setup_does_not_crash_when_dir_unwritable(fresh_logging, monkeypatch):
    """
    日志目录建不出来时，绝不能把服务拖崩。

    # 日志是辅助设施，不能成为故障源。
    # 部署到只读目录、或者磁盘满的时候，正确行为是"降级为只输出控制台"，
    # 而不是 ImportError 级别的启动失败。
    """

    def _boom(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(log_setup.Path, "mkdir", _boom)

    log_setup.setup_logging()  # 不抛异常就算通过

    kinds = {type(h).__name__ for h in fresh_logging.handlers}
    assert "StreamHandler" in kinds
    assert "RotatingFileHandler" not in kinds
