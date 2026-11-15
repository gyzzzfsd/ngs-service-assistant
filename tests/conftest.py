"""
conftest.py —— pytest 公共夹具

三条设计原则：
    1. 测试跑在真实数据库上（本项目没有内存替身，SQL 是核心逻辑，mock 掉就没意义了）。
    2. 但**绝不留下任何数据**：整场测试共用一条连接，每个用例前后用 SAVEPOINT 回滚。
    3. 不连大模型：LLM 调用统一在用例里 monkeypatch，保证快、免费、结果稳定。
       需要真调模型的用例单独标记 integration，默认跳过。

数据库隔离（2026-11-12 起生效，2026-11-15 随项目改名单库）：
    测试连的是**专用库** ngs_service_assistant_test，不是开发库。
    此前测试直接跑在开发库上，虽然不留数据，但用例会读库里已提交的状态
    （"林教授还没有会话"之类），手工在页面上聊两句就会把测试弄红 ——
    数据库是拿来用的，不是拿来看的，所以让测试搬进自己的库。
    测试库由 tests/setup_test_db.py 用 ngs_schema.sql + seed_ngs.sql 重建，
    正常情况永远不用重跑；种子数据大改后想同步时再跑一次。
"""

import logging
import os
from pathlib import Path

# 必须在 import app.config 之前设置，否则 .env 里的 INFO 会让测试输出被日志淹没。
# pydantic-settings 的优先级是"环境变量 > .env 文件"，所以这里能覆盖。
os.environ.setdefault("LOG_LEVEL", "WARNING")


def _test_database_url() -> str:
    """
    从 .env 读出开发库连接串，把库名换成 ngs_service_assistant_test。

    为什么不写死连接串：账号/密码/主机与开发库必须永远一致，
    写两份迟早漂移；读 .env 只换库名，改密码时这里自动跟着走。
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("DATABASE_URL="):
            dsn = line.split("=", 1)[1].strip()
            return dsn.rsplit("/", 1)[0] + "/ngs_service_assistant_test"
    raise RuntimeError(".env 里找不到 DATABASE_URL，无法定位测试库连接串")


# 必须在 import app.config 之前设置 —— settings 是 lru_cache，先到先得
os.environ["DATABASE_URL"] = _test_database_url()

from contextlib import contextmanager  # noqa: E402

import psycopg  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app import (
    api_actions,
    api_auth,
    api_chat,
    api_handoff,
    db,
    embedding,
    ingest,
    main,
    rag,
    run_log,
    tools,  # noqa: E402
)
from app.config import settings  # noqa: E402


@pytest.fixture(scope="session")
def raw_conn():
    """
    整场测试共用的一条连接（autocommit=False）。

    为什么共用一条而不是每个用例连一次：
        建连接约 10ms，几十个用例就是几百毫秒的纯浪费；
        更重要的是，共用一条连接才能让"整场测试最后统一 rollback"成为可能 ——
        这就是下面用例级 SAVEPOINT 的基础。

    结束后一定 rollback：测试期间插入的会话、消息、委托单状态变更全部作废，
    数据库回到测试前的样子。所以测试可以放心地跑，不用手工清理数据。

    Raises:
        RuntimeError: 测试库不存在（多半被误删）—— 直接告诉用户跑哪条命令
        重建，比让 psycopg 的连接报错自己猜要友好得多。
    """
    try:
        conn = psycopg.connect(settings.psycopg_dsn, autocommit=False)
    except psycopg.OperationalError as e:
        if "ngs_service_assistant_test" in str(e) and "does not exist" in str(e):
            raise RuntimeError(
                "测试库 ngs_service_assistant_test 不存在。请在项目根目录执行：\n"
                "  python tests/setup_test_db.py\n"
                "它会在服务器上用 ngs_schema.sql + seed_ngs.sql 重建测试库。"
            ) from e
        raise
    # 先执行一条语句把事务显式开起来 —— 没有活动事务时 SAVEPOINT 会直接报错
    conn.execute("SELECT 1")
    try:
        yield conn
    finally:
        conn.rollback()
        conn.close()


@pytest.fixture(autouse=True)
def db_isolation(raw_conn, monkeypatch):
    """
    自动套在每个用例上：把各模块的 get_conn 换成"永远返回同一条事务连接"，
    并给每个用例打一个 SAVEPOINT，用例结束回滚到该点。

    # 为什么要 monkeypatch 四个模块的 get_conn，而不是只改 db 模块？
    # 因为各业务模块写的是 `from app.db import get_conn`，这个 import 在模块加载时
    # 就把函数对象绑到自己的命名空间了。只改 db.get_conn 对它们没有影响 ——
    # 它们手里还是原来那个函数，照样去连接池借连接、照样提交事务。
    # 必须逐个模块替换，测试数据才真的不会落库。

    回滚到 SAVEPOINT 而不是 ROLLBACK：
        用例抛异常时事务会进入 aborted 状态，此时普通 ROLLBACK 会把整场测试的
        前置数据也一起丢掉；ROLLBACK TO SAVEPOINT 只撤掉当前用例的改动，后续用例不受影响。
    """

    @contextmanager
    def _fake_get_conn():
        # 不 commit、不 close：交给最外层的 raw_conn 统一回滚
        yield raw_conn

    for module in (db, tools, api_auth, api_chat, api_actions, api_handoff, run_log, rag, ingest):
        monkeypatch.setattr(module, "get_conn", _fake_get_conn, raising=True)

    raw_conn.execute("SAVEPOINT test_case")
    try:
        yield
    finally:
        raw_conn.execute("ROLLBACK TO SAVEPOINT test_case")


@pytest.fixture
def client():
    """
    FastAPI 测试客户端。

    故意不用 `with TestClient(app)`：那会触发 lifespan，真的去建数据库连接池，
    而本套测试的连接已经通过 db_isolation 换成事务连接了，多建一个池纯属浪费。
    """
    return TestClient(main.app)


@pytest.fixture
def auth_headers(client):
    """陈医生（user_id=1，7 张委托单覆盖多数节点）的登录凭证。"""
    resp = client.post("/api/auth/login", json={"username": "chen", "password": "123456"})
    assert resp.status_code == 200, f"登录失败：{resp.text}"
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def lin_headers(client):
    """林教授（user_id=2，2 张委托单）的登录凭证，用于验证数据隔离。"""
    resp = client.post("/api/auth/login", json={"username": "lin", "password": "123456"})
    assert resp.status_code == 200, f"登录失败：{resp.text}"
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def wang_headers(client):
    """
    王医生（user_id=3，名下唯一委托单是 qc_failed）的登录凭证，
    复检场景（质检不合格无条件放行）的主测试账号。
    """
    resp = client.post("/api/auth/login", json={"username": "wang", "password": "123456"})
    assert resp.status_code == 200, f"登录失败：{resp.text}"
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def agent_headers(client):
    """
    技术支持坐席 tech001（user_id=6，role=agent）的登录凭证。
    走真实登录接口而不是直接 create_token：顺带验证"坐席账号能正常登录"。
    """
    resp = client.post("/api/auth/login", json={"username": "tech001", "password": "123456"})
    assert resp.status_code == 200, f"坐席登录失败：{resp.text}"
    assert resp.json()["user"]["role"] == "agent"
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


@pytest.fixture
def stub_agent(monkeypatch):
    """
    把 agent.run 换成可控的假函数（返回调用记录列表，可断言"AI 是否被调用过"）。

    # patch 的是 app.agent.run，不是 app.api_chat.agent.run。
    # api_chat 里写的是 `from app import agent`，拿到的是模块对象本身，
    # 调用时才会去 agent 上找 run —— 所以在模块对象上替换才生效。

    假的 AgentRun 由 test_api_chat._fake_run 构造（那里有完整的契约说明）。
    """
    from app import agent as agent_mod

    def _install(result):
        calls = []

        def _run(user_message, history, user_id, nickname=""):
            calls.append({"user_message": user_message, "history": history, "user_id": user_id})
            return result

        monkeypatch.setattr(agent_mod, "run", _run)
        return calls

    return _install


@pytest.fixture
def conversation_id(client, auth_headers) -> int:
    """为陈医生准备一个 AI 状态的会话（种子已有会话 1 则复用），返回会话 ID。"""
    resp = client.post("/api/conversations", headers=auth_headers)
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


@pytest.fixture(autouse=True)
def _quiet_logs(caplog):
    """
    用例里默认只捕获 WARNING 以上，避免每个用例都刷一堆 INFO。
    需要断言日志的用例自己在内部用 caplog.set_level(logging.INFO) 打开。
    """
    caplog.set_level(logging.WARNING)


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """
    安全网：任何没有被显式 mock 的路径，都不许真的去调大模型。

    # 为什么这条必须有？
    # 只要有一个用例忘了 mock，它就会真的打智谱接口：耗掉一次配额、
    # 慢 3~7 秒、还可能因为限流随机失败 —— 让整套测试变成"有时红有时绿"。
    # 更糟的是这种失败看起来像业务 bug，排查方向一开始就是错的。
    # 这里让客户端构造阶段直接抛错，等于把"漏 mock"变成一条明确的断言失败。

    真需要调模型的用例请单独标记 @pytest.mark.integration 并默认跳过。
    """

    def _blocked():
        raise RuntimeError("测试中禁止真实调用大模型，请先 mock app.agent.chat 或 agent.run")

    monkeypatch.setattr("app.llm.get_client", _blocked)


@pytest.fixture(autouse=True)
def _no_real_embedding():
    """
    安全网：测试中禁止真实加载 bge-m3（2.27GB、5 秒起），并保证用例间
    模型单例不互相污染。

    # 为什么不像 _no_real_llm 那样拦 get_model 函数？
    # 因为懒加载单例有个特点：一旦某个用例注入了假模型，后续用例的
    # get_model 都会命中这个假模型 —— 单例的污染是跨用例的。
    # 所以这里在**每个用例前**都把 _model 换成一个"encode 就报错"的哨兵：
    #   - 普通用例：一旦有代码路径真的走到了向量化，立刻拿到明确报错
    #     （"请在测试里注入假模型"），而不是静默加载真模型拖慢整套测试；
    #   - 需要假模型的用例：在用例体内直接覆盖 embedding._model，
    #     用例结束后 finally 又恢复成 None，下个用例从干净状态开始。
    """

    class _BlockedModel:
        def encode(self, *args, **kwargs):
            raise RuntimeError(
                "测试中禁止真实加载 bge-m3：需要向量化的用例请注入假模型 "
                "（monkeypatch.setattr(embedding, '_model', FakeModel())）"
            )

    embedding._model = _BlockedModel()
    try:
        yield
    finally:
        embedding._model = None
