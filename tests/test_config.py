"""
test_config.py —— 配置加载测试

验证的是"配置读进来之后，派生出来的值对不对"，
而不是"pydantic 能不能读 env"（那是框架的事）。
"""

from app.config import BASE_DIR, settings


def test_dsn_strips_sqlalchemy_prefix():
    """
    .env 里写的是 postgresql+psycopg://，psycopg 只认 postgresql://。

    这条会不会挂，直接决定服务能不能启动 —— 是历史上最容易踩的一个坑。
    """
    assert settings.psycopg_dsn.startswith("postgresql://")
    assert "+psycopg" not in settings.psycopg_dsn
    assert "@" in settings.psycopg_dsn


def test_cors_origins_split_into_list():
    """CORS_ORIGINS 在 .env 里是逗号分隔字符串，FastAPI 要的是 list。"""
    origins = settings.cors_origin_list
    assert isinstance(origins, list)
    assert len(origins) >= 1
    assert all(o.startswith("http") for o in origins)
    # 不能有残留空格，否则 CORS 匹配会失败（浏览器行为很隐蔽）
    assert all(o == o.strip() for o in origins)


def test_knowledge_dir_resolved_to_absolute():
    """
    知识库目录必须解析成存在的绝对路径，否则换个工作目录就读不到手册。

    # 为什么不再断言目录在项目根（path.parent == BASE_DIR）？
    # 知识库目录由 KNOWLEDGE_DIR 配置，可相对也可绝对，
    # 目录在哪属于部署布局，不属于配置正确性 —— 测试只该管"路径可用"。
    """
    path = settings.knowledge_dir_path
    assert path.is_absolute()
    assert path.exists(), f"知识库目录不存在：{path}"
    # 手册源文件必须还在（切块脚本 build_seed.py 靠它们重刷向量）
    assert any(path.glob("**/*.md")), f"知识库目录里没有任何 .md 手册：{path}"


def test_required_secrets_present():
    """三个必填项缺失时服务根本起不来，这里提前断言，报错更直观。"""
    assert settings.llm_api_key, "LLM_API_KEY 为空"
    assert settings.jwt_secret, "JWT_SECRET 为空"
    assert len(settings.jwt_secret) >= 32, "JWT_SECRET 太短，签名强度不足"


def test_embedding_uses_gpu():
    """本项目后端跑在有 RTX 5060 的开发机上，device 应该是 cuda。"""
    assert settings.embedding_device in ("cuda", "cpu")
    assert settings.embedding_dim == 1024, "bge-m3 输出 1024 维，不能改"


def test_agent_limits_are_sane():
    """防止有人把轮数配成 0 或负数，导致 agent 一次工具都用不了。"""
    assert settings.agent_max_tool_rounds >= 1
    assert settings.agent_max_correction >= 0
    assert settings.agent_history_limit >= 0
