"""
config.py —— 全局配置中心

职责：
    把项目根目录 .env 里的配置读成一个强类型的 Settings 对象（带类型转换和默认值）。
    全项目只有这一个文件接触环境变量，其他地方一律 `from app.config import settings`。

不负责：
    任何业务逻辑、数据库操作、网络请求。

为什么要有这个文件：
    如果把 os.getenv 散落在十几个文件里，改一个端口号要全局搜索，漏改一处就出事故。
    集中到一处后，"配置有哪些、默认值是什么"看这一个文件就够了。
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 = app/ 的上一级。
# 用绝对路径定位 .env，这样不管 uvicorn 从哪个目录启动，读到的都是同一份配置。
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    七组配置，字段名与 .env 的变量名一一对应（大小写不敏感，DATABASE_URL → database_url）。

    没有默认值的字段（database_url / llm_api_key / jwt_secret）是必填项：
    .env 里漏写会在启动瞬间抛 ValidationError，比跑到一半才炸好排查。
    """

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # .env 里多出来的键（比如以后加的调试项）不会让程序启动失败
        extra="ignore",
    )

    # ---------- 应用 ----------
    app_name: str = "AI Customer Service"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8010
    log_level: str = "INFO"

    # ---------- 数据库（hadoop100 上的 PG16 + pgvector）----------
    database_url: str
    db_echo: bool = False
    db_pool_size: int = 5

    # ---------- 大模型（智谱 GLM，OpenAI 兼容协议）----------
    llm_provider: str = "zhipu"
    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4"
    llm_api_key: str
    llm_model: str = "glm-4.5-air"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    llm_timeout: int = 60
    # glm-4.x 关掉思考可省 3~5 秒/轮；glm-5.3-flash 不支持关闭（会报 1210）
    llm_disable_thinking: bool = True
    llm_reasoning_effort: str = "low"
    # 主力连续失败时自动切备用模型，留空则关闭
    llm_fallback_model: str = ""

    # ---------- Embedding（本地 bge-m3，1024 维）----------
    embedding_backend: str = "local"
    embedding_model_path: str = ""
    embedding_dim: int = 1024
    embedding_device: str = "cuda"
    embedding_batch_size: int = 8
    embedding_normalize: bool = True
    embedding_rebuild: bool = False

    # ---------- RAG 检索 ----------
    rag_top_k: int = 3
    rag_score_threshold: float = 0.5
    # 混合检索：每路先各取多少候选（稠密/稀疏各自 top rag_candidate_k），再 RRF 融合
    rag_candidate_k: int = 10
    # RRF 常数 k：score = Σ 1/(k + rank)。60 是业界通用值，刻意不做可调 ——
    # 它只影响排名间衰减曲线的陡峭度，保持默认值即可
    rag_rrf_k: int = 60
    chunk_min_chars: int = 100
    chunk_max_chars: int = 420
    knowledge_dir: str = "./document"

    # ---------- Agent / Harness ----------
    # 一次问答最多允许多少轮工具调用，防止模型死循环烧 token
    agent_max_tool_rounds: int = 5
    # 校验失败后允许模型重写几次（1 = 首次生成 + 1 次重写）
    agent_max_correction: int = 1
    # 取最近多少条历史消息进 prompt
    agent_history_limit: int = 20

    # ---------- 认证 ----------
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 720

    # ---------- 前端 / 实时推送 ----------
    cors_origins: str = "http://localhost:5173"
    ws_path: str = "/ws"

    # ---------- 由原始配置派生出来的值 ----------

    @property
    def psycopg_dsn(self) -> str:
        """
        把 SQLAlchemy 风格的连接串转成 psycopg 能直接用的 DSN。

        # .env 里写的是 `postgresql+psycopg://`（SQLAlchemy 的方言前缀），
        # 而 psycopg.connect() 只认 `postgresql://`，多出来的 `+psycopg` 会报
        # "invalid connection string"。同一份 .env 被两边共用，
        # 所以在这里一次性剥掉前缀，而不是要求 .env 里写两份连接串。
        """
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")

    @property
    def cors_origin_list(self) -> list[str]:
        """把逗号分隔的字符串拆成列表，FastAPI 的 CORSMiddleware 只吃 list。"""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def knowledge_dir_path(self) -> Path:
        """.env 里的 ./document 是相对路径，这里统一转成"相对项目根"的绝对路径。"""
        path = Path(self.knowledge_dir)
        return path if path.is_absolute() else (BASE_DIR / path).resolve()


@lru_cache
def get_settings() -> Settings:
    """
    返回全局唯一的 Settings 实例（lru_cache 保证只解析一次 .env）。

    单测里改了环境变量想重新读时，调用 get_settings.cache_clear() 即可。
    """
    return Settings()


settings = get_settings()
