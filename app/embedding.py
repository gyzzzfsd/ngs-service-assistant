"""
embedding.py —— 本地 bge-m3 模型的加载与双头文本编码（稠密 + 稀疏）

职责：
    1. 进程内只加载一次 bge-m3（单例），谁先用到谁触发，之后的调用直接复用
    2. 对外三个编码入口：
       - encode(texts)      稠密：一批文本 → (n, 1024) float32 归一化矩阵
       - encode_sparse(texts) 稀疏：一批文本 → 每条一个 {token_id: weight} 字典（检索用）
       - encode_both(texts)  双头：一次模型调用同时出稠密+稀疏（入库用，省一半 GPU 前向）

不负责：
    检索逻辑与 RRF 融合（rag.py）、向量怎么入库（ingest.py）、配置从哪来（config.py）。

为什么换成 FlagEmbedding 的 BGEM3FlagModel 而不是原 SentenceTransformer：
    bge-m3 是三头模型（dense / sparse / colbert），SentenceTransformer 只会用它的
    稠密头 —— 稀疏头（sparse_linear.pt，学习出来的词项权重，类 SPLADE）是模型
    自带的、已随权重文件在本地，换加载器即可激活，零额外下载。混合检索的两路
    输入都从这一个模型出，天然同源，不存在"稠密稀疏两套语义空间"的对齐问题。

为什么做成懒加载单例而不是启动时就加载：
    bge-m3 有 2.27GB，加载到 GPU 需要数秒到十几秒。如果服务一启动就加载，
    那么只问委托单进度的用户也要等这一下；而懒加载把成本推迟到
    "第一个问知识库问题的人"，其余请求完全无感。
    代价是第一次知识库检索会慢一些（日志里会打 WARNING 提醒），
    这是服务型部署里常见的取舍。

线程安全说明：
    FastAPI 的同步路由跑在线程池里，多个请求可能同时进来。
    这里用一把简单的锁保证"加载"只发生一次 —— 加载完成后读 _model
    不加锁（CPython 读取属性是原子的；BGEM3FlagModel.encode 本身
    可并发调用，内部有批处理保护）。
"""

import logging
import threading
from typing import TYPE_CHECKING

import numpy as np

from app.config import settings

if TYPE_CHECKING:  # 只在类型检查时导入，运行时不依赖 FlagEmbedding（测试环境可替换）
    from FlagEmbedding import BGEM3FlagModel

logger = logging.getLogger(__name__)

_model: "BGEM3FlagModel | None" = None
_load_lock = threading.Lock()


def get_model() -> "BGEM3FlagModel":
    """
    返回进程内唯一的 bge-m3 模型实例，第一次调用时才真正加载。

    Returns:
        已加载到 EMBEDDING_DEVICE（cuda/cpu）的 BGEM3FlagModel 实例。

    Raises:
        RuntimeError: 模型路径不存在（.env 里 EMBEDDING_MODEL_PATH 配错），
            或显卡/CPU 不可用导致加载失败。异常会带上下文，方便定位是
            "路径错"还是"环境错"。

    # 为什么加载失败要让整个请求失败（抛异常），而不是退回 CPU 再试？
    # 捕获 CUDA 加载失败自动降级 CPU 看似更健壮，实则掩盖配置问题：
    # 用户在 .env 里写 cuda，实际跑在 CPU 上编码几百条要几分钟，
    # 检索延迟劣化 10 倍却没有任何报错，排查起来比直接失败难得多。
    # 所以这里选择"快失败 + 日志说清原因"，让运维自己决定要不要降级。
    """
    global _model
    if _model is not None:
        return _model

    with _load_lock:  # 双重检查：等锁的第二个线程进来时，第一个已经加载完
        if _model is not None:
            return _model

        if not settings.embedding_model_path:
            raise RuntimeError(
                "EMBEDDING_MODEL_PATH 未配置：请检查 .env 的 Embedding 段"
            )

        import time

        from FlagEmbedding import BGEM3FlagModel  # 局部导入：不用向量时连库都不用加载

        started = time.time()
        logger.info(
            "加载 bge-m3 模型（双头） | path=%s device=%s（首次调用约需 5~20 秒）",
            settings.embedding_model_path,
            settings.embedding_device,
        )
        # use_fp16 只在 GPU 上开。半精度在 CPU 上无加速收益且易数值抖动，
        # 设备字符串直接透传给 FlagEmbedding（"cuda" / "cpu"），不做二次解析。
        use_fp16 = settings.embedding_device.lower().startswith("cuda")
        try:
            _model = BGEM3FlagModel(
                settings.embedding_model_path,
                use_fp16=use_fp16,
                devices=settings.embedding_device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"bge-m3 加载失败（path={settings.embedding_model_path}, "
                f"device={settings.embedding_device}）：{exc}"
            ) from exc
        logger.info("bge-m3 加载完成，耗时 %.1fs", time.time() - started)
        return _model


def _dense_matrix(out: dict) -> np.ndarray:
    """
    把 BGEM3FlagModel.encode 的返回整理成 pgvector 友好的稠密矩阵。

    Args:
        out: model.encode(...) 的返回 dict，取其 "dense_vecs"。

    Returns:
        形状 (n, EMBEDDING_DIM) 的 float32 矩阵，每行 L2 归一化。

    # bge-m3 的稠密头天然输出归一化向量（对比训练的产物），实测 norm=1；
    # 但 dtype 是 float16 —— pgvector 的适配器只认 float32，float16 直接写入
    # 会报类型错。所以这里必须 astype(float32)。
    # EMBEDDING_NORMALIZE=true 时再做一次显式归一化：对已归一的向量是幂等操作，
    # 看似多余，但它把"归一化责任"留在本函数而不是依赖模型实现的内部行为 ——
    # 将来换模型（输出未归一）时阈值标定不会被悄悄破坏。
    """
    dense = np.asarray(out["dense_vecs"], dtype=np.float32)
    if settings.embedding_normalize:
        norms = np.linalg.norm(dense, axis=1, keepdims=True)
        # 零向量保护：除零会产生 nan 污染整列，长文本极端截断时理论上可出现
        safe = np.where(norms == 0.0, 1.0, norms)
        dense = dense / safe
    return dense


def _sparse_dicts(out: dict) -> list[dict[int, float]]:
    """
    把 BGEM3FlagModel.encode 的稀疏头输出整理成纯 Python 字典列表。

    Args:
        out: model.encode(...) 的返回 dict，取其 "lexical_weights"。

    Returns:
        每条文本对应一个 {token_id(int): weight(float)} 字典。

    # 为什么必须在这里做类型清洗，而不是让下游各自处理？
    # 实测 FlagEmbedding 返回的 lexical_weights 是 defaultdict{str: np.float16}：
    #   - key 是字符串数字 —— PostgreSQL 的 INT[] / unnest 只认 Python int，
    #     psycopg 拿到 str 会按文本数组发送，JOIN 类型不匹配直接 SQL 报错；
    #   - value 是 np.float16 —— psycopg 同样不认识这个 numpy 标量类型。
    # 类型清洗与模型实现强耦合，属于"编码"职责的一部分，放在这里一处收口，
    # rag.py（查询侧）和 ingest.py（入库侧）拿到的都是干净的 {int: float}。
    """
    return [
        {int(k): float(v) for k, v in lw.items()}
        for lw in out["lexical_weights"]
    ]


def encode(texts: list[str]) -> np.ndarray:
    """
    把一批文本编码成稠密向量（同步阻塞，跑在 FastAPI 的线程池里）。

    Args:
        texts: 文本列表。空列表直接返回空数组，不触发模型加载。

    Returns:
        形状 (len(texts), 1024) 的 float32 矩阵，每行已 L2 归一化。

    Raises:
        RuntimeError: 模型加载失败（见 get_model）。
    """
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32)

    model = get_model()
    out = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=True,
        return_sparse=False,
    )
    return _dense_matrix(out)


def encode_sparse(texts: list[str]) -> list[dict[int, float]]:
    """
    把一批文本编码成稀疏词项权重（检索的稀疏路用）。

    Args:
        texts: 文本列表。空列表返回空列表，不触发模型加载。

    Returns:
        每条文本一个 {token_id: weight} 字典，token_id 是 bge-m3 词表 id，
        weight 是学习出来的词项权重（常见词低、领域专名高）。

    Raises:
        RuntimeError: 模型加载失败（见 get_model）。
    """
    if not texts:
        return []

    model = get_model()
    out = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=False,
        return_sparse=True,
    )
    return _sparse_dicts(out)


def encode_both(texts: list[str]) -> tuple[np.ndarray, list[dict[int, float]]]:
    """
    一次模型调用同时返回稠密矩阵和稀疏词项（入库用）。

    Args:
        texts: 文本列表。空列表返回 ((0, dim) 空矩阵, [])。

    Returns:
        (dense_matrix, sparse_dicts) 二元组，元素规格与 encode / encode_sparse
        完全一致 —— 调用方（ingest.py）不需要知道两路是同一次前向算出来的。

    Raises:
        RuntimeError: 模型加载失败（见 get_model）。

    # 为什么要单设 encode_both 而不是让调用方连调 encode + encode_sparse？
    # 两次调用 = 两次完整 GPU 前向，几百条切块入库时耗时直接翻倍。
    # bge-m3 一次前向天然产出双头，本函数就是"一次前向、两个出口"的封装 ——
    # 稠密检索单路时（encode）才关掉稀疏头省那部分后处理开销。
    """
    if not texts:
        return np.zeros((0, settings.embedding_dim), dtype=np.float32), []

    model = get_model()
    out = model.encode(
        texts,
        batch_size=settings.embedding_batch_size,
        return_dense=True,
        return_sparse=True,
    )
    return _dense_matrix(out), _sparse_dicts(out)


def encode_one(text: str) -> np.ndarray:
    """
    单条文本便捷入口（检索时用）。语义等价于 encode([text])[0]。

    Args:
        text: 一条文本。

    Returns:
        形状 (1024,) 的一维向量，归一化与否与 encode 保持一致。

    # 为什么单独包一个函数而不是让调用方自己 encode([q])[0]？
    # 两个原因：一是一维和二维数组在 pgvector 适配、余弦计算里的行为不同
    # （二维要 reshape），调用方容易踩坑；二是"单条"这个语义在检索侧
    # 出现频率极高，有个名字可读性明显更好。
    """
    return encode([text])[0]


def reset_model() -> None:
    """
    丢弃已加载的模型实例（测试专用）。

    正常业务代码永远用不到它；单测需要在"装了假模型"和"没装模型"
    两种状态间切换时调用，保证用例之间互不污染。
    """
    global _model
    with _load_lock:
        _model = None
