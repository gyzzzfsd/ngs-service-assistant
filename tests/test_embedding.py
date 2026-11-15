"""
test_embedding.py —— 向量化模块的单元测试

策略：
    真模型（2.27GB）绝不在测试里加载（conftest 有哨兵拦截），
    这里注入一个假 BGEM3FlagModel，专测：
      - 空列表不触发模型加载
      - batch_size / return_dense / return_sparse 参数确实传给了模型
      - 稠密输出的归一化与 float32 转换（bge-m3 实测输出 float16）
      - 稀疏词项的类型清洗（实测 key 是 str、value 是 np.float16）
      - 单例缓存与 reset
      - 路径配置错误的报错行为
"""

import numpy as np
import pytest

from app import embedding
from app.config import settings


class FakeModel:
    """
    假 BGEM3FlagModel：记录每次 encode 的入参，按开关返回双头输出。

    故意模拟实测到的两个"脏"点（真实模型的行为）：
      - dense_vecs 是 float16 且未归一化（考验 _dense_matrix 的转换）
      - lexical_weights 的 key 是 str、value 是 np.float16（考验 _sparse_dicts）
    """

    def __init__(self, dim=1024):
        self.dim = dim
        self.calls: list[dict] = []

    def encode(self, texts, batch_size=None, return_dense=True, return_sparse=False, **kwargs):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "return_dense": return_dense,
                "return_sparse": return_sparse,
            }
        )
        out = {}
        if return_dense:
            # float16 + 故意不归一（norm != 1），看 embedding 模块是否自己兜住
            out["dense_vecs"] = (
                np.arange(len(texts) * self.dim, dtype=np.float16)
                .reshape(len(texts), self.dim)
                + 1.0
            )
        if return_sparse:
            out["lexical_weights"] = [
                {"1234": np.float16(0.5), "5678": np.float16(0.25)} for _ in texts
            ]
        return out


@pytest.fixture
def fake_model():
    """注入假模型（覆盖 conftest 的哨兵），并保证用例结束后还原。"""
    m = FakeModel()
    embedding._model = m
    yield m
    embedding._model = None


def test_encode_empty_list_skips_model():
    """空列表直接返回 (0, dim)，不触发模型加载 —— conftest 哨兵在，没报错就说明没碰模型。"""
    out = embedding.encode([])

    assert out.shape == (0, settings.embedding_dim)
    assert out.dtype == np.float32


def test_encode_calls_model_with_dense_only_flags(fake_model):
    """
    稠密单路编码：return_dense=True / return_sparse=False 必须真的传给模型。

    # 这两个开关不是可有可无 —— 检索侧只要稠密向量时把稀疏头的
    # 后处理开销也背上，等于每次问答白算一遍词项权重。
    """
    embedding.encode(["问题一", "问题二"])

    assert len(fake_model.calls) == 1
    call = fake_model.calls[0]
    assert call["texts"] == ["问题一", "问题二"]
    assert call["return_dense"] is True
    assert call["return_sparse"] is False
    assert call["batch_size"] == settings.embedding_batch_size


def test_encode_output_is_float32_and_normalized(fake_model):
    """
    稠密输出必须转 float32 并归一化。

    # 实测 bge-m3 的稠密头输出 dtype 是 float16 —— pgvector 的
    # 适配器只认 float32，不转会在写入时报类型错。归一化同理：
    # <=> 是余弦距离，只有归一化向量之间的 <=> 才有那个几何意义，
    # 不归一化的阈值标定全部失效。假模型故意返回未归一的 float16，
    # 就是为了证明这两步在本模块里兜住了，而不是恰好靠模型行为。
    """
    out = embedding.encode(["问题一"])

    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_sparse_cleans_key_and_value_types(fake_model):
    """
    稀疏词项的类型清洗：str key / np.float16 value 必须转成 int / float。

    # 实测 FlagEmbedding 返回的 lexical_weights 是
    # defaultdict{str: np.float16} —— psycopg 不认识这两种类型，
    # 直接进 INT[] 会 SQL 报错。清洗收口在 embedding 一处，
    # rag/ingest 拿到的必须是干净的 {int: float}。
    """
    result = embedding.encode_sparse(["问题一"])

    assert result == [{1234: 0.5, 5678: 0.25}]
    assert all(isinstance(k, int) for k in result[0])
    assert all(isinstance(v, float) for v in result[0].values())


def test_encode_both_requests_two_heads_in_one_call(fake_model):
    """
    encode_both 必须一次模型调用同时拿双头。

    # 两次调用 = 两次完整 GPU 前向，几百条切块入库耗时直接翻倍。
    # 假模型会记录调用次数 —— 多于 1 次就说明实现退化成了"连调两次"。
    """
    dense, sparse = embedding.encode_both(["样本一", "样本二"])

    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["return_dense"] is True
    assert fake_model.calls[0]["return_sparse"] is True
    assert dense.shape == (2, settings.embedding_dim)
    assert dense.dtype == np.float32
    assert len(sparse) == 2


def test_encode_both_empty_list_is_noop():
    dense, sparse = embedding.encode_both([])

    assert dense.shape == (0, settings.embedding_dim)
    assert sparse == []


def test_encode_one_returns_single_vector(fake_model):
    """encode_one 返回一维向量（形状 (dim,)），供 pgvector 直接使用。"""
    vec = embedding.encode_one("单个问题")

    assert vec.shape == (settings.embedding_dim,)
    assert vec.dtype == np.float32


def test_get_model_returns_singleton(fake_model):
    """模型只加载一次：get_model 两次调用拿到同一个实例。"""
    assert embedding.get_model() is fake_model
    assert embedding.get_model() is fake_model


def test_get_model_raises_when_path_missing(monkeypatch):
    """EMBEDDING_MODEL_PATH 未配置时快失败，错误信息指向 .env。"""
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setattr(settings, "embedding_model_path", "")

    with pytest.raises(RuntimeError, match="EMBEDDING_MODEL_PATH"):
        embedding.get_model()


def test_reset_model_clears_singleton(fake_model):
    """reset_model 后 get_model 会重新走加载路径（测试间互不污染的保障）。"""
    embedding.reset_model()

    assert embedding._model is None
