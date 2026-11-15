"""
test_rag.py —— 混合检索（稠密 + 稀疏 + RRF）的单元测试

策略：
    检索 SQL 的"正确性"（<=> 算符、unnest 倒排、状态过滤）用假连接喂给
    search_chunks，断言真正发出去的 SQL 文本与参数 —— SQL 是检索的
    核心资产，值得逐字钉住；RRF 融合是纯函数，直接构造两路排名测组合行为。
    真实向量检索（需要真 bge-m3）不在单测范围，语料入库后手动验证。

# 为什么阈值/降级行为值得单测？
    这两条是"混合检索不能悄悄坏掉"的守门逻辑：阈值只作用稠密路
    （稀疏分是词项内积，量纲随语料漂移，套同一阈值无意义）；
    稀疏路失败只降级不报错（两路是补台关系，不是串联依赖）。
    谁改坏了其中一条，检索质量都会无声劣化 —— 必须有测试盯着。
"""

from contextlib import contextmanager

import numpy as np
import pytest

from app import rag


def _row(cid, title, score):
    """构造一路检索返回的行（chunk_id/title/content/score 四键齐全）。"""
    return {"chunk_id": cid, "title": title, "content": f"内容{title}", "score": score}


@pytest.fixture
def patch_rag_deps(monkeypatch):
    """
    把 search_chunks 的外部依赖换成可控假件：
      - embedding.encode_both → 固定向量 + 固定稀疏词项（两路都会尝试）
      - _dense_search / _sparse_search → 用例自己决定各路返回什么
    返回 install(dense_rows, sparse_rows) 函数。
    """

    def _install(dense_rows, sparse_rows):
        monkeypatch.setattr(
            rag.embedding,
            "encode_both",
            lambda texts: (np.zeros((1, 1024), dtype=np.float32), [{7: 0.3}]),
        )
        monkeypatch.setattr(rag, "_dense_search", lambda qvec, cand_k: dense_rows)
        monkeypatch.setattr(rag, "_sparse_search", lambda terms, cand_k: sparse_rows)

    return _install


def test_blank_query_returns_empty_without_db():
    """空/纯空白 query 直接返回 []，连编码和数据库都不碰（省一次往返）。"""
    assert rag.search_chunks("") == []
    assert rag.search_chunks("   ") == []


def test_threshold_filters_low_dense_scores(patch_rag_deps):
    """
    稠密分低于 RAG_SCORE_THRESHOLD 的命中被丢弃（融合之前过滤）。

    # 这是"宁可丢弃不硬答"的落地处 ——
    # 候选里可能全是 0.3 分的不相关段落，不过滤的话模型会被喂一堆噪音。
    # 稀疏路此处置空，专测稠密阈值这一件事。
    """
    patch_rag_deps(
        [
            _row(1, "加急政策", 0.82),
            _row(2, "无关段落", 0.45),   # 低于 0.5
            _row(3, "报告解读", 0.51),
        ],
        [],
    )

    hits = rag.search_chunks("多久能出结果")

    assert [h["title"] for h in hits] == ["加急政策", "报告解读"]


def test_all_below_threshold_returns_empty(patch_rag_deps):
    """稠密全部低于阈值且稀疏无命中 → 空列表是正常业务结果。"""
    patch_rag_deps([_row(1, "无关段落", 0.30)], [])

    assert rag.search_chunks("随便什么问题") == []


def test_rrf_fuse_prefers_double_hit_over_dense_leader():
    """
    RRF 融合：双路命中的块反超稠密分更高的块。

    # 这正是混合检索的价值所在 —— 稠密路把"语义最像"排在前面，
    # 但精确命中关键词的块（稀疏路）应该把名次拉上来。加权分数融合
    # 做不到这件事（两路分数量纲不可比），RRF 只看排名，天然免疫。
    """
    dense_rows = [_row(1, "A", 0.614), _row(2, "B", 0.592)]
    sparse_rows = [_row(2, "B", 1.20), _row(3, "C", 0.80)]

    fused = rag._rrf_fuse(dense_rows, sparse_rows, rrf_k=60, top_k=3)

    # B 双路命中（1/61 + 1/61）压过 A 的单路 1/61；C 稀疏独有排最后
    assert [h["title"] for h in fused] == ["B", "A", "C"]
    by_title = {h["title"]: h for h in fused}
    assert by_title["B"]["dense_score"] == 0.592       # 双路块保留稠密分
    assert by_title["C"]["dense_score"] is None       # 稀疏独有块如实留 None


def test_rrf_fuse_respects_top_k():
    dense_rows = [_row(i, f"D{i}", 0.9 - i * 0.1) for i in range(5)]
    fused = rag._rrf_fuse(dense_rows, [], rrf_k=60, top_k=2)

    assert len(fused) == 2


def test_sparse_failure_degrades_to_dense(monkeypatch, caplog):
    """
    稀疏路抛异常 → 降级纯稠密，打 WARNING，不拖垮检索。

    # 静默降级等于让"混合检索"悄悄变单路而无人知晓 ——
    # 警告日志是这条降级路径的强制伴生物，必须一起断言。
    """
    monkeypatch.setattr(
        rag.embedding,
        "encode_both",
        lambda texts: (np.zeros((1, 1024), dtype=np.float32), [{7: 0.3}]),
    )
    monkeypatch.setattr(rag, "_dense_search", lambda qvec, cand_k: [_row(1, "加急政策", 0.82)])

    def _boom(terms, cand_k):
        raise RuntimeError("倒排表还不存在")

    monkeypatch.setattr(rag, "_sparse_search", _boom)

    hits = rag.search_chunks("多久能出结果")

    assert [h["title"] for h in hits] == ["加急政策"]
    assert "稀疏路检索失败" in caplog.text


def test_sql_uses_cosine_operator_and_unnest_join(monkeypatch):
    """
    稠密路 SQL 必须用 <=>（余弦距离），稀疏路 SQL 必须用 unnest 倒排。

    # document_chunks 的 HNSW 索引按 vector_cosine_ops 建立，
    # 只有 <=> 查询才走索引；有人手滑改成 <-> 会静默退化为全表扫描
    # （数据量小时完全无感，量大会慢几十倍）。SQL 原文钉死在断言里。
    """

    class _FakeCursor:
        def __init__(self, recorder):
            self._recorder = recorder

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, sql, params=None):
            self._recorder.append(sql)

        def fetchall(self):
            return []

    class _FakeConn:
        def __init__(self):
            self.sqls: list[str] = []

        def cursor(self, row_factory=None):
            return _FakeCursor(self.sqls)

    fake_conn = _FakeConn()

    @contextmanager
    def _fake_get_conn():
        yield fake_conn

    monkeypatch.setattr(
        rag.embedding,
        "encode_both",
        lambda texts: (np.zeros((1, 1024), dtype=np.float32), [{7: 0.3}]),
    )
    monkeypatch.setattr(rag, "get_conn", _fake_get_conn, raising=True)

    rag.search_chunks("问题")

    joined = "\n---\n".join(fake_conn.sqls)
    assert "<=>" in joined                      # 稠密路余弦算符（走 HNSW）
    assert "d.status = 'ready'" in joined       # 未就绪文档不参与召回
    assert "unnest" in joined                   # 稀疏路倒排点积


def test_search_chunks_passes_candidate_k(patch_rag_deps, monkeypatch):
    """候选深度用 .env 的 RAG_CANDIDATE_K（两路共用），不是最终 top_k。"""
    seen = {}

    def _dense(qvec, cand_k):
        seen["dense"] = cand_k
        return []

    def _sparse(terms, cand_k):
        seen["sparse"] = cand_k
        return []

    monkeypatch.setattr(
        rag.embedding,
        "encode_both",
        lambda texts: (np.zeros((1, 1024), dtype=np.float32), [{7: 0.3}]),
    )
    monkeypatch.setattr(rag, "_dense_search", _dense)
    monkeypatch.setattr(rag, "_sparse_search", _sparse)

    rag.search_chunks("问题", top_k=2)

    from app.config import settings

    assert seen["dense"] == settings.rag_candidate_k
    assert seen["sparse"] == settings.rag_candidate_k
