"""
rag.py —— 知识库的混合检索（稠密向量 + 稀疏词项 + RRF 融合）

职责：
    把"用户的一句问题"变成"最相关的几段语料内容"，两路召回再融合：
      1. 问题编码（embedding.encode_both 一次出双头：稠密向量 + 稀疏词项权重）
      2. 稠密路：pgvector 余弦近邻（<=> + HNSW），top rag_candidate_k ——
         强在语义泛化（"多久出结果"能召回"检测周期"）
      3. 稀疏路：chunk_sparse_terms 倒排表与查询词项做点积聚合，top rag_candidate_k ——
         强在精确标识符（样本号、基因符号、数字这类 embedding 近乎无语义的 token）
      4. RRF（Reciprocal Rank Fusion）融合两路排名，取 top_k
    向量与词项的入库（回填）不在本文件，在 ingest.py —— 入库是一次性
    运维动作，检索是每次问答都要跑的在线路径，两者的失败处理策略完全不同。

不负责：
    向量化本身（embedding.py）、怎么把结果交给模型（tools.py 的
    search_knowledge 工具）、切块规则（种子期已完成）。

为什么稀疏路失败只降级不报错：
    两路是"互相补台"的关系而不是串联依赖 —— 稀疏表还没灌、或 SQL
    出任何问题时，稠密路单独工作仍能给出合理结果。这与项目"工具失败
    不抛异常、统一降级"的既有哲学一致；降级必须打 WARNING，因为
    静默降级等于让"混合检索"悄悄变成单路而无人知晓。
"""

import logging

import numpy as np
from psycopg.rows import dict_row

from app import embedding
from app.config import settings
from app.db import get_conn

logger = logging.getLogger(__name__)

# 稠密路 SQL：一次往返完成"按相似度排序 + 取前 k 条"。
# 难点（必须用 <=> 算符）：document_chunks 上的 HNSW 索引是按
# vector_cosine_ops 建的，只有查询算符同样是 <=> 时索引才会被使用；
# 换成 <->（欧氏距离）或 <#>（负内积），PG 会静默退化为全表扫描 ——
# 几百行时无感，库长到几十万行时会慢几个数量级，且没有任何报错提示。
_DENSE_SQL = """
SELECT c.id    AS chunk_id,
       d.title AS title,
       c.content AS content,
       1 - (c.embedding <=> %(qvec)s) AS score
FROM document_chunks c
JOIN knowledge_documents d ON d.id = c.document_id
WHERE d.status = 'ready'
ORDER BY c.embedding <=> %(qvec)s
LIMIT %(cand_k)s
"""

# 稀疏路 SQL：查询词项 unnest 成临时表，与倒排表按 token_id JOIN，
# 逐块聚合 Σ(查询权重 × 块权重) —— 这就是 SPLADE 式内积打分在
# 纯 SQL 里的等价实现。
# 为什么 JOIN unnest 而不是把词项拼成 OR 展开？
# 一条查询有几十个非零词项，OR 展开的 SQL 文本长度随词项数线性膨胀，
# 且无法做"乘权重再求和"的聚合；unnest 让词项以集合方式参与 JOIN，
# 倒排表上 term_id 的 btree 索引（idx_sparse_term）每次都能吃到。
_SPARSE_SQL = """
SELECT t.chunk_id AS chunk_id,
       d.title    AS title,
       c.content AS content,
       SUM(q.w * t.weight) AS score
FROM chunk_sparse_terms t
JOIN unnest(%(term_ids)s::int[], %(weights)s::float8[]) AS q(term_id, w)
  ON q.term_id = t.term_id
JOIN document_chunks c ON c.id = t.chunk_id
JOIN knowledge_documents d ON d.id = c.document_id
WHERE d.status = 'ready'
GROUP BY t.chunk_id, d.title, c.content
HAVING SUM(q.w * t.weight) > 0
ORDER BY score DESC
LIMIT %(cand_k)s
"""


def _dense_search(qvec: np.ndarray, cand_k: int) -> list[dict]:
    """稠密路：余弦近邻取前 cand_k 条（SQL 异常原样抛给调用方统一降级处理）。"""
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_DENSE_SQL, {"qvec": qvec, "cand_k": cand_k})
        return cur.fetchall()


def _sparse_search(term_weights: dict[int, float], cand_k: int) -> list[dict]:
    """
    稀疏路：查询词项与倒排表做点积聚合，取前 cand_k 条。

    Args:
        term_weights: 查询的 {token_id: weight}（embedding 编码出的稀疏头）。
        cand_k: 候选深度。

    Raises:
        psycopg 异常原样抛出 —— 稀疏路的降级决策（只 WARNING 不失败）
        统一收口在 search_chunks 里，本函数保持"纯粹的一路查询"。
    """
    # 词项先按权重降序排序：unnest 展开顺序不影响 JOIN 结果，
    # 但稳定的入参顺序让日志和 EXPLAIN 可复现，排查时不用猜
    items = sorted(term_weights.items(), key=lambda kv: -kv[1])
    term_ids = [tid for tid, _ in items]
    weights = [w for _, w in items]
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            _SPARSE_SQL,
            {"term_ids": term_ids, "weights": weights, "cand_k": cand_k},
        )
        return cur.fetchall()


def _rrf_fuse(
    dense_rows: list[dict],
    sparse_rows: list[dict],
    rrf_k: int,
    top_k: int,
) -> list[dict]:
    """
    RRF 融合两路排名：score(d) = Σ 1/(k + rank_i(d))，按融合分降序取 top_k。

    Args:
        dense_rows / sparse_rows: 两路各自的候选行（含 chunk_id/title/content/score）。
        rrf_k: RRF 常数（.env RAG_RRF_K，默认 60）。
        top_k: 融合后最终保留条数。

    Returns:
        [{"title": ..., "content": ..., "score": 融合分, "dense_score": 稠密相似度或 None}, ...]

    # 为什么用 RRF 而不是把两路分数加权相加？
    # 两路的分数量纲完全不可比：稠密是余弦相似度（0~1），稀疏是词项内积
    # （无上界、随语料长度变化）。加权融合必须先做分数归一化，而归一化
    # 方案要随语料分布重新标定 —— 需按实际语料分布标定。
    # RRF 只消费"排名"这一与量纲无关的信息，k=60 是业界通用值，天然免疫
    # 分数不可比问题。代价是丢弃了分差信息（第一名压倒性优势和险胜同权），
    # 对 top-3 级别的召回场景这个损失可忽略。
    """
    fused: dict[int, float] = {}
    meta: dict[int, dict] = {}
    for rank, row in enumerate(dense_rows, start=1):
        cid = row["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        meta[cid] = {
            "title": row["title"],
            "content": row["content"],
            "dense_score": float(row["score"]),
        }
    for rank, row in enumerate(sparse_rows, start=1):
        cid = row["chunk_id"]
        fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        if cid not in meta:  # 稀疏独有的块没有稠密相似度可记，如实留 None
            meta[cid] = {
                "title": row["title"],
                "content": row["content"],
                "dense_score": None,
            }
    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [{**meta[cid], "score": fscore} for cid, fscore in ranked]


def search_chunks(query: str, top_k: int | None = None) -> list[dict]:
    """
    在知识库里做混合检索，返回与 query 最相关的段落。

    Args:
        query: 用户问题（或模型提炼后的检索词）。空串/纯空白返回空列表。
        top_k: 最终取几条，None 时用 .env 的 RAG_TOP_K（默认 3）。

    Returns:
        按融合分降序的字典列表，每项形如
        {"title": "加急复检与退单政策", "content": "……",
         "score": 0.032（RRF 融合分，量纲无关）, "dense_score": 0.82 或 None}。
        两路都无结果、或稠密路全部低于 RAG_SCORE_THRESHOLD 时返回空列表 ——
        **返回空列表是正常业务结果**，含义是"知识库确实没有相关内容"，
        调用方（tools.search_knowledge）据此让模型如实告知，而不是把
        这里当成错误。

    Raises:
        RuntimeError: 向量化失败（模型没配好）。稠密路数据库异常由调用方
        （execute_tool 的兜底）转成 ToolResult.fail，这里不吞；
        稀疏路任何异常都在本函数内降级（WARNING + 只走稠密路）。

    # 为什么阈值过滤只作用于稠密路、放在融合之前？
    # RAG_SCORE_THRESHOLD 标定的是"余弦相似度低于多少视为不相关"，它只对
    # 稠密分有语义。稀疏分是词项内积，量纲随语料漂移，套用同一个阈值
    # 没有意义 —— 稀疏路能命中（查询里有语料里出现过的实词）本身就是
    # 强相关性证据，这恰是"样本号/基因名"类查询的召回保障。
    # 放在融合之前而非之后，是因为 RRF 之后分数已变成量纲无关的融合分，
    # 再想按稠密阈值过滤就拿不到原始相似度了。
    """
    query = (query or "").strip()
    if not query:
        return []

    k = top_k if top_k is not None else settings.rag_top_k
    cand_k = settings.rag_candidate_k
    # 一次前向同时出双头 —— 稠密路要向量、稀疏路要词项，分开编码等于算两遍
    qvec, sparse_list = embedding.encode_both([query])
    qvec = qvec[0]
    term_weights = sparse_list[0] if sparse_list else {}

    dense_rows = [
        r
        for r in _dense_search(qvec, cand_k)
        if float(r["score"]) >= settings.rag_score_threshold
    ]

    sparse_rows: list[dict] = []
    if term_weights:
        try:
            sparse_rows = _sparse_search(term_weights, cand_k)
        except Exception as exc:  # noqa: BLE001 —— 稀疏路是可降级的旁路，不拖垮主路
            logger.warning(
                "稀疏路检索失败，本轮降级为纯稠密检索 | query=%s err=%s: %s",
                query[:40],
                type(exc).__name__,
                exc,
            )
    else:
        logger.debug("查询无稀疏词项（空权重），本轮只走稠密路 | query=%s", query[:40])

    if not dense_rows and not sparse_rows:
        # 两路都空 → 大概率语料还没入库（阶段③ 之前的状态），这条 WARNING
        # 就是为这种情况留的定位线索
        logger.warning(
            "知识库检索双路空召回 | query=%s（语料是否已入库？）", query[:40]
        )
        return []

    hits = _rrf_fuse(dense_rows, sparse_rows, settings.rag_rrf_k, k)
    logger.info(
        "知识库混合检索 | query=%s 稠密过阈值=%d 稀疏命中=%d 融合后=%d "
        "最高稠密分=%s",
        query[:40],
        len(dense_rows),
        len(sparse_rows),
        len(hits),
        f"{float(dense_rows[0]['score']):.3f}" if dense_rows else "N/A",
    )
    return hits
