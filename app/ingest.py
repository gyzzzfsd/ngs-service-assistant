"""
ingest.py —— 语料入库（切块 + 双路索引回填）的命令行脚本

职责分两段，一条命令完成：
    1. 切块入库：扫描 KNOWLEDGE_DIR/docs/ 下的 *.md 语料，按 RAG 设计
       方案的规则切块，写入 knowledge_documents + document_chunks
       （embedding 留空，等第 2 段回填）。
    2. 双路编码：把全部知识切块用 bge-m3 编码，一次前向同时产出两路并写库：
       2.1 稠密路：1024 维归一化向量 → document_chunks.embedding（pgvector 检索用）
       2.2 稀疏路：词项权重 {token_id: weight} → chunk_sparse_terms 倒排表（混合检索用）
       这是语料入库期的一次性动作，之后每次改动语料内容都可以重跑。

用法（项目根目录）：
    python -m app.ingest
    加 --rebuild 参数强制全量重算（默认只补"新文档 + embedding 为 NULL
    或缺稀疏词项"的行，与 .env 的 EMBEDDING_REBUILD=true 等效，命令行参数优先）。

切块规则：
    1. 以「## 章节」为一级边界，章节内以空行分段；
    2. 段落 < chunk_min_chars（100 字）→ 与同章节下一段合并；
    3. 段落 > chunk_max_chars（420 字）→ 按「。！？；」切成 ~300 字的块；
    4. 每块内容前加《文档名》【章节名】前缀（标题参与向量化）。
    文档 # 主标题下的开头导语不单独成块（过短且无章节上下文）。

不负责：
    在线检索（rag.py）、测试库的语料（测试向量检索用例全部 mock，
    不依赖真实向量，见 tests/setup_test_db.py 说明）。

幂等性设计：
    默认"只补缺"使脚本可以放心重跑 —— 切块侧只插入 file_name 尚未入库
    的新文档（旧文档连同切块原样保留）；编码侧只补"embedding 为 NULL
    或缺稀疏词项"的行，中途断掉、失败若干行，下次运行自动接着补。
    全量重建模式先删全部文档（切块与倒排表级联清空）再重切重算，
    适合语料内容更新后重刷。

为什么两张存储必须同批同步写：
    稠密和稀疏共用同一次模型前向的输出。若分两次跑（先回填向量、
    再回填词项），中间任何一次失败都会造成"两路索引覆盖的切块集合不一致"——
    稀疏路命中一个块而它的 embedding 还是 NULL，检索结果两路都谈不上可信。
    同批写入 + 同批提交把两者绑定成一个原子单元，失败即整体留在"缺"状态，
    重跑时一起补齐。
"""

import logging
import re
import sys
from pathlib import Path

import numpy as np

from app import embedding
from app.config import settings
from app.db import close_pool, get_conn, init_pool
from app.log_setup import setup_logging

logger = logging.getLogger(__name__)

# 一次从库里取多少行做一批编码。值与 EMBEDDING_BATCH_SIZE 解耦：
# 编码批大小影响 GPU 上的吞吐，取行批大小影响内存占用和事务粒度，
# 两者没有理由绑在一起。几百条总量很小，64 是安全值。
_FETCH_BATCH = 64

# 切块目标长度（护栏 chunk_min_chars~chunk_max_chars 之间的中间值）。
# 不暴露成 .env 项：100/420 是护栏，300 是护栏内的目标，没有单独调参的需要。
_CHUNK_TARGET = 300

# 句末标点断言切分：按「。！？；」切句而不吞掉标点。
_SENT_SPLIT = re.compile(r"(?<=[。！？；])")


# 第一段：切块入库（读 docs/*.md → knowledge_documents/document_chunks）


def _normalize(text: str) -> str:
    """合并手动换行、压缩连续空格，保证一个切块内容是连续的正文文本。"""
    text = re.sub(r"[ \t]*\n[\s]*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_long(text: str) -> list[str]:
    """
    把超长段落按句切到 _CHUNK_TARGET 附近，单块不超过 chunk_max_chars。

    Args:
        text: 已 normalize 的超长段落（len > chunk_max_chars）。

    Returns:
        切句后的块列表；收尾碎块（< 80 字）并回上一块，避免过短噪声块。
    """
    pieces: list[str] = []
    buf = ""
    for sent in _SENT_SPLIT.split(text):
        if not sent:
            continue
        if len(buf) + len(sent) <= _CHUNK_TARGET:
            buf += sent
        else:
            if buf:
                pieces.append(buf)
            buf = sent
    if buf:
        pieces.append(buf)

    out: list[str] = []
    for piece in pieces:
        while len(piece) > settings.chunk_max_chars:   # 单句超长时硬切
            out.append(piece[: settings.chunk_max_chars])
            piece = piece[settings.chunk_max_chars :]
        if piece:
            out.append(piece)

    if len(out) >= 2 and len(out[-1]) < 80:
        out[-2] += out[-1]
        out.pop()
    return out


def chunk_document(raw_text: str, doc_title: str) -> list[str]:
    """
    把一份 md 语料切成带《文档名》【章节名】前缀的切块列表。

    纯函数（不碰库、不碰文件系统），方便单测直接验证切分规则；
    规则四条见模块 docstring，实现顺序：分章节 → 段落 normalize →
    短段合并 → 超长段切句 → 加前缀。

    Args:
        raw_text: md 文件全文。
        doc_title: 文档名（文件名去扩展名），进前缀参与向量化。

    Returns:
        切块内容列表；文档没有 ## 章节时返回空列表（调用方跳过）。
    """
    parts = re.split(r"^##\s+(.+)$", raw_text, flags=re.M)
    chunks: list[str] = []
    # parts[0] 是 # 主标题与导语（不成块），parts[1::2] 是章节名，parts[2::2] 是正文
    for i in range(1, len(parts), 2):
        section = parts[i].strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        paras = [p for p in (_normalize(x) for x in re.split(r"\n\s*\n", body)) if p]
        if not paras:
            continue

        # 规则 2：短段与同章节下一段合并
        merged: list[str] = []
        buf = ""
        for para in paras:
            buf = f"{buf}{para}" if buf else para
            if len(buf) >= settings.chunk_min_chars:
                merged.append(buf)
                buf = ""
        if buf:
            if merged:
                merged[-1] += buf
            else:
                merged.append(buf)

        # 规则 3 + 4：超长段切句，统一加标题前缀
        for para in merged:
            pieces = [para] if len(para) <= settings.chunk_max_chars else _split_long(para)
            chunks.extend(f"《{doc_title}》【{section}】{p}" for p in pieces)
    return chunks


def insert_corpus(rebuild: bool) -> int:
    """
    扫描 docs 目录，把语料切块写入 knowledge_documents / document_chunks。

    Args:
        rebuild: True 先清空旧语料（document_chunks 及其倒排表级联清空，
            再删 knowledge_documents）再重切重灌；
            False 只插入 file_name 尚未入库的新文档（幂等）。

    Returns:
        本次新插入的切块数；目录不存在/为空时返回 0 并告警。

    Raises:
        psycopg 异常：数据库连不上/表不存在。命令行脚本直接让异常
        冒出去打印堆栈，不需要包装成友好的错误对象。
    """
    docs_dir = settings.knowledge_dir_path / "docs"
    if not docs_dir.is_dir():
        logger.warning("语料目录不存在：%s，跳过切块入库", docs_dir)
        return 0
    # glob 只匹配 docs/ 顶层 *.md：_archive_ecom/ 等归档子目录天然排除
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        logger.warning("语料目录为空：%s，跳过切块入库", docs_dir)
        return 0

    inserted = 0
    with get_conn() as conn:
        with conn.cursor() as cur:
            if rebuild:
                # 删除顺序不能颠倒。chunk_sparse_terms 对 document_chunks
                # 建了 ON DELETE CASCADE，但 document_chunks 对 knowledge_documents
                # 没有 —— 先删父表会 ForeignKeyViolation，必须先删子表再删父表。
                cur.execute("DELETE FROM document_chunks")
                cur.execute("DELETE FROM knowledge_documents")
                logger.info("全量重建：已清空旧语料（文档/切块/倒排词项）")

            cur.execute("SELECT file_name FROM knowledge_documents")
            existing = {r[0] for r in cur.fetchall()}

            for path in files:
                if not rebuild and path.name in existing:
                    continue
                chunks = chunk_document(path.read_text(encoding="utf-8"), path.stem)
                if not chunks:
                    logger.warning("%s 没有 ## 章节，跳过", path.name)
                    continue
                cur.execute(
                    "INSERT INTO knowledge_documents (title, file_name, status) "
                    "VALUES (%s, %s, 'ready') RETURNING id",
                    (path.stem, path.name),
                )
                doc_id = cur.fetchone()[0]
                cur.executemany(
                    "INSERT INTO document_chunks (document_id, chunk_index, content) "
                    "VALUES (%s, %s, %s)",
                    [(doc_id, idx, c) for idx, c in enumerate(chunks)],
                )
                inserted += len(chunks)
                logger.info("已切块入库：%s → %d 块", path.name, len(chunks))
        # 这里不显式 conn.commit()。get_conn 正常退出时由连接池自动
        # commit，生产语义不变；而测试里 get_conn 被换成共享事务连接，
        # 显式 commit 会把整场测试的事务真提交、逃过 SAVEPOINT 回滚，
        # 测试数据泄漏进测试库（write_index 保留显式 commit 是因为它的
        # 用例整函数 mock，永远不会真的摸到这条连接）。
    return inserted


# 第二段：双路编码回填（原有逻辑不变）


def fetch_pending(rebuild: bool) -> list[tuple[int, str]]:
    """
    取出需要编码的切块（id, content）。

    Args:
        rebuild: True 取全部行（之后会先清空两路旧数据），
            False 只取"embedding 为 NULL 或在倒排表里没有任何词项"的行
            —— 两个条件满足其一即视为不完整，必须补齐（见模块 docstring
            的"同批同步写"说明）。

    Returns:
        [(chunk_id, text), ...] 列表，按 id 升序，方便日志和断点续跑。

    Raises:
        psycopg 异常：数据库连不上/表不存在。命令行脚本直接让异常
        冒出去打印堆栈，不需要包装成友好的错误对象。
    """
    if rebuild:
        sql = "SELECT id, content FROM document_chunks ORDER BY id"
    else:
        sql = """
        SELECT c.id, c.content
        FROM document_chunks c
        WHERE c.embedding IS NULL
           OR NOT EXISTS (SELECT 1 FROM chunk_sparse_terms t WHERE t.chunk_id = c.id)
        ORDER BY c.id
        """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql)
        return [(r[0], r[1]) for r in cur.fetchall()]


def clear_stores() -> int:
    """全量重建前清空两路存储（旧向量置 NULL + 倒排表清空），返回受影响切块数。"""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE document_chunks SET embedding = NULL")
            affected = cur.rowcount
            # 倒排表靠 chunk 级联删除，但这里块本身不删，必须显式清
            cur.execute("DELETE FROM chunk_sparse_terms")
        conn.commit()
    return affected


def write_index(
    dense_rows: list[tuple[int, np.ndarray]],
    sparse_rows: list[tuple[int, dict[int, float]]],
) -> None:
    """
    把一批切块的双路编码结果写回数据库（稠密向量 + 稀疏词项同批提交）。

    Args:
        dense_rows: [(chunk_id, 1024 维向量), ...]
        sparse_rows: [(chunk_id, {token_id: weight}), ...]
        两个列表必须覆盖同一批切块 id（由调用方 run 保证）。

    # 为什么要按批 commit 而不是全部一个事务？
    # 一个事务意味着"任何一行写失败就全部回滚"，听起来更安全，实则不然：
    # 回填是最典型的幂等任务（重跑只补缺），分批提交后，前面已提交的
    # 批次在下一次运行时自动跳过，失败损失被限制在最后一个批次里。
    # 一个大事务还会在写期间持有行锁、撑大 WAL，对在线库不友好。
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE document_chunks SET embedding = %s WHERE id = %s",
                [(vec, cid) for cid, vec in dense_rows],
            )
            # 词项展平成行级插入：一个块几百个非零词项，几百块 × 几百词项
            # ≈ 数万行 —— 对 PostgreSQL 只是普通小表，executemany 批量足够
            cur.executemany(
                "INSERT INTO chunk_sparse_terms (chunk_id, term_id, weight) "
                "VALUES (%s, %s, %s) "
                "ON CONFLICT (chunk_id, term_id) DO UPDATE SET weight = EXCLUDED.weight",
                [
                    (cid, term_id, weight)
                    for cid, terms in sparse_rows
                    for term_id, weight in terms.items()
                ],
            )
        conn.commit()  # get_conn 默认在退出时提交，这里显式写出来强调"双路同批原子"


def run(rebuild: bool) -> None:
    """
    编码回填主流程：统计 → 双头编码 → 双路写库 → 汇报。

    Args:
        rebuild: 是否全量重算（含清空两路旧数据）。

    Raises:
        RuntimeError: 模型加载失败时由 embedding 抛出，直接终止脚本 ——
        入库失败必须显式失败，绝不能"跳过向量继续跑"（那会让检索
        永远空召回且无人知晓）。
    """
    if rebuild:
        cleared = clear_stores()
        logger.info("全量重建：已清空 %d 条切块的两路旧数据", cleared)

    pending = fetch_pending(rebuild)
    if not pending:
        logger.info("没有待回填的切块（两路索引均已就绪），无事可做")
        return
    logger.info("待回填切块：%d 条，开始双头编码", len(pending))

    total_written = 0
    for start in range(0, len(pending), _FETCH_BATCH):
        batch = pending[start : start + _FETCH_BATCH]
        ids = [cid for cid, _ in batch]
        texts = [text for _, text in batch]

        vectors, sparse_list = embedding.encode_both(texts)

        # 维度自检。若模型被换成非 1024 维的（比如调成了 bge-large，
        # 768 维），PG 的 vector(1024) 列会在写入时报错，但那时已经编完
        # 一整批、报错信息只有 SQL 侧一句"dimensions do not match"。
        # 在写库前主动比对，能把错误定位到"模型配置"而不是"数据库"。
        if vectors.shape[1] != settings.embedding_dim:
            raise RuntimeError(
                f"模型输出 {vectors.shape[1]} 维，与库表 vector({settings.embedding_dim}) 不符，"
                f"请检查 EMBEDDING_MODEL_PATH / EMBEDDING_DIM 是否配套"
            )
        if len(sparse_list) != len(batch):
            raise RuntimeError(
                f"稀疏头输出 {len(sparse_list)} 条与输入 {len(batch)} 条不符，"
                "模型实现异常，拒绝部分入库（两路必须同批一致）"
            )

        write_index(
            list(zip(ids, vectors)),
            list(zip(ids, sparse_list)),
        )
        total_written += len(batch)
        logger.info("已写入 %d / %d 条（含各自稀疏词项）", total_written, len(pending))

    logger.info("回填完成：共写入 %d 条切块的双路索引", total_written)


def main() -> int:
    """
    命令行入口：切块入库 → 双路编码回填，两段串行。

    返回退出码（0 成功，1 失败），供脚本调用方判断。切块在编码之前：
    先有 document_chunks 行，fetch_pending 才能取到它们去编码。
    """
    setup_logging()
    # 命令行参数优先于 .env：--rebuild 显式要求全量时覆盖 EMBEDDING_REBUILD
    rebuild = "--rebuild" in sys.argv or settings.embedding_rebuild

    init_pool()
    try:
        insert_corpus(rebuild)
        run(rebuild)
        return 0
    except Exception:
        logger.exception("语料入库 / 双路索引回填失败")
        return 1
    finally:
        close_pool()


if __name__ == "__main__":
    sys.exit(main())
