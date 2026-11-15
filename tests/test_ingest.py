"""
test_ingest.py —— 切块入库 + 双路索引回填脚本的单元测试

策略：
    切块是纯函数，直接喂构造的 md 文本断言规则（前缀/短段合并/超长切句）；
    入库幂等性连真实测试库验证（临时 docs 目录 + SAVEPOINT 回滚不留数据）；
    回填的核心风险有四个：漏补（幂等性失效）、维度错配写到一半才发现、
    稀疏头输出条数与输入不符、批次提交策略被悄悄改坏。前三件用假件验证
    控制流；第四件由 run 的日志覆盖。真实回填（GPU 编码）只在语料入库时
    手动跑一次，不做自动化。
"""

import numpy as np
import pytest

from app import ingest
from app.config import settings


@pytest.fixture
def fake_flow(monkeypatch):
    """
    把 ingest.run 的协作者换成记录器：
      fetch_pending → 预置待回填行
      clear_stores → 只记录调用
      embedding.encode_both → 预置双路输出
      write_index → 只记录调用不碰库
    返回 dict，用例从里面取记录结果。
    """
    state = {
        "pending": [(1, "文本一"), (2, "文本二"), (3, "文本三")],
        "encoded": [],
        "written": [],
        "cleared": False,
    }

    def _encode_both(texts):
        state["encoded"].append(list(texts))
        return (
            np.ones((len(texts), 1024), dtype=np.float32),
            [{11: 0.5, 22: 0.25} for _ in texts],
        )

    monkeypatch.setattr(ingest, "fetch_pending", lambda rebuild: state["pending"])
    monkeypatch.setattr(ingest, "clear_stores", lambda: state.__setitem__("cleared", True) or 10)
    monkeypatch.setattr(ingest.embedding, "encode_both", _encode_both)
    monkeypatch.setattr(ingest, "write_index", lambda d, s: state["written"].append((d, s)))
    return state


def test_run_encodes_and_writes_both_paths(fake_flow):
    """
    正常路径：全部待回填行被编码一次、两路结果一起写库。

    # 写库必须是 (dense_rows, sparse_rows) 成对出现 ——
    # 两路共用同一次前向，分两次写就会出现"覆盖的切块集合不一致"，
    # 稀疏路命中一个块而它的 embedding 还是 NULL。
    """
    ingest.run(rebuild=False)

    assert fake_flow["encoded"] == [["文本一", "文本二", "文本三"]]
    assert len(fake_flow["written"]) == 1
    dense_rows, sparse_rows = fake_flow["written"][0]
    assert [cid for cid, _vec in dense_rows] == [1, 2, 3]
    assert [cid for cid, _terms in sparse_rows] == [1, 2, 3]
    assert fake_flow["cleared"] is False  # 非重建模式不清空旧索引


def test_run_rebuild_clears_first(fake_flow):
    """rebuild=True 先清空两路旧数据再重算（全量重建语义）。"""
    ingest.run(rebuild=True)

    assert fake_flow["cleared"] is True


def test_run_noop_when_nothing_pending(fake_flow):
    """没有待回填行时是优雅的 no-op，不编码不写库。"""
    fake_flow["pending"] = []

    ingest.run(rebuild=False)

    assert fake_flow["encoded"] == []
    assert fake_flow["written"] == []


def test_run_aborts_on_dimension_mismatch(fake_flow, monkeypatch):
    """
    模型输出维度与 vector(1024) 列不符时，写库前就报错终止。

    # 这个自检把失败从"SQL 报 dimensions do not match"提前到
    # "Python 侧报模型配置错"。前者只回滚最后一批、信息含糊；
    # 后者在写任何一行之前发生，且错误信息直接指向 .env 的两个配置项。
    """
    monkeypatch.setattr(
        ingest.embedding,
        "encode_both",
        lambda texts: (np.ones((len(texts), 768), dtype=np.float32), [{1: 0.5} for _ in texts]),
    )

    with pytest.raises(RuntimeError, match="768"):
        ingest.run(rebuild=False)

    # 关键：一条都没写进去
    assert fake_flow["written"] == []


def test_run_aborts_on_sparse_count_mismatch(fake_flow, monkeypatch):
    """
    稀疏头输出条数与输入不符 → 拒绝入库（两路必须同批一致）。

    # 这是"双路原子"的另一半防线 —— 维度错会被 vector 列拦下，
    # 但条数不齐没有任何数据库约束兜底，必须在 Python 侧显式拒绝，
    # 否则会出现"稠密路写满、稀疏路缺行"的静默不一致。
    """
    monkeypatch.setattr(
        ingest.embedding,
        "encode_both",
        lambda texts: (np.ones((len(texts), 1024), dtype=np.float32), [{1: 0.5}]),
    )

    with pytest.raises(RuntimeError, match="稀疏头输出"):
        ingest.run(rebuild=False)

    assert fake_flow["written"] == []


# 幂等性判定（连真实测试库）


def _insert_chunk(raw_conn, content: str, with_embedding: bool) -> int:
    """现场插一条切块（embedding 可控），随用例 SAVEPOINT 回滚不留数据。"""
    with raw_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO knowledge_documents (title, file_name, status) "
            "VALUES (%s, %s, 'ready') RETURNING id",
            ("幂等性临时文档", "_tmp.md"),
        )
        doc_id = cur.fetchone()[0]
        vec = "[" + ",".join(["0.1"] * 1024) + "]" if with_embedding else None
        cur.execute(
            "INSERT INTO document_chunks (document_id, chunk_index, content, embedding) "
            "VALUES (%s, 0, %s, %s::vector) RETURNING id",
            (doc_id, content, vec),
        )
        return cur.fetchone()[0]


def test_fetch_pending_takes_null_embedding_rows(raw_conn):
    """幂等性：embedding 为 NULL 的行才待回填（已回填的不重复编码）。"""
    cid = _insert_chunk(raw_conn, "待回填切块", with_embedding=False)

    pending = ingest.fetch_pending(rebuild=False)

    assert (cid, "待回填切块") in pending


def test_fetch_pending_takes_rows_missing_sparse_terms(raw_conn):
    """
    有稠密向量但缺稀疏词项的行同样待回填。

    # 混合检索时代"完整"有两个条件 —— embedding 非空 **且**
    # 倒排表里有词项。只判前者会把升级前的旧块当成已完成，
    # 稀疏路永远查不到它们，混合检索悄悄瘸了一条腿。
    """
    cid = _insert_chunk(raw_conn, "只有稠密没有稀疏", with_embedding=True)

    pending = ingest.fetch_pending(rebuild=False)

    assert (cid, "只有稠密没有稀疏") in pending


def test_fetch_pending_rebuild_takes_all(raw_conn):
    """重建模式取全部行（包括两路齐备的）。"""
    cid = _insert_chunk(raw_conn, "重建模式测试切块", with_embedding=True)
    with raw_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO chunk_sparse_terms (chunk_id, term_id, weight) VALUES (%s, 99, 0.5)",
            (cid,),
        )

    pending = ingest.fetch_pending(rebuild=True)

    ids = [row[0] for row in pending]
    assert cid in ids
    assert ids == sorted(ids)  # id 升序，断点续跑依赖这个顺序


# 切块规则（chunk_document 纯函数，不碰库）


def test_chunk_document_adds_title_and_section_prefix():
    """规则 4：每块带《文档名》【章节名】前缀，标题参与向量化。"""
    body = "周" * 120
    chunks = ingest.chunk_document(f"## 周期说明\n\n{body}", "肺癌11基因检测产品说明")

    assert chunks == [f"《肺癌11基因检测产品说明》【周期说明】{body}"]


def test_chunk_document_merges_short_paragraphs():
    """规则 2：低于 chunk_min_chars 的短段并入同章节的下一段。"""
    md = "## 一\n\n短段。\n\n" + "长" * settings.chunk_min_chars
    chunks = ingest.chunk_document(md, "文档")

    assert len(chunks) == 1
    assert "短段。" in chunks[0]
    assert "长" * settings.chunk_min_chars in chunks[0]


def test_chunk_document_splits_long_paragraph():
    """
    规则 3：超长段按句切分，块长（不含前缀）不超过 chunk_max_chars。

    # 断言"每句恰好出现一次"而不是字符串全等 —— 切分点在句边界，
    # 各块拼回应等于原文（去掉前缀后），这比只看块数更能防"切丢句子"。
    """
    sentences = ["句子。"] * 150                      # 450 字 > 420 上限
    md = "## 二\n\n" + "".join(sentences)

    chunks = ingest.chunk_document(md, "文档")

    assert len(chunks) >= 2
    joined = "".join(c.split("】", 1)[1] for c in chunks)
    assert joined == "".join(sentences)
    assert all(len(c.split("】", 1)[1]) <= settings.chunk_max_chars for c in chunks)


def test_chunk_document_ignores_preamble():
    """# 主标题下的导语不单独成块（过短且无章节上下文）。"""
    md = "# 标题\n\n导语不入块。\n\n## 章节\n\n" + "正" * 120

    chunks = ingest.chunk_document(md, "文档")

    assert len(chunks) == 1
    assert "导语不入块" not in chunks[0]


def test_chunk_document_without_sections_returns_empty():
    """没有 ## 章节的文档切不出任何块（调用方按"跳过"处理）。"""
    assert ingest.chunk_document("# 只有主标题\n\n正文正文", "文档") == []


# 切块入库幂等性（连真实测试库，SAVEPOINT 回滚不留数据）


def _make_docs_dir(tmp_path, file_name: str, body: str):
    """在临时目录里造一个 docs/ 语料目录，返回目录路径。"""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / file_name).write_text(f"## 一\n\n{body}", encoding="utf-8")
    return docs_dir


def test_insert_corpus_is_idempotent(tmp_path, monkeypatch, raw_conn):
    """
    默认模式幂等：同一份语料第二次跑不再插入（file_name 已入库即跳过）。

    # monkeypatch 的是 settings.knowledge_dir 而不是
    # knowledge_dir_path —— 后者是实例 property，patch 不了；前者是普通
    # pydantic 字段，knowledge_dir_path 每次访问都基于它现算，改它即生效。
    """
    docs_dir = _make_docs_dir(tmp_path, "手册A.md", "甲" * 120)
    monkeypatch.setattr(settings, "knowledge_dir", str(docs_dir.parent))

    first = ingest.insert_corpus(rebuild=False)
    second = ingest.insert_corpus(rebuild=False)

    assert first == 1
    assert second == 0
    with raw_conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM document_chunks WHERE content LIKE '%手册A%'")
        assert cur.fetchone()[0] == 1


def test_insert_corpus_rebuild_replaces_old_docs(tmp_path, monkeypatch, raw_conn):
    """重建模式先删全部旧文档（级联清空切块与倒排表）再重切重灌。"""
    docs_dir = _make_docs_dir(tmp_path, "手册A.md", "甲" * 120)
    monkeypatch.setattr(settings, "knowledge_dir", str(docs_dir.parent))
    ingest.insert_corpus(rebuild=False)

    # 换一份语料后全量重建：A 应被删除（连带切块），只剩 B
    (docs_dir / "手册B.md").write_text("## 一\n\n" + "乙" * 120, encoding="utf-8")
    (docs_dir / "手册A.md").unlink()
    ingest.insert_corpus(rebuild=True)

    with raw_conn.cursor() as cur:
        cur.execute("SELECT file_name FROM knowledge_documents ORDER BY file_name")
        assert [r[0] for r in cur.fetchall()] == ["手册B.md"]
        cur.execute("SELECT count(*) FROM document_chunks WHERE content LIKE '%手册A%'")
        assert cur.fetchone()[0] == 0


def test_insert_corpus_missing_dir_is_graceful(tmp_path, monkeypatch):
    """语料目录不存在时告警返回 0，而不是抛异常（命令行脚本的宽容度）。"""
    monkeypatch.setattr(settings, "knowledge_dir", str(tmp_path / "不存在"))

    assert ingest.insert_corpus(rebuild=False) == 0
