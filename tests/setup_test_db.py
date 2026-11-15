"""
setup_test_db.py —— 创建/重置测试专用数据库

职责：
    在 hadoop100 上重建 ngs_service_assistant_test 库：直接执行
    ngs_schema.sql（12 张表）+ seed_ngs.sql（种子数据）。
    比旧版"从开发库整表拷贝"更确定：开发库被手工使用后的脏状态
    不会被带进测试库 —— 测试的前提永远来自同一份种子 SQL。

为什么需要它（2026-11-12 引入，2026-11-15 改为 SQL 直建）：
    此前测试直接跑在开发库上。测试虽然用 SAVEPOINT 回滚、不留数据，
    但它**读**库里已提交的状态 —— 用例写着"王医生名下只有 1 张
    qc_failed 委托单"，你一用系统手工复检两次，前提就破了，测试变红。
    根治办法：测试跑在独立库上，开发库随便用，互不影响。

用法（Windows，项目根目录）：
    D:\\app\\anaconda\\envs\\llm1\\python.exe tests\\setup_test_db.py

什么时候需要重跑：
    - script/ 里的种子 SQL 改了内容时（想让测试用上新种子）
    - 测试库被误删时
    - 测试期间的写入都会被 SAVEPOINT 回滚掉，所以正常情况**永远不用重跑**。

不负责：
    修改开发库（完全不碰）、在线服务的连接（服务连的是开发库）、
    语料切块的向量入库（语料到位后用 `python -m app.ingest --rebuild`，
    对开发库执行；测试库的向量检索用例全部 mock，不依赖真实向量）。
"""

import sys
from pathlib import Path

import psycopg

# ---------- 连接参数（与 .env 的开发库保持同一账号，只是库名不同） ----------
HOST = "192.168.218.100"
PORT = 5432
USER = "ai_cs"
PASSWORD = "ai_cs_pwd_2026"
TEST_DB = "ngs_service_assistant_test"


def _script_dir() -> Path:
    """
    定位伴生数据目录里的 script/（ngs_schema.sql / seed_ngs.sql 所在）。

    # script/ 不在代码仓里而在伴生数据目录（代码与数据分离），
    # 因此要用代码仓根目录（ngs_service_assistant）的父目录去找同级的
    # ngs_service_assistant_data/。
    # 找不到 schema 文件时直接报错，不做静默回退。
    """
    project_root = Path(__file__).resolve().parent.parent  # 代码仓根（ngs_service_assistant/）
    candidate = project_root.parent / "ngs_service_assistant_data" / "script"
    if (candidate / "ngs_schema.sql").exists():
        return candidate
    raise RuntimeError(
        "找不到 ngs_schema.sql：请在 ngs_service_assistant_data/script/ 下确认文件存在"
    )


def _statements(sql_text: str) -> list[str]:
    """
    把一份 SQL 文本切成逐条可执行的语句。

    # 必须先剔除 `-- 注释行`再按分号切分 —— 注释里也含分号，
    # 后切会截出"半截注释"当成 SQL 执行报语法错。种子的正文里
    # 只有中文标点，不含 ASCII 分号，切分本身是安全的。
    """
    lines = [l for l in sql_text.splitlines() if not l.strip().startswith("--")]
    return [s.strip() for s in "\n".join(lines).split(";") if s.strip()]


def recreate_test_db() -> None:
    """
    删除并重建测试库，然后执行 schema + 种子 SQL。

    异常：
        psycopg.OperationalError：账号权限不足或服务器不可达。
        本脚本是命令行工具，直接让异常冒出去给完整的报错信息。
    """
    schema_path = _script_dir() / "ngs_schema.sql"
    seed_path = _script_dir() / "seed_ngs.sql"
    schema_stmts = _statements(schema_path.read_text(encoding="utf-8"))
    seed_stmts = _statements(seed_path.read_text(encoding="utf-8"))

    # DROP/CREATE DATABASE 不能在事务里跑，必须 autocommit
    with psycopg.connect(host=HOST, port=PORT, dbname="postgres",
                         user=USER, password=PASSWORD, autocommit=True) as conn:
        # 有人/有测试正连着测试库时 DROP 会失败，提示信息里直接说明原因
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB}")
        conn.execute(f"CREATE DATABASE {TEST_DB}")

    with psycopg.connect(host=HOST, port=PORT, dbname=TEST_DB,
                         user=USER, password=PASSWORD) as conn:
        # vector 扩展是"库级别"的，新库不会自动继承开发库的扩展，
        # 必须先建扩展再建 document_chunks（vector 列类型依赖它）。
        # schema 的第一条语句就是 CREATE EXTENSION，这里无需单独处理。
        for stmt in schema_stmts + seed_stmts:
            conn.execute(stmt)
        conn.commit()


def main() -> int:
    """命令行入口：重建 → 自检 → 汇总。返回 0 成功。"""
    print(f"重建测试库 {TEST_DB}（ngs_schema.sql + seed_ngs.sql 直建）…")
    recreate_test_db()

    # 自检：种子的关键前提能查到，测试用例依赖这些数字
    with psycopg.connect(host=HOST, port=PORT, dbname=TEST_DB,
                         user=USER, password=PASSWORD) as conn:
        with conn.cursor() as cur:
            checks = {
                "账号数": ("SELECT count(*) FROM users", 6),
                "委托单数": ("SELECT count(*) FROM orders", 11),
                "样本数": ("SELECT count(*) FROM order_items", 12),
                "报告数": ("SELECT count(*) FROM reports", 3),
                "变异数": ("SELECT count(*) FROM report_variants", 10),
                "表数": ("SELECT count(*) FROM pg_tables WHERE schemaname='public'", 12),
            }
            for name, (sql, expected) in checks.items():
                got = cur.execute(sql).fetchone()[0]
                status = "OK" if got == expected else f"!!! 期望 {expected}"
                print(f"  {name}: {got} {status}")

    print(f"完成：测试库 {TEST_DB} 就绪。")
    print("现在可以正常运行 pytest，它只会碰这个库，不再影响开发库。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
