-- ============================================================
-- ngs_schema.sql —— 肿瘤 NGS 送检全流程智能助手 · 表结构（12 张表）
--
-- 表结构总览：
--   users / conversations / messages / knowledge_documents /
--   document_chunks / agent_runs 为通用基础设施表；
--   products / orders / order_items 承载检测产品 / 委托单 / 样本；
--   reports / report_variants / chunk_sparse_terms 三张新增。
--
-- 执行方式：psycopg 整文件执行（脚本见 tests/setup_test_db.py 同款逻辑）。
-- 幂等：CREATE TABLE IF NOT EXISTS + CREATE INDEX IF NOT EXISTS，可重复执行。
-- ⚠ 本脚本只建表不灌数据，种子数据见 seed_ngs.sql。
-- ============================================================

BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------
-- 1. 账号表（role 正式入 DDL）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id            BIGSERIAL   PRIMARY KEY,
    username      VARCHAR(50) NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    nickname      VARCHAR(50),
    role          VARCHAR(20) NOT NULL DEFAULT 'customer',   -- customer：送检客户 / agent：技术支持坐席
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_users_username UNIQUE (username)
);

-- ------------------------------------------------------------
-- 2. 会话表（转人工状态机）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS conversations (
    id             BIGSERIAL  PRIMARY KEY,
    user_id        BIGINT     NOT NULL REFERENCES users(id),
    status         VARCHAR(20) NOT NULL DEFAULT 'AI',   -- AI / QUEUED / HUMAN / CLOSED
    claimed_by     BIGINT     REFERENCES users(id) ON DELETE SET NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_status ON conversations(status);

-- ------------------------------------------------------------
-- 3. 消息表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id              BIGSERIAL  PRIMARY KEY,
    conversation_id BIGINT     NOT NULL REFERENCES conversations(id),
    role            VARCHAR(10) NOT NULL,               -- user / ai
    content         TEXT       NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, id);

-- ------------------------------------------------------------
-- 4. 检测产品表
--    stock → turnaround_days：检测周期是本域的"高频事实"，库存无意义
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS products (
    id               BIGSERIAL PRIMARY KEY,
    name             VARCHAR(200) NOT NULL,
    category         VARCHAR(50),                -- 适用范围：肺癌 / 实体瘤 / 随访监测 / 遗传筛查 / 科研 …
    price            NUMERIC(10,2) NOT NULL,
    turnaround_days  INT NOT NULL,               -- 检测周期（工作日），从收样日起算
    status           VARCHAR(20) NOT NULL DEFAULT 'on_sale',  -- on_sale / paused
    description      TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 5. 检测委托单表（节点 9 档）
--
-- 为什么状态枚举用 DB CHECK 而不是只在 Python 常量里约束？
-- 状态是本项目最核心的事实源（harness V3/V4 全靠它比对），写脏一个
-- 非法值（如 recived 手滑）会让校验层静默漏杀。DB CHECK 是最后一道闸，
-- 与应用层词表双保险；将来若有其他语言的管理端写入也能被拦住。
--
-- is_urgent / retest_count 用标志位而不是改节点状态。
-- 加急、复检都不应让样本"倒退"回早期节点，否则会人为制造
-- "上一轮说已发报告、这轮又显示已接收"的事实漂移；退单例外——
-- cancelled 是终止态，节点只前进到终局，不允许再发起其他动作。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS orders (
    id            BIGSERIAL PRIMARY KEY,
    user_id       BIGINT NOT NULL REFERENCES users(id),
    status        VARCHAR(24) NOT NULL,
    total_amount  NUMERIC(10,2) NOT NULL,
    department    VARCHAR(200),                  -- 送检单位/科室
    is_urgent     BOOLEAN NOT NULL DEFAULT FALSE,
    retest_count  INT NOT NULL DEFAULT 0,
    qc_note       TEXT,                          -- 质检不合格原因（仅 qc_failed 有值；合格为 NULL）
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),   -- 语义=收样日期
    CONSTRAINT chk_orders_status CHECK (status IN (
        'received',       -- 已接收
        'qc_passed',      -- 质检合格
        'qc_failed',      -- 质检不合格
        'library_built',  -- 建库完成
        'sequencing',     -- 上机测序中
        'analyzing',      -- 生信分析中
        'report_review',  -- 报告审核中
        'report_issued',  -- 已发报告
        'cancelled'       -- 已退单
    ))
);
CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status);

-- ------------------------------------------------------------
-- 6. 样本表
--    sample_no 是本域的"订单号"等价物：harness V3 的正则提取对象
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS order_items (
    id           BIGSERIAL PRIMARY KEY,
    order_id     BIGINT NOT NULL REFERENCES orders(id),
    product_id   BIGINT NOT NULL REFERENCES products(id),
    product_name VARCHAR(200) NOT NULL,          -- 下单时快照
    sample_no    VARCHAR(20) NOT NULL,           -- 样本编号 HP-2026-XXXXX
    sample_type  VARCHAR(30) NOT NULL,           -- FFPE组织 / 外周血 / 胸水
    quantity     INT NOT NULL,                   -- 样本份数（如切片数）
    unit_price   NUMERIC(10,2) NOT NULL,         -- 下单时快照
    CONSTRAINT uq_sample_no UNIQUE (sample_no)
);
CREATE INDEX IF NOT EXISTS idx_order_items_order ON order_items(order_id);

-- ------------------------------------------------------------
-- 7. 检测报告表（新增）—— report_issued 委托单的报告
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS reports (
    id          BIGSERIAL PRIMARY KEY,
    order_id    BIGINT NOT NULL REFERENCES orders(id),
    report_no   VARCHAR(20) NOT NULL,            -- RPT-2026-XXX
    issued_at   TIMESTAMPTZ NOT NULL,            -- 签发时间（复检 7 日窗口的锚点）
    conclusion  TEXT,                            -- 结论摘要（阴性报告的落点）
    CONSTRAINT uq_report_no UNIQUE (report_no)
);
CREATE INDEX IF NOT EXISTS idx_reports_order ON reports(order_id);

-- ------------------------------------------------------------
-- 8. 报告变异表（新增）—— harness V3 变异比对的事实源
--    模型说出的 gene/c./p./Tier 都要与这里逐条核对
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS report_variants (
    id             BIGSERIAL PRIMARY KEY,
    report_id      BIGINT NOT NULL REFERENCES reports(id) ON DELETE CASCADE,
    gene           VARCHAR(20) NOT NULL,         -- HGNC 基因符号，如 EGFR
    c_change       VARCHAR(50),                  -- 核苷酸改变，如 c.2573T>C
    p_change       VARCHAR(50),                  -- 蛋白改变，如 p.L858R
    vaf            NUMERIC(5,4),                 -- 变异等位基因频率 0~1
    tier           INT NOT NULL,                 -- 1~4 对应 Tier I~IV
    interpretation TEXT,                         -- 一句话解读（报事实，不做决策）
    CONSTRAINT chk_variants_tier CHECK (tier BETWEEN 1 AND 4)
);
CREATE INDEX IF NOT EXISTS idx_variants_report ON report_variants(report_id);
CREATE INDEX IF NOT EXISTS idx_variants_gene ON report_variants(gene);

-- ------------------------------------------------------------
-- 9. 知识文档表
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS knowledge_documents (
    id         BIGSERIAL PRIMARY KEY,
    title      VARCHAR(200) NOT NULL,
    file_name  VARCHAR(255),
    status     VARCHAR(20) NOT NULL DEFAULT 'processing', -- processing / ready / failed
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------
-- 10. 知识切块表（语料为 SOP / 报告解读指南）
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES knowledge_documents(id),
    chunk_index INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(1024),                    -- 维度与 embedding 模型绑定！
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON document_chunks
    USING hnsw (embedding vector_cosine_ops);    -- 余弦相似度索引

-- ------------------------------------------------------------
-- 11. 稀疏倒排表（新增，bge-m3 混合检索用）
--     每行 = "某切块在某词项上的稀疏权重"；查询侧与 query 稀疏权重
--     做点积聚合，得到稀疏路排名，与稠密路 RRF 融合。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chunk_sparse_terms (
    chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    term_id  INT NOT NULL,
    weight   REAL NOT NULL,
    PRIMARY KEY (chunk_id, term_id)
);
CREATE INDEX IF NOT EXISTS idx_sparse_term ON chunk_sparse_terms(term_id);

-- ------------------------------------------------------------
-- 12. agent_runs 观测表
--     故意不建外键：观测日志的价值在于"会话删了记录还在"。
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS agent_runs (
    id                 BIGSERIAL PRIMARY KEY,
    conversation_id    BIGINT,
    user_id            BIGINT NOT NULL,
    question           TEXT NOT NULL,
    reply_type         VARCHAR(32),
    validation_status  VARCHAR(32) NOT NULL,
    error_codes        TEXT[] NOT NULL DEFAULT '{}',
    tool_call_count    INT NOT NULL DEFAULT 0,
    duration_ms        INT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_conversation ON agent_runs (conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_runs_status ON agent_runs (validation_status);

COMMIT;

-- 验证：SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1;  应返回 12 张表
