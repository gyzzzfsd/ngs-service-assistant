-- ============================================================
-- seed_ngs.sql —— 肿瘤 NGS 送检助手 · 种子数据
--
-- ⚠ 脚本开头会清空全部业务表并重置自增 ID，仅用于开发/测试环境。
-- 可重复执行（幂等）。日期用 now()-interval 生成，保证
-- "报告签发 7 日内可复检"这类时效性演示场景永远新鲜。
--
-- 设计对齐：script/生信版改造设计说明.md §九；口径与
-- script/语料生成提示词-生信版.md 的统一数字口径表完全一致。
--
-- 密码说明：全部账号密码 123456（sha256 十六进制）。
-- 生产环境应改 bcrypt（见 app/security.py 注释）。
-- ============================================================

BEGIN;

TRUNCATE report_variants, reports, order_items, orders,
         products, messages, conversations, users,
         knowledge_documents, document_chunks, chunk_sparse_terms,
         agent_runs
RESTART IDENTITY CASCADE;

-- ------------------------------------------------------------
-- 1. 账号（6 个）：主演 / 隔离 / 复检演示 / 0单降级 / 越权 / 坐席
-- ------------------------------------------------------------
INSERT INTO users (id, username, password_hash, nickname, role) VALUES
    (1, 'chen',     '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '陈医生',   'customer'),
    (2, 'lin',      '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '林教授',   'customer'),
    (3, 'wang',     '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '王医生',   'customer'),
    (4, 'zhao',     '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '赵同学',   'customer'),
    (5, 'zhou',     '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '销售小周', 'customer'),
    (6, 'tech001',  '8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92', '技术支持老徐', 'agent');

-- ------------------------------------------------------------
-- 2. 检测产品（8 个）：周期与语料口径一致
--    肺癌11基因=7 工作日；520大panel=10；MRD=15；其余不在语料里，错开取值
-- ------------------------------------------------------------
INSERT INTO products (id, name, category, price, turnaround_days, status, description) VALUES
    (1, '肺癌11基因检测', '肺癌', 3600.00, 7, 'on_sale',
     '覆盖 EGFR、KRAS、ALK、ROS1 等 11 个肺癌核心基因的检测 panel，可检测点突变与插入缺失，适用于非小细胞肺癌的分子特征分析，支持 FFPE 组织、外周血、胸水三类样本。'),
    (2, '实体瘤520基因大panel', '实体瘤', 9800.00, 10, 'on_sale',
     '覆盖 520 个基因的实体瘤大 panel，含点突变、插入缺失及部分基因拷贝数变化，检测下限 VAF 0.5%，适用于需要广泛筛查潜在获益基因的场景。'),
    (3, 'MRD残留病灶监测', '随访监测', 4800.00, 15, 'on_sale',
     '基于个体化定制 panel 的微小残留病灶监测，通过外周血循环肿瘤 DNA 追踪治疗后残留信号，分基线与随访两阶段，单次随访 15 个工作日出报告。'),
    (4, 'HRD检测', '卵巢/乳腺癌', 5200.00, 10, 'on_sale',
     '同源重组修复缺陷检测，评估基因组不稳定状态，适用于卵巢癌、乳腺癌等瘤种的伴随信息查询。'),
    (5, '21基因遗传筛查', '遗传筛查', 2800.00, 12, 'on_sale',
     '针对 21 个遗传易感基因的筛查，适用于肿瘤患者及高危人群的家系遗传风险评估的信息查询。'),
    (6, '全外显子组测序（WES）', '科研', 6800.00, 12, 'on_sale',
     '全外显子组捕获测序，面向科研合作项目，数据交付含原始数据与标准分析结果。'),
    (7, '消化道肿瘤26基因panel', '消化道肿瘤', 4200.00, 10, 'paused',
     '覆盖消化道肿瘤 26 个核心基因的检测 panel，当前暂停接样，恢复时间另行通知。'),
    (8, '泛癌种648基因超panel', '泛癌种', 12800.00, 15, 'paused',
     '覆盖 648 个基因的超大 panel，处于验证阶段，暂未开放接样。');

-- ------------------------------------------------------------
-- 3. 委托单（11 张）：9 档节点全覆盖 + 阴性/阳性报告分置
--    qc_note 仅 qc_failed 单有值（王医生单：DNA 浓度不足）
-- ------------------------------------------------------------
INSERT INTO orders (id, user_id, status, total_amount, department, is_urgent, retest_count, qc_note, created_at) VALUES
    (1,  1, 'received',       3600.00, '市一医院呼吸内科',   false, 0, NULL, now() - interval '2 day'),
    (2,  1, 'report_issued',  9800.00, '市一医院呼吸内科',   false, 0, NULL, now() - interval '30 day'),
    (3,  1, 'cancelled',      3600.00, '市一医院呼吸内科',   false, 0, NULL, now() - interval '45 day'),
    (4,  2, 'sequencing',     4800.00, '医科大学肿瘤研究所', false, 0, NULL, now() - interval '4 day'),
    (5,  3, 'qc_failed',      3600.00, '市二医院胸外科',     false, 0,
         '质检不合格：组织 DNA 浓度低于本检测的最低投入量要求，可申请免费复检一次，或补送新样本后重新安排实验。', now() - interval '3 day'),
    (6,  1, 'analyzing',      9800.00, '市一医院呼吸内科',   false, 0, NULL, now() - interval '6 day'),
    (7,  1, 'report_review',  3600.00, '市一医院呼吸内科',   false, 0, NULL, now() - interval '5 day'),
    (8,  1, 'report_issued',  7200.00, '市一医院呼吸内科',   false, 0, NULL, now() - interval '12 day'),
    (9,  5, 'library_built',  4800.00, '市三医院肿瘤内科',   false, 0, NULL, now() - interval '3 day'),
    (10, 2, 'report_issued',  3600.00, '医科大学肿瘤研究所', false, 0, NULL, now() - interval '20 day'),
    (11, 1, 'qc_passed',      5200.00, '市一医院妇科肿瘤',   false, 0, NULL, now() - interval '1 day');

-- ------------------------------------------------------------
-- 4. 样本（12 条）：编号 HP-2026-XXXXX；双样本单=委托单8
-- ------------------------------------------------------------
INSERT INTO order_items (id, order_id, product_id, product_name, sample_no, sample_type, quantity, unit_price) VALUES
    (1,  1,  1, '肺癌11基因检测',    'HP-2026-01001', 'FFPE组织', 1, 3600.00),
    (2,  2,  2, '实体瘤520基因大panel', 'HP-2026-00902', '外周血',   1, 9800.00),
    (3,  3,  1, '肺癌11基因检测',    'HP-2026-00850', 'FFPE组织', 1, 3600.00),
    (4,  4,  3, 'MRD残留病灶监测',   'HP-2026-00950', '外周血',   1, 4800.00),
    (5,  5,  1, '肺癌11基因检测',    'HP-2026-00961', 'FFPE组织', 1, 3600.00),
    (6,  6,  2, '实体瘤520基因大panel', 'HP-2026-00930', 'FFPE组织', 1, 9800.00),
    (7,  7,  1, '肺癌11基因检测',    'HP-2026-00940', '胸水',     1, 3600.00),
    (8,  8,  1, '肺癌11基因检测',    'HP-2026-00890', 'FFPE组织', 1, 3600.00),
    (9,  8,  1, '肺癌11基因检测',    'HP-2026-00891', '胸水',     1, 3600.00),
    (10, 9,  3, 'MRD残留病灶监测',   'HP-2026-00955', '外周血',   1, 4800.00),
    (11, 10, 1, '肺癌11基因检测',    'HP-2026-00900', '外周血',   1, 3600.00),
    (12, 11, 4, 'HRD检测',           'HP-2026-00970', 'FFPE组织', 1, 5200.00);

-- ------------------------------------------------------------
-- 5. 报告（3 份）：阳性大panel / 双样本阳性 / 阴性（未检出）
--    报告2 签发在 2 天前 → 处于"7 日内可复检"窗口（演示复检放行）
--    报告1 签发在 30 天前 → 超窗口（演示复检 DECLINE 转人工）
-- ------------------------------------------------------------
INSERT INTO reports (id, order_id, report_no, issued_at, conclusion) VALUES
    (1, 2,  'RPT-2026-0001', now() - interval '30 day',
     '本次大 panel 检出 6 条变异，含 Tier I 变异 2 条，详见变异列表。本报告仅供信息查询与流程支持，不构成医学建议。'),
    (2, 8,  'RPT-2026-0002', now() - interval '2 day',
     '双样本检测共检出 4 条变异，含 Tier I 变异 1 条，详见变异列表。本报告仅供信息查询与流程支持，不构成医学建议。'),
    (3, 10, 'RPT-2026-0003', now() - interval '20 day',
     '本次检测未检出明确致病变异（阴性报告）。未检出不等于体内无肿瘤，受 panel 覆盖范围、样本质量与检测下限影响，详见报告解读指南。');

-- ------------------------------------------------------------
-- 6. 报告变异（10 条）：真实知名位点；c./p./VAF/Tier 是 V3 比对源
--    ERBB2 p.L755S 专为基因别名归一化（HER2=ERBB2）演示保留
-- ------------------------------------------------------------
INSERT INTO report_variants (id, report_id, gene, c_change, p_change, vaf, tier, interpretation) VALUES
    (1,  1, 'EGFR',   'c.2573T>C', 'p.L858R',  0.4200, 1, '有强临床意义，与已获批靶向药物的敏感性证据等级相关。'),
    (2,  1, 'KRAS',   'c.35G>A',   'p.G12D',   0.1900, 1, '有强临床意义，与部分已获批药物的耐药性证据等级相关。'),
    (3,  1, 'PIK3CA', 'c.1633G>A', 'p.E545K',  0.0800, 2, '有潜在临床意义，可关注相关临床试验信息。'),
    (4,  1, 'TP53',   'c.524G>A',  'p.R175H',  0.3500, 3, '临床意义未明，建议结合样本与文献综合判断。'),
    (5,  1, 'KEAP1',  'c.601G>A',  'p.E201K',  0.1200, 3, '临床意义未明。'),
    (6,  1, 'RBM10',  'c.1906G>A', 'p.E636K',  0.0900, 3, '临床意义未明。'),
    (7,  2, 'KRAS',   'c.34G>T',   'p.G12C',   0.2800, 1, '有强临床意义，与已获批靶向药物的敏感性证据等级相关。'),
    (8,  2, 'TP53',   'c.818G>A',  'p.R273H',  0.5100, 3, '临床意义未明。'),
    (9,  2, 'ERBB2',  'c.2264T>C', 'p.L755S',  0.0600, 2, '有潜在临床意义（用户问 HER2 时应命中本条：HER2 为 ERBB2 别名）。'),
    (10, 2, 'PIK3CA', 'c.3140A>G', 'p.H1047R', 0.1100, 2, '有潜在临床意义，可关注相关临床试验信息。');

-- ------------------------------------------------------------
-- 7. 预置会话（1 段）：刻意包含一个"已过期的节点"
--    第 4 条消息里 AI 说大panel单"上机测序中"，但该单现已
--    report_issued —— 专门用来复现"跨轮引用历史导致事实漂移"缺陷
-- ------------------------------------------------------------
INSERT INTO conversations (id, user_id, status, created_at, last_active_at) VALUES
    (1, 1, 'AI', now() - interval '3 day', now() - interval '3 day');

INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES
    (1, 1, 'user', '你好，我的样本 HP-2026-01001 现在到哪一步了？', now() - interval '3 day'),
    (2, 1, 'ai',   '陈医生您好，您的样本 HP-2026-01001（肺癌11基因检测，委托单1）目前已接收，正在等待质检排期。按 7 个工作日的检测周期预计很快安排，收到后会第一时间告知您。', now() - interval '3 day'),
    (3, 1, 'user', '那之前那张大 panel 的单子呢？', now() - interval '3 day'),
    (4, 1, 'ai',   '您的委托单2（实体瘤520基因大panel，样本 HP-2026-00902）目前在上机测序中，按 10 个工作日的周期，报告还需要一些时间，请耐心等待。', now() - interval '3 day');

-- ------------------------------------------------------------
-- 8. 重置序列（显式插 id 后必须，否则后续 INSERT 会主键冲突）
-- ------------------------------------------------------------
SELECT setval('users_id_seq',            (SELECT MAX(id) FROM users));
SELECT setval('products_id_seq',         (SELECT MAX(id) FROM products));
SELECT setval('orders_id_seq',           (SELECT MAX(id) FROM orders));
SELECT setval('order_items_id_seq',      (SELECT MAX(id) FROM order_items));
SELECT setval('reports_id_seq',          (SELECT MAX(id) FROM reports));
SELECT setval('report_variants_id_seq',  (SELECT MAX(id) FROM report_variants));
SELECT setval('conversations_id_seq',    (SELECT MAX(id) FROM conversations));
SELECT setval('messages_id_seq',         (SELECT MAX(id) FROM messages));

COMMIT;

-- ============================================================
-- 自检（执行后人工核对，不满足预期说明数据写错）
-- ============================================================

-- 自检 1：按用户聚合委托单数（赵同学应为 0，其余见下）
--   陈医生 7 / 林教授 2 / 王医生 1 / 赵同学 0 / 销售小周 1 / 技术支持 0
SELECT u.nickname, count(o.id) AS order_count
FROM users u LEFT JOIN orders o ON o.user_id = u.id
GROUP BY u.nickname ORDER BY order_count DESC;

-- 自检 2：金额对账（委托单总额 = 明细单价×份数 的合计；应返回 0 行）
SELECT o.id AS order_id, o.total_amount, sum(i.unit_price * i.quantity) AS item_total
FROM orders o JOIN order_items i ON i.order_id = o.id
GROUP BY o.id, o.total_amount
HAVING o.total_amount <> sum(i.unit_price * i.quantity);

-- 自检 3：节点覆盖（9 档各 ≥1 张）
SELECT status, count(*) FROM orders GROUP BY status ORDER BY status;

-- 自检 4：报告-变异一致性（阴性报告 RPT-2026-0003 应为 0 条变异）
SELECT r.report_no, count(rv.id) AS variant_count
FROM reports r LEFT JOIN report_variants rv ON rv.report_id = r.id
GROUP BY r.report_no ORDER BY r.report_no;
