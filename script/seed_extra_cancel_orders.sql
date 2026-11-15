-- ============================================================
-- seed_extra_cancel_orders.sql —— 陈医生退单演示扩展数据
--
-- 用途：补充 3 张委托单 + 3 条样本，专供「申请退单」四段式演示。
--       与 seed_ngs.sql 分离，不污染基础种子，也不进测试库。
--
-- 为什么只有 3 张（硬约束）：
--   app/tools.py 的 get_user_orders 是 ORDER BY created_at DESC LIMIT 10，
--   陈医生基础种子已有 7 张。再加 3 张 = 恰好 10 张，全部可见；
--   超过 10 张会把最老的委托单 2/3 挤出工具结果，
--   「报告签发 30 天超复检窗口」「已退单不可重复提交」两个既有演示会失效。
--
-- 幂等 / 复位：
--   脚本用固定 ID 段 101~103（约定为本脚本专用，勿挪作他用），
--   每次执行先删后插。演示中把某张单退成 cancelled 后，
--   重新执行本脚本即可把它恢复成初始节点，可以反复演示。
--
-- ⚠ 只执行到开发库 ngs_service_assistant，不要对测试库执行：
--   测试库由 tests/setup_test_db.py 用 ngs_schema.sql + seed_ngs.sql 直建，
--   229 个用例依赖精确的订单数量与状态。
--
-- 口径：密码/金额/周期沿用 seed_ngs.sql；total_amount 必须等于明细
--       单价×份数（harness V3 金额比对源，也是本文件自检 2）。
-- ============================================================

BEGIN;

-- order_items 对 orders 是普通外键、没有 ON DELETE CASCADE，
-- 必须先删子表再删父表；否则 DELETE orders 会直接报外键约束错。
-- 三张单都不挂报告（退单/复检演示只改 orders 标志位与节点），
-- 所以无需清理 reports / report_variants。
DELETE FROM order_items WHERE order_id IN (101, 102, 103);
DELETE FROM orders      WHERE id       IN (101, 102, 103) AND user_id = 1;

-- ------------------------------------------------------------
-- 1. 委托单（3 张）：退单白名单 2 张 + 边界拦截 1 张
--    CANCEL_ALLOWED_STATUS = received / qc_passed / qc_failed
--    （见 app/harness.py:118 与 app/api_actions.py 的 UPDATE 三重条件）
-- ------------------------------------------------------------
INSERT INTO orders (id, user_id, status, total_amount, department,
                    is_urgent, retest_count, qc_note, created_at) VALUES
    -- 101 已接收：可退单，同时也可加急（一张单演示两种引导动作）。
    (101, 1, 'received',      3600.00, '市一医院呼吸内科', false, 0, NULL,
     now() - interval '6 hour'),

    -- 102 质检不合格：可退单，也可免费复检。qc_note 仅 qc_failed 单有值，
    -- 与 schema 约定一致；文案刻意写明「可退单」，让模型有据可依。
    (102, 1, 'qc_failed',     9800.00, '市一医院病理科',   false, 0,
     '质检不合格：送检胸水样本中肿瘤细胞比例低于本 panel 的最低要求，可申请免费复检一次，或在线申请退单。',
     now() - interval '1 day'),

    -- 103 建库完成：紧邻阈值的反例。已建库 → 不在退单白名单，V4 应拦截、
    -- 前端不出现确认按钮、引导转人工；但它仍在加急白名单内（可对比演示）。
    (103, 1, 'library_built', 4800.00, '市一医院肿瘤内科', false, 0, NULL,
     now() - interval '2 day');

-- ------------------------------------------------------------
-- 2. 样本（3 条）：HP-2026-01011~01013，与基础种子的编号段不冲突
--    单价取 products 表同款价格，保证金额对账为 0 行
-- ------------------------------------------------------------
INSERT INTO order_items (id, order_id, product_id, product_name,
                         sample_no, sample_type, quantity, unit_price) VALUES
    (101, 101, 1, '肺癌11基因检测',         'HP-2026-01011', 'FFPE组织', 1, 3600.00),
    (102, 102, 2, '实体瘤520基因大panel',   'HP-2026-01012', '胸水',     1, 9800.00),
    (103, 103, 3, 'MRD残留病灶监测',        'HP-2026-01013', '外周血',   1, 4800.00);

-- ------------------------------------------------------------
-- 3. 重置序列（显式插 id 后必须，否则后续 INSERT 会主键冲突）
-- ------------------------------------------------------------
SELECT setval('orders_id_seq',      (SELECT MAX(id) FROM orders));
SELECT setval('order_items_id_seq', (SELECT MAX(id) FROM order_items));

COMMIT;

-- ============================================================
-- 自检（执行后人工核对）
-- ============================================================

-- 自检 1：陈医生应为 10 张（LIMIT 10 刚好全可见，不多不少）
SELECT u.nickname, count(o.id) AS order_count
FROM users u LEFT JOIN orders o ON o.user_id = u.id
WHERE u.id = 1 GROUP BY u.nickname;

-- 自检 2：金额对账（应返回 0 行）
SELECT o.id, o.total_amount, sum(i.unit_price * i.quantity) AS item_total
FROM orders o JOIN order_items i ON i.order_id = o.id
WHERE o.id IN (101, 102, 103)
GROUP BY o.id, o.total_amount
HAVING o.total_amount <> sum(i.unit_price * i.quantity);

-- 自检 3：三张单的节点与样本（101 received / 102 qc_failed / 103 library_built）
SELECT o.id, o.status, o.is_urgent, o.retest_count, i.sample_no, i.sample_type
FROM orders o JOIN order_items i ON i.order_id = o.id
WHERE o.id IN (101, 102, 103) ORDER BY o.id;

-- 自检 4：陈医生的退单候选（白名单 3 种节点；应至少 3 张）
SELECT o.id, o.status FROM orders o
WHERE o.user_id = 1 AND o.status IN ('received', 'qc_passed', 'qc_failed')
ORDER BY o.id;

-- 自检 5：工具可见性（模拟 get_user_orders 的 LIMIT 10，三张新单都应出现）
SELECT id, status FROM orders
WHERE user_id = 1 ORDER BY created_at DESC LIMIT 10;
