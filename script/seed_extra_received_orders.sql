-- ============================================================
-- seed_extra_received_orders.sql —— 陈医生批量「已接收」委托单
--
-- 用途：再补 6 张状态全为 received 的委托单 + 6 条样本，
--       用于批量退单演示（received 在退单白名单内，可申请加急/退单）。
--       与 seed_ngs.sql、seed_extra_cancel_orders.sql 分离，不污染基础种子。
--
-- 配套改动（必须）：
--   app/tools.py 的 get_user_orders 查询上限已从 LIMIT 10 放宽到 LIMIT 30。
--   陈医生此前已有 10 张单，再加 6 张 = 16 张；若仍是 10，
--   最老的 6 张（含复检超窗口、双样本、已退单等演示单）会被挤出去。
--   改完 tools.py 后**需要重启后端** run.py 才生效。
--
-- 为什么 6 张正好覆盖 6 个在售产品：
--   products 表共 8 款，id 1~6 为 on_sale、7/8 为 paused。
--   一张单配一款在售产品，既让样品类型/检测周期有区分度，
--   又保证金额对账（total_amount = 明细单价×份数）一眼可核。
--
-- 幂等 / 复位：固定 ID 段 104~109（本脚本专用，勿挪作他用），
--   每次执行先删后插。演示中把这些单退成 cancelled 后重跑即恢复 received。
--
-- ⚠ 只执行到开发库 ngs_service_assistant，不要对测试库执行：
--   测试库由 tests/setup_test_db.py 直建，233 个用例依赖精确的 7 张种子单。
-- ============================================================

BEGIN;

-- order_items 对 orders 是普通外键、没有 ON DELETE CASCADE，
-- 必须先删子表再删父表，否则 DELETE orders 直接报外键约束错。
DELETE FROM order_items WHERE order_id IN (104, 105, 106, 107, 108, 109);
DELETE FROM orders      WHERE id       IN (104, 105, 106, 107, 108, 109) AND user_id = 1;

-- ------------------------------------------------------------
-- 1. 委托单（6 张）：全部 received（已接收，未建库 → 可退单 / 可加急）
--    created_at 用分钟级递减，保证 6 张都排在最前面、顺序稳定
-- ------------------------------------------------------------
INSERT INTO orders (id, user_id, status, total_amount, department,
                    is_urgent, retest_count, qc_note, created_at) VALUES
    (104, 1, 'received', 3600.00, '市一医院呼吸内科',     false, 0, NULL, now() - interval '10 minutes'),
    (105, 1, 'received', 9800.00, '市一医院肿瘤内科',     false, 0, NULL, now() - interval '20 minutes'),
    (106, 1, 'received', 4800.00, '市一医院肿瘤内科',     false, 0, NULL, now() - interval '30 minutes'),
    (107, 1, 'received', 5200.00, '市一医院妇科肿瘤',     false, 0, NULL, now() - interval '40 minutes'),
    (108, 1, 'received', 2800.00, '市一医院遗传咨询门诊', false, 0, NULL, now() - interval '50 minutes'),
    (109, 1, 'received', 6800.00, '市一医院科研合作组',   false, 0, NULL, now() - interval '60 minutes');

-- ------------------------------------------------------------
-- 2. 样本（6 条）：HP-2026-01014~01019，与既有编号段不冲突
--    单价取 products 同款价格；数量均为 1，故总额即单价
-- ------------------------------------------------------------
INSERT INTO order_items (id, order_id, product_id, product_name,
                         sample_no, sample_type, quantity, unit_price) VALUES
    (104, 104, 1, '肺癌11基因检测',       'HP-2026-01014', 'FFPE组织', 1, 3600.00),
    (105, 105, 2, '实体瘤520基因大panel', 'HP-2026-01015', '外周血',   1, 9800.00),
    (106, 106, 3, 'MRD残留病灶监测',      'HP-2026-01016', '外周血',   1, 4800.00),
    (107, 107, 4, 'HRD检测',              'HP-2026-01017', 'FFPE组织', 1, 5200.00),
    (108, 108, 5, '21基因遗传筛查',        'HP-2026-01018', '外周血',   1, 2800.00),
    (109, 109, 6, '全外显子组测序（WES）', 'HP-2026-01019', 'FFPE组织', 1, 6800.00);

-- ------------------------------------------------------------
-- 3. 重置序列（显式插 id 后必须，否则后续 INSERT 会主键冲突）
-- ------------------------------------------------------------
SELECT setval('orders_id_seq',      (SELECT MAX(id) FROM orders));
SELECT setval('order_items_id_seq', (SELECT MAX(id) FROM order_items));

COMMIT;

-- ============================================================
-- 自检（执行后人工核对）
-- ============================================================

-- 自检 1：陈医生应为 16 张（原 7 + 退单扩展 3 + 本次 6）
SELECT count(*) AS chen_order_count FROM orders WHERE user_id = 1;

-- 自检 2：6 张新单状态与样本（应全为 received，各 1 条样本）
SELECT o.id, o.status, o.total_amount, i.sample_no, i.sample_type, p.name AS product
FROM orders o
JOIN order_items i ON i.order_id = o.id
JOIN products p ON p.id = i.product_id
WHERE o.id IN (104, 105, 106, 107, 108, 109)
ORDER BY o.id;

-- 自检 3：金额对账（应返回 0 行）
SELECT o.id, o.total_amount, sum(i.unit_price * i.quantity) AS item_total
FROM orders o JOIN order_items i ON i.order_id = o.id
WHERE o.id IN (104, 105, 106, 107, 108, 109)
GROUP BY o.id, o.total_amount
HAVING o.total_amount <> sum(i.unit_price * i.quantity);

-- 自检 4：陈医生可退单候选（应为 9 张：11 / 101 / 102 / 104~109）
SELECT id, status FROM orders
WHERE user_id = 1 AND status IN ('received', 'qc_passed', 'qc_failed')
ORDER BY id;

-- 自检 5：可见性（模拟 get_user_orders 放宽后的 LIMIT 30；应返回全部 16 张）
SELECT id, status FROM orders
WHERE user_id = 1 ORDER BY created_at DESC LIMIT 30;
