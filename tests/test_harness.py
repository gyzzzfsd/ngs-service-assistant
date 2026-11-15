"""
test_harness.py —— 五关卡校验测试（检测域版）

证据全部从真实数据库构建（陈医生查一次委托单就是种子数据的七张单、
查一次报告就是两份真实报告），不造假证据 —— 证据造假会把
"比对逻辑对不对"测成"假数据造得像不像"。

本域新增的比对对象各有专门用例：
    样本号（HP-YYYY-NNNNN）、检测周期、基因白名单与别名（HER2=ERBB2）、
    变异位点（含裸蛋白写法 L858R）、Tier 分级错报（TIER_MISMATCH）。
"""

from decimal import Decimal

import pytest

from app import harness, run_log, tools
from app.prompts import TOOL_FAIL_CONTENT
from app.schemas import AgentOutput, PageAction, ReplyType, ToolCallRecord


def _answer(content: str, action: PageAction | None = None, reply_type=ReplyType.ANSWER) -> AgentOutput:
    """造一个待校验的 AgentOutput。"""
    return AgentOutput(reply_type=reply_type, content=content, action=action)


@pytest.fixture
def chen_trace(db_isolation) -> list[ToolCallRecord]:
    """陈医生真实查过委托单的轨迹（7 张单），是构建证据的标准来源。"""
    result = tools.execute_tool("get_user_orders", "{}", user_id=1)
    assert result.success
    return [ToolCallRecord(tool_name="get_user_orders", arguments={}, result=result.data, success=True)]


@pytest.fixture
def chen_report_trace(db_isolation) -> list[ToolCallRecord]:
    """陈医生真实查过报告变异的轨迹（两份报告 10 条变异）。"""
    result = tools.execute_tool("get_report_variants", "{}", user_id=1)
    assert result.success
    return [ToolCallRecord(tool_name="get_report_variants", arguments={}, result=result.data, success=True)]


@pytest.fixture
def product_trace(db_isolation) -> list[ToolCallRecord]:
    """查过检测产品的轨迹，金额与周期证据来自产品表。"""
    result = tools.execute_tool("search_products", '{"query": "肺癌"}', user_id=1)
    assert result.success
    return [ToolCallRecord(tool_name="search_products", arguments={"query": "肺癌"}, result=result.data, success=True)]


# 提取（正则只提取不判断）


def test_extract_order_ids():
    f = harness.extract_facts("您有委托单 1 和委托单2，委托单编号 3 已发报告")
    assert f.order_ids == {1, 2, 3}


def test_extract_order_ids_negation_exempt():
    """
    否定语境里的委托单号不算业务事实。

    # 没有这条豁免，"没有委托单9"这种如实的回答也会被 V3 拦下，
    # 用户问一个不存在的单号，只能收到兜底转人工 —— 误伤比漏杀更伤体验。
    # 相邻式（"没有委托单9"）与短语式（"我这边查不到委托单9"）分别覆盖。
    """
    # 短语式：中间隔了几个字
    assert harness.extract_facts("我这边查不到委托单9，请核对后再试").order_ids == set()

    # 相邻式：模型实测最爱的写法
    assert harness.extract_facts("您只有7张委托单，没有委托单9。").order_ids == set()

    # 混句：否定只豁免紧跟着它的那个单号
    assert harness.extract_facts("查不到委托单9，但您的委托单1已接收").order_ids == {1}

    # 正面陈述不受豁免保护；"没有问题，"不会误豁免后面的单号
    assert harness.extract_facts("您的委托单9已发报告").order_ids == {9}
    assert harness.extract_facts("没有问题，委托单1已接收").order_ids == {1}


def test_extract_sample_nos():
    f = harness.extract_facts("样本 HP-2026-01001 已接收，HP-2026-00890 已发报告")
    assert f.sample_nos == {"HP-2026-01001", "HP-2026-00890"}

    # 否定语境豁免与单号同构："查无此样本"不能算业务事实
    assert harness.extract_facts("未找到样本 HP-2026-99999 的记录").sample_nos == set()


def test_extract_amounts_only_with_business_label():
    """带标签的金额提取，裸数字不提取 —— 防误杀的核心设计。"""
    f = harness.extract_facts(
        "委托单1合计 3600 元，检测周期 7 个工作日，覆盖 520 基因，加急 3 个工作日"
    )
    assert f.amounts == {Decimal("3600")}
    # 520 基因的 520 既不是金额也不参与周期比对；7 与 3 是周期
    assert f.turnaround_days == {7, 3}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("委托单1已接收", {"已接收"}),
        ("委托单 2 已发报告，委托单3 已退单", {"已发报告", "已退单"}),
        ("您的样本在上机测序中", {"上机测序中"}),
        ("您好请问有什么可以帮您", set()),
    ],
    ids=["单节点", "双节点", "陷阱词", "无节点"],
)
def test_extract_status_words(text, expected):
    assert harness.extract_facts(text).status_words == expected


def test_extract_genes_and_variants():
    """基因/变异提取 + 非基因缩写豁免（FFPE/VAF 不是基因）。"""
    f = harness.extract_facts("检出 EGFR L858R（c.2573T>C）与 KRAS G12C，FFPE 样本质量良好，MRD 检测 VAF 0.28。")
    assert f.genes == {"EGFR", "KRAS"}, f.genes
    # 裸蛋白写法归一成 p. 前缀，c. 写法原样收
    assert f.variants == {"p.L858R", "c.2573T>C", "p.G12C"}, f.variants


def test_extract_gene_negation_exempt():
    """「未检出」否定语境里的基因不要求证据 —— 阴性报告话术是有效事实。"""
    f = harness.extract_facts("本次检测未检出 EGFR 突变，未发现 HER2 变异。")
    assert f.genes == set()
    assert f.variants == set()


def test_extract_gene_alias_normalized():
    """HER2 归一成 ERBB2 —— 用户口语说别名，库里存 HGNC 正名。"""
    f = harness.extract_facts("检出 HER2 变异。")
    assert f.genes == {"ERBB2"}


# 证据构建（来自真实工具结果）


def test_evidence_from_real_orders(chen_trace):
    ev = harness.build_evidence(chen_trace)

    assert ev.has_order_query is True
    assert ev.order_ids == {1, 2, 3, 6, 7, 8, 11}
    assert ev.order_status == {
        1: "received", 2: "report_issued", 3: "cancelled",
        6: "analyzing", 7: "report_review", 8: "report_issued", 11: "qc_passed",
    }
    # 总额 + 明细单价都算合法金额
    assert Decimal("3600") in ev.amounts          # 委托单1 总额
    assert Decimal("9800") in ev.amounts          # 大 panel 总额
    assert Decimal("7200") in ev.amounts          # 双样本单总额
    # 样本号与周期也进证据
    assert "HP-2026-01001" in ev.sample_nos
    assert "HP-2026-00891" in ev.sample_nos       # 双样本单的第二条
    assert ev.turnaround_days == {7, 10}           # 肺癌11基因=7 / 大panel与HRD=10


def test_evidence_from_real_reports(chen_report_trace):
    ev = harness.build_evidence(chen_report_trace)

    assert ev.has_report_query is True
    assert "EGFR" in ev.genes and "ERBB2" in ev.genes
    assert "p.L858R" in ev.variants and "c.2573T>C" in ev.variants
    assert ev.variant_tier["p.L858R"] == 1
    assert ev.variant_tier["p.L755S"] == 2        # ERBB2 的变异是 Tier II


def test_evidence_ignores_failed_and_knowledge():
    """失败的工具、知识库工具都不进证据。"""
    trace = [
        ToolCallRecord(tool_name="get_user_orders", arguments={}, result=None, success=False),
        ToolCallRecord(
            tool_name="search_knowledge",
            arguments={},
            result={"chunks": [{"content": "委托单1金额为 99999 元"}]},
            success=True,
        ),
    ]
    ev = harness.build_evidence(trace)

    assert ev.has_order_query is False
    assert ev.order_ids == set()
    assert ev.amounts == set()   # SOP 内容不能变成"合法金额"，否则幻觉有了挡箭牌


# 核心用例（全部基于种子数据）


def test_case_1_hallucinated_order(chen_trace):
    """用例1：幻觉委托单。陈医生只有 1/2/3/6/7/8/11，说"委托单9"必须拦。"""
    r = harness.validate(_answer("您的委托单9已发报告"), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "委托单9" in r.error_message


def test_case_2_hallucinated_amount(chen_trace):
    """用例2：编造金额。委托单1 实际 3600，说 3500 必须拦。"""
    r = harness.validate(_answer("委托单1合计 3500 元"), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "3500" in r.error_message


def test_case_3_hallucinated_status(chen_trace):
    """用例3：编造节点。委托单1 实际已接收，说"已发报告"必须拦。"""
    r = harness.validate(_answer("委托单1已发报告"), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "委托单1" in r.error_message


def test_case_3b_status_stolen_from_another_order(chen_trace):
    """
    节点词确实在证据里，但属于另一张委托单 —— 同样必须拦。

    # 这条用例专治"集合比对"的漏杀。
    # 陈医生有 7 张单，其中委托单 2 就是已发报告：
    #   说"委托单2已发报告" → 对，放行；
    #   说"委托单1已发报告" → 节点词在证据里（属于委托单2），集合比对照样通过，
    #                         但这是把 B 单的节点安到 A 单头上，属于事实漂移。
    # 逐条一致性检查（_check_status_consistency）就是为这个场景加的。
    """
    ok = harness.validate(_answer("委托单2已发报告"), chen_trace)
    assert ok.passed is True, "委托单2 本来就是已发报告，不该被拦"

    bad = harness.validate(_answer("委托单1已发报告"), chen_trace)
    assert bad.passed is False
    assert bad.error_codes == [harness.UNSUPPORTED_FACT]
    assert "委托单1" in bad.error_message


def test_case_4_cross_user_action(chen_trace):
    """用例4：越权动作。委托单4 属于林教授，永远进不了陈医生的证据。

    正文故意不写委托单号：写"委托单4"的话会先被 V3 以 UNSUPPORTED_FACT 拦下
    （正文提到无依据单号同样是幻觉），这里专测 V4 对 action 的拦截。
    """
    action = PageAction(code="APPLY_URGENT", order_id=4)
    r = harness.validate(_answer("好的，马上为您处理", action, ReplyType.ACTION_GUIDE), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNVERIFIED_ACTION_RESOURCE]
    assert "委托单4" in r.error_message
    # 陈医生已经查过委托单，补查救不回来（再查也没有委托单4），不该要求补查
    assert r.needs_order_query is False


def test_case_5_urgent_blocked_after_sequencing(chen_trace):
    """用例5：不可加急的单给按钮。委托单6 已进入生信分析（上机后）。"""
    action = PageAction(code="APPLY_URGENT", order_id=6)
    r = harness.validate(_answer("委托单6可以加急", action, ReplyType.ACTION_GUIDE), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNVERIFIED_ACTION_RESOURCE]
    assert "analyzing" in r.error_message


def test_case_6_invented_action(chen_trace):
    """用例6：发明动作。白名单里只有三个 APPLY_* 码。"""
    action = PageAction(code="REFUND_ORDER", order_id=1)
    r = harness.validate(_answer("好的，为您退款", action, ReplyType.ACTION_GUIDE), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNVERIFIED_ACTION_RESOURCE]
    assert "REFUND_ORDER" in r.error_message


def test_case_7_normal_pass(chen_trace):
    """用例7：正常通过。单号、金额、节点、周期全部有据。"""
    r = harness.validate(_answer("委托单1已接收，合计 3600 元，按 7 个工作日的周期安排"), chen_trace)

    assert r.passed is True
    assert r.error_codes == []
    assert r.degraded is None


def test_case_8_no_false_positive_on_policy_constants(chen_trace):
    """
    用例8：误杀回归。政策常量的数字不带业务标签，不得当编造拦。

    # 这条比"能拦住"更重要。校验器拦得越狠，误杀越多，
    # 用户收到"已转人工"的次数就越多 —— 拦截是手段，不误伤正常回复才是底线。
    # 本域的两个高危裸数字：加急周期 3（不在任何单据字段里，是政策原文）、
    # "520 基因"的 520（panel 大小，既非金额也非周期）。
    """
    r = harness.validate(_answer("加急后检测周期缩短至 3 个工作日，委托单1已接收"), chen_trace)

    assert r.passed is True


# 三b、检测域新增比对对象（基因 / 变异 / Tier）


def test_hallucinated_gene_blocked(chen_report_trace):
    """编造基因（不在白名单也不在证据里）必须拦 —— 本域最高危幻觉。"""
    r = harness.validate(_answer("检出 QQZZ9 基因变异。"), chen_report_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "QQZZ9" in r.error_message


def test_whitelist_gene_passes_without_report_query():
    """白名单基因（HGNC 常见基因）无证据也放行 —— 只拦"查无此基因"的编造。"""
    r = harness.validate(_answer("BRAF 是实体瘤中的常见变异基因。"), [])

    assert r.passed is True


def test_hallucinated_variant_blocked(chen_report_trace):
    """编造变异位点：EGFR 是真基因，但 L999R 不在任何报告里。"""
    r = harness.validate(_answer("您的报告中检出 EGFR L999R。"), chen_report_trace)

    assert r.passed is False
    assert r.error_codes == [harness.UNSUPPORTED_FACT]
    assert "p.L999R" in r.error_message


def test_real_variant_restatement_passes(chen_report_trace):
    """真实变异的正常复述放行（裸蛋白写法 + 括号 Tier 都要过得去）。"""
    r = harness.validate(_answer("检出 EGFR L858R（Tier I），KRAS G12C（Tier I）。"), chen_report_trace)

    assert r.passed is True


def test_tier_mismatch_blocked(chen_report_trace):
    """
    Tier 错报单列拦截（TIER_MISMATCH）：p.L858R 实际 Tier I，说成 Tier III 必须拦。

    # 为什么单列错误码而不是并入 UNSUPPORTED_FACT？
    # Tier I 和 Tier III 对医生的意义完全不同，说错等级比编造位点更隐蔽；
    # 单列后 agent_runs 能直接统计"分级错报率"这个独有指标。
    """
    r = harness.validate(_answer("检出 EGFR L858R，分级为 Tier III。"), chen_report_trace)

    assert r.passed is False
    assert r.error_codes == [harness.TIER_MISMATCH]
    assert "p.L858R" in r.error_message


def test_negative_report_statement_passes():
    """阴性报告话术（未检出 EGFR）在无证据时也放行 —— 有效事实不误杀。"""
    r = harness.validate(_answer("本次检测未检出 EGFR 变异，FFPE 样本质量良好。"), [])

    assert r.passed is True


def test_her2_alias_passes_with_erbb2_evidence(chen_report_trace):
    """用户问 HER2、库里存 ERBB2 —— 别名归一后比对，不算编造。"""
    r = harness.validate(_answer("检出 HER2（ERBB2）变异，分级为 Tier II。"), chen_report_trace)

    assert r.passed is True


# V1 结构 / V2 前置降级 / 补查信号


def test_v1_action_with_wrong_reply_type(chen_trace):
    """action 只能出现在 ACTION_GUIDE 里。"""
    action = PageAction(code="APPLY_URGENT", order_id=1)
    r = harness.validate(_answer("委托单1已接收", action, ReplyType.ANSWER), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.MODEL_OUTPUT_INVALID]


def test_v1_action_guide_without_action(chen_trace):
    """ACTION_GUIDE 必须携带 action。"""
    r = harness.validate(_answer("请确认加急委托单1", None, ReplyType.ACTION_GUIDE), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.MODEL_OUTPUT_INVALID]


def test_v1_empty_content(chen_trace):
    r = harness.validate(_answer("   "), chen_trace)

    assert r.passed is False
    assert r.error_codes == [harness.MODEL_OUTPUT_INVALID]


def test_v2_all_business_tools_failed():
    """业务工具全部失败 → 静默换话术、清空 action，且算"通过"（内容是系统控制的）。"""
    trace = [
        ToolCallRecord(tool_name="get_user_orders", arguments={}, result=None, success=False),
        ToolCallRecord(tool_name="search_products", arguments={}, result=None, success=False),
    ]
    output = _answer("您的委托单1合计 3500 元")  # 编造的话也无所谓，反正要被换掉

    r = harness.validate(output, trace)

    assert r.passed is True
    assert r.degraded == "tool_failure"
    assert r.output.content == TOOL_FAIL_CONTENT
    assert r.output.reply_type == ReplyType.REQUEST_HANDOFF
    assert r.output.action is None


def test_v2_not_triggered_when_no_tool_called():
    """一个工具都没调 → 不算基础设施失败（该拦还是拦，走补查/重写那套）。"""
    r = harness.validate(_answer("委托单1已接收"), [])

    assert r.degraded is None
    assert r.passed is False                # 没证据 → UNSUPPORTED_FACT
    assert r.needs_order_query is True      # 但值得补查一次


def test_v2_knowledge_failure_is_not_infra_failure():
    """知识库里没查到是正常业务结果，不是故障，不触发 V2。"""
    trace = [ToolCallRecord(tool_name="search_knowledge", arguments={}, result=None, success=False)]

    r = harness.validate(_answer("我查不到相关 SOP，建议转人工"), trace)

    assert r.passed is True
    assert r.degraded is None


def test_needs_order_query_when_action_without_evidence():
    """
    按钮指向委托单但从未查过委托单 → 要补查，不是直接拒。

    这是实测缺陷的对症设计：模型沿用历史直接给按钮时，
    先补一次查询，复验大概率通过，用户不会被无端拒绝。
    """
    action = PageAction(code="APPLY_URGENT", order_id=1)
    r = harness.validate(_answer("好的，为委托单1登记加急", action, ReplyType.ACTION_GUIDE), [])

    assert r.passed is False
    assert r.needs_order_query is True


# V4 三动作的放行矩阵（基于种子数据）


def _action_output(code: str, order_id: int) -> AgentOutput:
    return AgentOutput(
        reply_type=ReplyType.ACTION_GUIDE,
        content="请确认",
        action=PageAction(code=code, order_id=order_id),
    )


@pytest.mark.parametrize(
    "code, order_id, should_pass, expected_fragment",
    [
        # 加急：未上机放行（received / qc_passed / library_built）
        ("APPLY_URGENT", 1, True, None),           # received
        ("APPLY_URGENT", 11, True, None),           # qc_passed
        # 加急：上机后拦截（sequencing 由陈医生证据里的 analyzing 代表"上机后"）
        ("APPLY_URGENT", 6, False, "上机"),          # analyzing
        ("APPLY_URGENT", 2, False, "上机"),          # report_issued
        # 退单：未建库放行
        ("APPLY_CANCEL", 11, True, None),           # qc_passed
        ("APPLY_CANCEL", 1, True, None),            # received
        # 退单：进入实验流程后拦截
        ("APPLY_CANCEL", 6, False, "退单"),          # analyzing
        # 复检：非 qc_failed 且非已发报告的单拦截
        ("APPLY_RETEST", 1, False, "复检"),          # received
    ],
    ids=[
        "加急-received放行", "加急-qc_passed放行", "加急-analyzing拦截", "加急-已发报告拦截",
        "退单-qc_passed放行", "退单-received放行", "退单-analyzing拦截", "复检-received拦截",
    ],
)
def test_v4_action_matrix(chen_trace, code, order_id, should_pass, expected_fragment):
    """V4 放行矩阵：三动作码 × 种子里各节点状态的委托单。"""
    r = harness.validate(_action_output(code, order_id), chen_trace)
    assert r.passed is should_pass, r.error_message
    if expected_fragment:
        assert expected_fragment in r.error_message


def test_v4_retest_within_window_passes(chen_report_trace):
    """
    复检的 7 日窗口：委托单8 的报告签发在 2 天前 → 窗口内，放行。

    # 窗口判定需要"报告签发时间"这个证据 —— 它只可能来自
    # get_report_variants 工具（委托单查询不含签发时间）。
    # 用报告轨迹而不是委托单轨迹构造，正是为了证明这条数据通路。
    """
    r = harness.validate(_action_output("APPLY_RETEST", 8), chen_report_trace)
    assert r.passed is True, r.error_message


def test_v4_retest_out_of_window_blocked(chen_report_trace):
    """委托单2 的报告签发在 30 天前 → 超出 7 日窗口，拦截。"""
    r = harness.validate(_action_output("APPLY_RETEST", 2), chen_report_trace)

    assert r.passed is False
    assert "7" in r.error_message


def test_v4_qc_failed_retest_passes(db_isolation):
    """王医生的 qc_failed 委托单：复检无条件放行（质检不合格是补救手段）。"""
    result = tools.execute_tool("get_user_orders", "{}", user_id=3)
    trace = [ToolCallRecord(tool_name="get_user_orders", arguments={}, result=result.data, success=True)]

    r = harness.validate(_action_output("APPLY_RETEST", 5), trace)

    assert r.passed is True, r.error_message


# summarize 与 agent_runs 落库


class _FakeRun:
    """summarize 只取属性，用最小对象就能测，不必构造完整 AgentRun。"""

    def __init__(self, validation, corrected=False, error=None):
        self.validation = validation
        self.corrected = corrected
        self.error = error


def test_summarize_passed(chen_trace):
    r = harness.validate(_answer("委托单1已接收"), chen_trace)
    assert run_log.summarize(_FakeRun(r))["status"] == "passed"


def test_summarize_corrected():
    r = harness.HarnessResult(passed=True, output=_answer("x"))
    s = run_log.summarize(_FakeRun(r, corrected=True))
    assert s["status"] == "corrected"
    assert s["passed"] is True


def test_summarize_rejected():
    r = harness.HarnessResult(passed=False, error_codes=[harness.UNSUPPORTED_FACT], output=_answer("x"))
    s = run_log.summarize(_FakeRun(r))
    assert s["status"] == "rejected"
    assert s["error_codes"] == [harness.UNSUPPORTED_FACT]


def test_summarize_degraded_tools():
    r = harness.HarnessResult(passed=True, degraded="tool_failure", output=_answer("x"))
    assert run_log.summarize(_FakeRun(r))["status"] == "degraded_tools"


def test_summarize_degraded_model():
    s = run_log.summarize(_FakeRun(None, error="LLM_ERROR: 主备模型都失败"))
    assert s["status"] == "degraded_model"
    assert s["error_codes"] == ["LLM_ERROR"]


def test_record_run_writes_row(raw_conn):
    """观测表落库 + 查回验证（用例结束回滚，不留数据）。"""
    # 问题文本用唯一值：同一场测试里 api_chat 的用例也会落 agent_runs，
    # 按问题文本查回时会串行，用唯一值避免测试顺序依赖。
    question = "落库验证专用问题-唯一"
    run_log.record_run(
        conversation_id=1,
        user_id=1,
        question=question,
        reply_type="ACTION_GUIDE",
        status="corrected",
        error_codes=["UNVERIFIED_ACTION_RESOURCE"],
        tool_call_count=2,
        duration_ms=1234,
    )
    with raw_conn.cursor() as cur:
        cur.execute(
            "select question, validation_status, error_codes, tool_call_count "
            "from agent_runs where question = %s",
            (question,),
        )
        row = cur.fetchone()

    assert row is not None
    assert row[1] == "corrected"
    assert row[2] == ["UNVERIFIED_ACTION_RESOURCE"]
    assert row[3] == 2


def test_correction_feedback_contains_specific_error():
    text = harness.correction_feedback("委托单9不在您的委托单列表中")
    assert "委托单9" in text
    assert "未通过服务端校验" in text
