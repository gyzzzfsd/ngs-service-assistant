"""
harness.py —— 五关卡校验（本项目的核心亮点）

职责：
    校验"AI 说的每句业务事实有没有工具依据"。模型负责生成，工具提供事实，
    本文件负责核对，核对不过的回复不发给用户。五道关卡顺序固定：

        V1 结构校验   reply_type/action 的组合是否合法（查表，零正则）
        V2 前置降级   业务工具全部失败 → 静默换固定话术，不再往下走
        V3 事实校验   正则提取六类事实 → 与证据集合比对
                      （委托单号 / 样本号 / 金额 / 节点状态 / 检测周期 / 基因与变异）
        V4 动作校验   动作码白名单 + 委托单已查询 + 节点满足加急/复检/退单条件
        V5 纠错与兜底  由 refine.py 编排：反馈错误重写 1 次，仍不过换兜底

    配套：correction_feedback() 拼纠错反馈文本（V5 的"喂回去的错误"）。

不负责：
    生成回复（agent.py）、收口状态机（refine.py）、执行工具（tools.py）、
    提示词文案（prompts.py）、观测落库（run_log.py）。

# V5 的重试循环为什么不放在这里？
# 因为"重写"需要再调一次模型，那是编排行为。harness 若自己调模型，
# 就变成了"既当裁判又当教练"，单元测试也得 mock 模型才能跑。
# 拆开后 harness 是纯函数（进：输出+证据，出：结论），测试不需要任何 mock；
# refine 拿着结论决定"补查、重写还是兜底"。判断和执行分离，两边都简单。

证据规则：
    只认本次运行 run_trace 里 success=True 的 get_user_orders / search_products /
    get_report_variants；search_knowledge 的结果永远不进证据 ——
    SOP/政策内容是"说法"不是"业务事实"，AI 引用政策说"怎么送样本"无需证据，
    说"您的委托单 1 是 3600 元"必须有工具背书。

# 本域的"不误杀"纪律（宁可漏杀，不能把正常回复打回）：
#   1. 「7 个工作日」的 7 只参与周期比对，不当金额（与原版"7 天退货"同构）；
#   2. 「未检出 / 未发现致病变异」是阴性报告的有效事实，基因/变异的否定语境一律豁免；
#   3. FFPE/DNA/VAF/MRD 这类大写缩写不是基因名，进豁免词表，否则句句误杀；
#   4. 加急周期 3 个工作日是政策常量（不出现在任何单据字段里），
#      但模型复述它属于合法事实 —— 必须预先放进合法周期集合。
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from app.prompts import CORRECTION_FEEDBACK_TEMPLATE, TOOL_FAIL_CONTENT
from app.schemas import ALLOWED_ACTION_CODES, AgentOutput, ReplyType, ToolCallRecord

logger = logging.getLogger(__name__)

# 错误码与领域常量

# 稳定错误码：前端和 agent_runs 表都按它过滤统计
MODEL_OUTPUT_INVALID = "MODEL_OUTPUT_INVALID"          # 结构/组合不合法
UNSUPPORTED_FACT = "UNSUPPORTED_FACT"                  # 单号/样本号/金额/节点/基因等查无实据
UNVERIFIED_ACTION_RESOURCE = "UNVERIFIED_ACTION_RESOURCE"  # 动作无依据或不可执行
TIER_MISMATCH = "TIER_MISMATCH"                        # 变异分级错报（如 Tier III 说成 Tier I）

# 委托单号：只认"委托单 + 数字"的写法，"委托单 1""委托单编号 3"都命中。
# 本域用户的口头习惯是"单子 1"，但模型被 prompt 约束用"委托单 N"的写法。
ORDER_ID_RE = re.compile(r"委托单\s*(?:编号\s*)?(\d+)")

# 样本号：HP-YYYY-NNNNN。大写固定前缀，无需大小写容错。
SAMPLE_NO_RE = re.compile(r"HP-\d{4}-\d{5}")

# 金额：只认带业务标签的数字（合计/总价/金额/价格/售价/费用……元）。
# 这是控制误杀的关键 —— "7 个工作日"的 7、"520 基因"的 520 都不带金额标签，
# 一律不管。宁可漏杀（放过个别编造），不能误杀（把正常回复打回）。
AMOUNT_RE = re.compile(r"(?:合计|总价|金额|价格|售价|费用)\D{0,3}(\d+(?:\.\d{1,2})?)\s*元")

# 检测周期：数字 + （个）工作日/自然日/天。只参与"周期比对"，不当金额。
TURNAROUND_RE = re.compile(r"(\d+)\s*个?\s*(?:工作日|自然日|天)")

# 基因名/大写缩写：独立的"字母(+连字符)+结尾数字(≤2)"词形（BRCA1 / TP53 / ERBB2）。
# 注意一：必须用 ASCII 边界（lookaround）而不是 \b —— \b 对中文是单词字符，
#   "检出EGFR"里"出"和"E"之间没有 \b 边界，会漏掉。
# 注意二：词形刻意排除"字母+中间数字+字母"（L858R / G12C 式的蛋白写法）——
#   那是变异位点不是基因，当成基因比对必然误杀。数学上：基因符号的数字只在结尾
#   （BRCA1），蛋白突变的数字夹在字母中间（L858R），用 \d{0,2}$ 尾缀即可区分。
# 注意三：两侧边界排除 '_' —— 模型偶尔把动作码 APPLY_URGENT 复述进正文，
#   不排除的话 APPLY/URGENT 会被切成"未知基因"误杀。
GENE_TOKEN_RE = re.compile(r"(?<![A-Za-z0-9.>_-])[A-Z][A-Z-]{1,9}\d{0,2}(?![A-Za-z0-9_-])")

# 变异位点：c.2573T>C / p.L858R 这类 HGVS 写法（前缀不能是字母数字，防止拼进词里）
VARIANT_RE = re.compile(r"(?<![A-Za-z0-9.])[cp]\.[A-Za-z0-9]+(?:>[A-Za-z0-9]+)?")

# 裸蛋白写法：L858R / G12C（不带 p. 前缀）。口语里"EGFR L858R"极常见，
# 提取后统一归一成 "p.L858R" 再与证据比对 —— 否则编造的 "EGFR L999R" 会漏杀。
# 词形单字母+≥2数字+单字母在本域没有合法的其他用途，误杀面极小。
BARE_PROTEIN_RE = re.compile(r"(?<![A-Za-z0-9.>])[A-Z]\d{2,}[A-Z](?![A-Za-z0-9])")

# Tier 分级：Tier I/II/III/IV 或 Tier 1~4
TIER_RE = re.compile(r"Tier\s*[-–]?\s*(I{1,3}|IV|[1-4])", re.IGNORECASE)
_TIER_MAP = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}

# 节点状态词表：覆盖 orders.status 的全部 9 档真实取值。
# 注意：状态校验不能只做"这个词出现过没有"——
# 用户有多张委托单时，"委托单1已发报告"里的"已发报告"可能来自委托单2，
# 集合比对会漏杀。所以 _check_facts 之后还有一道逐条一致性检查。
STATUS_WORDS = {
    "已接收": "received",
    "质检合格": "qc_passed",
    "质检不合格": "qc_failed",
    "建库完成": "library_built",
    "上机测序中": "sequencing",
    "生信分析中": "analyzing",
    "报告审核中": "report_review",
    "已发报告": "report_issued",
    "已退单": "cancelled",
}

# V4 放行条件（与 api_actions.py 的 UPDATE 条件、语料政策文档三方逐字一致）
URGENT_ALLOWED_STATUS = frozenset({"received", "qc_passed", "library_built"})   # 加急：未上机
CANCEL_ALLOWED_STATUS = frozenset({"received", "qc_passed", "qc_failed"})        # 退单：未建库
RETEST_WINDOW_DAYS = 7                                                          # 复检：发报告后 7 日内

# 政策常量：加急检测周期。不出现在任何单据字段，但属于模型可合法复述的事实
#（详见模块 docstring 的"不误杀"第 4 条）。
URGENT_TURNAROUND_DAYS = 3

# HGNC 常见基因白名单（文件内常量，约 40 个）∪ 本轮工具查到的基因。
# 编造基因是本域最高危的幻觉（"EGFR L858R"编成"EGFR L999R"甚至编出"EGFRX"），
# 不在白名单判 UNSUPPORTED_FACT。
GENE_WHITELIST = frozenset({
    "EGFR", "KRAS", "ALK", "ROS1", "BRAF", "ERBB2", "PIK3CA", "TP53", "KEAP1",
    "RBM10", "MET", "RET", "NTRK1", "NTRK2", "NTRK3", "BRCA1", "BRCA2", "PALB2",
    "CHEK2", "ATM", "PTEN", "STK11", "NF1", "NRAS", "MAP2K1", "ERBB3", "FGFR1",
    "FGFR2", "FGFR3", "KIT", "PDGFRA", "IDH1", "IDH2", "TERT", "CTNNB1", "APC",
    "SMARCA4", "ARID1A", "CDKN2A", "CCNE1", "MYC", "RB1", "AKT1", "ERCC1",
})

# 基因别名归一化：用户/模型说别名，库里存 HGNC 正名，归一后再比对。
# 不归一会出现"用户问 HER2、库里存 ERBB2"被判编造的假阳性，必须消除。
GENE_ALIASES = {"HER2": "ERBB2"}

# 非基因的大写缩写词表：FFPE/DNA/VAF/MRD 这类词在回复里高频出现，
# 它们长得和基因符号一模一样，不豁免就句句误杀（见模块 docstring 第 3 条）。
# TIER/II/III/IV 是分级写法，同样不是基因。
NON_GENE_TOKENS = frozenset({
    "FFPE", "DNA", "RNA", "NGS", "MRD", "HRD", "VAF", "WES", "SOP", "RPT",
    "NSCLC", "QC", "PCR", "MSI", "TMB", "SNV", "CNV", "LOH", "HP", "CFDNA",
    "TIER", "II", "III", "IV",
})

# 否定语境线索：出现在单号/样本号**前面**时，这个编号不算"业务事实"。
# 用于区分两种句子：
#     "样本 HP-2026-01001 已接收"   —— 正面陈述，必须有证据，拦
#     "没有样本 HP-2026-99999"     —— 否定陈述，说的就是"它不存在"，
#                                     证据里没有它恰恰是对的
# 没有这条豁免，用户问一个不存在的样本号，AI 如实回答"查无此样本"也会被打回，
# 兜底成"转人工"—— 用户被无端拒绝，这正是 harness 最不该犯的错。
_ADJACENT_CUES = ("没有", "无", "未", "查不到", "找不到", "不存在")
_PHRASE_CUES = ("查不到", "找不到", "没有查到", "未找到", "没有找到", "不存在", "无此", "没有这个")

# 基因的否定语境：阴性报告话术"未检出 EGFR 变异"里的基因不该被要求有证据
#（见模块 docstring"不误杀"第 2 条 —— 「未检出」是有效事实）。
_GENE_NEGATION_CUES = ("未检出", "没有检出", "未发现", "没有发现", "未检测到", "未测到", "无")

# 分句符：逐条核对只在"一个分句里恰好一个锚点 + 一个状态词/分级"时生效。
# 为什么必须分句：模型正常回答是"委托单1已接收，委托单2已发报告"，
# 整句看有两个单号、两个状态，无法归属；切开后每个小句都能对上号。
_CLAUSE_SPLIT_RE = re.compile(r"[，。；;！!？?\n]")


# 数据结构


@dataclass
class Evidence:
    """本次运行从工具结果里构建的"盖章答案"。"""

    order_ids: set[int] = field(default_factory=set)              # 查到的委托单号
    sample_nos: set[str] = field(default_factory=set)             # 查到的样本号
    amounts: set[Decimal] = field(default_factory=set)             # 合法金额（总额+单价+产品售价）
    statuses: set[str] = field(default_factory=set)               # 出现过的节点码
    order_status: dict[int, str] = field(default_factory=dict)     # 委托单号→节点
    sample_status: dict[str, str] = field(default_factory=dict)    # 样本号→节点（同单）
    order_is_urgent: dict[int, bool] = field(default_factory=dict)  # 委托单号→加急标记
    order_report_issued: dict[int, datetime] = field(default_factory=dict)  # 单号→报告签发时间
    turnaround_days: set[int] = field(default_factory=set)         # 合法周期值
    genes: set[str] = field(default_factory=set)                   # 查到的基因（归一化后）
    variants: set[str] = field(default_factory=set)                # 查到的 c./p. 变异写法
    variant_tier: dict[str, int] = field(default_factory=dict)     # 变异写法→Tier（1~4）
    has_order_query: bool = False                                  # 本次是否成功查过委托单
    has_report_query: bool = False                                 # 本次是否成功查过报告


@dataclass
class ExtractedFacts:
    """从回复正文提取的六类业务事实（正则只干"提取"，不干"判断"）。"""

    order_ids: set[int] = field(default_factory=set)
    sample_nos: set[str] = field(default_factory=set)
    amounts: set[Decimal] = field(default_factory=set)
    status_words: set[str] = field(default_factory=set)
    turnaround_days: set[int] = field(default_factory=set)
    genes: set[str] = field(default_factory=set)      # 归一化后
    variants: set[str] = field(default_factory=set)


@dataclass
class HarnessResult:
    """一次 validate 的完整结论。"""

    passed: bool                          # True 才允许发给用户
    output: AgentOutput                   # 最终输出（V2 降级时已被替换成话术）
    error_codes: list[str] = field(default_factory=list)
    error_message: str = ""               # 喂给模型重写的具体错误（V5 用）
    needs_order_query: bool = False       # 编排层据此强制补查一次（见下方说明）
    degraded: str | None = None           # 'tool_failure' / None

    # 为什么要 needs_order_query 这个字段？
    # 实测发现：用户追问"帮我加急委托单1"时，模型可能根本不调工具，
    # 直接沿用上一轮历史里的节点信息作答（日志里表现为 工具=0次）。
    # 此时 V3/V4 失败的原因是"没有证据"，而不是"证据矛盾"——
    # 直接打回重写，模型手里还是没有证据，只会被无端拒绝（用户被误伤）。
    # 正确动作是先强制补查一次 get_user_orders，再拿新证据复验。
    # "该不该补查"是判断，harness 给出；"怎么补查"是执行，agent 负责。


# 对外入口


def validate(output: AgentOutput, tool_trace: list[ToolCallRecord]) -> HarnessResult:
    """
    五关卡校验（V5 的重写循环在 agent.py，本函数只判断一轮）。

    Args:
        output: agent 结构化出的 AgentOutput。
        tool_trace: 本次运行的工具调用记录（证据唯一来源）。

    Returns:
        HarnessResult。passed=True 表示可以发给用户（V2 降级也算"通过"，
        但 output 已被替换且 degraded='tool_failure'）。
    """
    # ---------- V1 结构：组合规则 ----------
    err = _check_structure(output)
    if err:
        return HarnessResult(
            passed=False,
            output=output,
            error_codes=[MODEL_OUTPUT_INVALID],
            error_message=err,
        )

    # ---------- V2 前置降级：业务工具全部失败 ----------
    if _all_business_tools_failed(tool_trace):
        degraded_output = AgentOutput(
            reply_type=ReplyType.REQUEST_HANDOFF,
            content=TOOL_FAIL_CONTENT,
            action=None,
        )
        logger.warning("V2 前置降级 | 业务工具全部失败，静默换固定话术")
        return HarnessResult(passed=True, output=degraded_output, degraded="tool_failure")

    # ---------- V3 事实：六类比对 + 两道一致性补强 ----------
    evidence = build_evidence(tool_trace)
    facts = extract_facts(output.content)

    v3 = (
        _check_facts(facts, evidence)
        or _check_status_consistency(output.content, evidence)
        or _check_tier_consistency(output.content, evidence)
    )
    if v3:
        code, msg = v3
        # 只有"提到单号/样本号但从未查过单"才值得补查；查过仍对不上就是编造
        return HarnessResult(
            passed=False,
            output=output,
            error_codes=[code],
            error_message=msg,
            needs_order_query=(
                bool(facts.order_ids or facts.sample_nos) and not evidence.has_order_query
            ),
        )

    # ---------- V4 动作 ----------
    v4_msg = _check_action(output, evidence)
    if v4_msg:
        return HarnessResult(
            passed=False,
            output=output,
            error_codes=[UNVERIFIED_ACTION_RESOURCE],
            error_message=v4_msg,
            # 按钮指向的委托单不在证据里，且从未查过单 → 补查可能救回来
            needs_order_query=(
                output.action is not None
                and output.action.order_id not in evidence.order_ids
                and not evidence.has_order_query
            ),
        )

    return HarnessResult(passed=True, output=output)


def build_evidence(tool_trace: list[ToolCallRecord]) -> Evidence:
    """
    从工具轨迹构建证据集合。

    只收 success=True 的 get_user_orders / search_products / get_report_variants；
    search_knowledge 的结果永远不进证据（SOP 是说法，不是业务事实）。

    # 归属核验为什么不需要单独做？
    # get_user_orders / get_report_variants 的 SQL 写死了 WHERE user_id=JWT 用户
    # （见 tools.py），证据集合天然只含本人单据 —— 林教授的委托单 4 永远进不了
    # 陈医生的证据，越权动作会在 V4 的"单号不在证据里"这一条被自然拦掉。
    """
    ev = Evidence()
    for record in tool_trace:
        if not record.success or record.result is None:
            continue

        if record.tool_name == "get_user_orders":
            ev.has_order_query = True
            for order in record.result.get("orders", []):
                ev.order_ids.add(order["id"])
                ev.statuses.add(order["status"])
                ev.order_status[order["id"]] = order["status"]
                ev.order_is_urgent[order["id"]] = bool(order["is_urgent"])
                ev.amounts.add(Decimal(str(order["total_amount"])))
                for item in order.get("items", []):
                    ev.sample_nos.add(item["sample_no"])
                    ev.sample_status[item["sample_no"]] = order["status"]
                    ev.amounts.add(Decimal(str(item["unit_price"])))
                    ev.turnaround_days.add(int(item["turnaround_days"]))

        elif record.tool_name == "search_products":
            for product in record.result.get("products", []):
                ev.amounts.add(Decimal(str(product["price"])))
                ev.turnaround_days.add(int(product["turnaround_days"]))

        elif record.tool_name == "get_report_variants":
            ev.has_report_query = True
            for report in record.result.get("reports", []):
                ev.order_ids.add(report["order_id"])
                ev.order_status[report["order_id"]] = "report_issued"
                ev.statuses.add("report_issued")
                ev.order_report_issued[report["order_id"]] = report["issued_at"]
                for v in report.get("variants", []):
                    ev.genes.add(_normalize_gene(v["gene"]))
                    if v.get("c_change"):
                        ev.variants.add(v["c_change"])
                        ev.variant_tier[v["c_change"]] = int(v["tier"])
                    if v.get("p_change"):
                        ev.variants.add(v["p_change"])
                        ev.variant_tier[v["p_change"]] = int(v["tier"])
    return ev


def extract_facts(text: str) -> ExtractedFacts:
    """
    从回复正文提取业务事实（正则在这里只干"提取"这一件事）。

    Args:
        text: AgentOutput.content。

    Returns:
        ExtractedFacts。提取结果为空 = 这句话不含需要证据的业务事实，直接放行。

    # 正则只提取、不判断，这是本文件最重要的纪律。
    # "样本号对不对"是集合成员问题，正则管不了：金额的合法写法无穷多
    # （3600 元 / ¥3,600.00 / 3600.0），试图用一个大正则判断"整句是否合法"
    # 必然枚举不全。先提取、再用 Decimal 精确比对，把问题降维成查表。
    """
    facts = ExtractedFacts()
    _extract_order_ids(text, facts)
    _extract_sample_nos(text, facts)
    facts.amounts = {Decimal(m) for m in AMOUNT_RE.findall(text)}
    facts.status_words = {w for w in STATUS_WORDS if w in text}
    facts.turnaround_days = {int(m) for m in TURNAROUND_RE.findall(text)}
    _extract_genes_and_variants(text, facts)
    return facts


def correction_feedback(error_message: str) -> str:
    """拼 V5 纠错反馈的完整文本（模板在 prompts.py，话术统一放一处）。"""
    return CORRECTION_FEEDBACK_TEMPLATE.format(error=error_message)


# 提取的内部实现


def _negated(text: str, pos: int, window: int = 8) -> bool:
    """pos 前面的窗口里是否出现否定语境（单号/样本号共用，线索表见上方说明）。"""
    prefix = text[max(0, pos - window) : pos]
    return prefix.endswith(_ADJACENT_CUES) or any(cue in prefix for cue in _PHRASE_CUES)


def _extract_order_ids(text: str, facts: ExtractedFacts) -> None:
    """提取"委托单 N"。否定语境（查无此单）豁免 —— 见 _ADJACENT_CUES 的说明。"""
    for m in ORDER_ID_RE.finditer(text):
        if not _negated(text, m.start()):
            facts.order_ids.add(int(m.group(1)))


def _extract_sample_nos(text: str, facts: ExtractedFacts) -> None:
    """提取样本号 HP-YYYY-NNNNN，否定语境豁免（理由同上）。"""
    for m in SAMPLE_NO_RE.finditer(text):
        if not _negated(text, m.start()):
            facts.sample_nos.add(m.group(0))


def _normalize_gene(token: str) -> str:
    """基因别名归一化（HER2→ERBB2）。归一化只用于比对，不改下发文案。"""
    return GENE_ALIASES.get(token.upper(), token.upper())


def _variant_tokens(text: str) -> set[str]:
    """
    提取一段文本里的全部变异写法，归一到证据的存储口径。

    两种来源：① 带 c./p. 前缀的 HGVS 写法原样收；② 裸蛋白写法（L858R）
    补上 "p." 前缀再收。证据集合里存的是 report_variants 的 c_change/p_change
    原文，归一化后两边口径才能对上。
    """
    tokens = set(VARIANT_RE.findall(text))
    for m in BARE_PROTEIN_RE.finditer(text):
        tokens.add("p." + m.group(0))
    return tokens


def _extract_genes_and_variants(text: str, facts: ExtractedFacts) -> None:
    """
    提取基因名与变异位点。

    基因要做两道豁免：非基因缩写词表（FFPE/DNA/VAF…）+ 否定语境（未检出…）。
    变异位点不需要否定豁免 —— "未检出 p.L858R" 这种话里 p.L858R 出现在
    否定语境之外的概率极低，且变异集合比对本身足够特异，误杀面小。
    """
    for m in GENE_TOKEN_RE.finditer(text):
        token = m.group(0)
        if token in NON_GENE_TOKENS:
            continue
        prefix = text[max(0, m.start() - 8) : m.start()]
        if any(cue in prefix for cue in _GENE_NEGATION_CUES):
            continue
        facts.genes.add(_normalize_gene(token))
    facts.variants.update(_variant_tokens(text))


# 各关卡内部实现


def _check_structure(output: AgentOutput) -> str | None:
    """
    V1：组合规则（字段类型已由 Pydantic 保证，这里只查跨字段契约）。

    Returns:
        错误描述（None 表示通过）。
    """
    if not (output.content or "").strip():
        return "content 为空字符串"
    if output.action is not None and output.reply_type != ReplyType.ACTION_GUIDE:
        return f"action 只能出现在 ACTION_GUIDE 中，当前 reply_type 是 {output.reply_type.value}"
    if output.reply_type == ReplyType.ACTION_GUIDE and output.action is None:
        return "ACTION_GUIDE 必须携带 action（code + order_id）"
    return None


def _all_business_tools_failed(tool_trace: list[ToolCallRecord]) -> bool:
    """
    V2：业务工具（委托单/产品/报告）是否"调过且全部失败"。

    两个容易搞错的边界：
        - 一个都没调 → 不算失败（走 V3，由证据缺失那套逻辑处理）；
        - search_knowledge 失败不算 —— "知识库里没写"是正常业务结果，
          不是基础设施故障，模型应该转人工而不是被换话术。
    """
    business = [
        r for r in tool_trace
        if r.tool_name in ("get_user_orders", "search_products", "get_report_variants")
    ]
    return bool(business) and not any(r.success for r in business)


def _check_facts(facts: ExtractedFacts, evidence: Evidence) -> tuple[str, str] | None:
    """
    V3：提取出的事实逐个到证据集合里找，找不到的收集起来。

    Returns:
        (错误码, 错误描述) 或 None。错误必须具体到"哪个事实没依据"，
        否则模型不知道该删哪句话。

    # 金额比对为什么必须用 Decimal？
    # 数据库里是 NUMERIC("3600.00")，模型嘴里的可能是"3600"或"3600.0"。
    # float 会先把 3600.00 变成 3599.9999999 再比较；Decimal 则保证
    # Decimal("3600.00") == Decimal("3600") 为真，格式差异自动消失。

    # 周期比对为什么要并入常量 3？
    # 合法周期 = 工具查到的 turnaround_days ∪ {加急周期 3}。
    # 3 不在任何单据字段里，但"加急后 3 个工作日"是政策原文的合法复述；
    # 不并入的话，模型转述加急政策这句话会被当成编造周期而打回（假阳性）。
    """
    problems: list[str] = []

    bad_ids = facts.order_ids - evidence.order_ids
    if bad_ids:
        problems.append("委托单" + "、".join(str(i) for i in sorted(bad_ids)))

    bad_samples = facts.sample_nos - evidence.sample_nos
    if bad_samples:
        problems.append("样本" + "、".join(sorted(bad_samples)))

    bad_amounts = facts.amounts - evidence.amounts
    if bad_amounts:
        problems.append("金额" + "、".join(str(a) for a in sorted(bad_amounts)) + "元")

    # 状态词先过映射表转成节点码，再和证据里的节点码集合比对
    bad_statuses = [w for w in sorted(facts.status_words) if STATUS_WORDS[w] not in evidence.statuses]
    if bad_statuses:
        problems.append("节点" + "、".join(bad_statuses))

    legal_days = evidence.turnaround_days | {URGENT_TURNAROUND_DAYS}
    bad_days = facts.turnaround_days - legal_days
    if bad_days:
        problems.append("周期" + "、".join(f"{d} 个工作日" for d in sorted(bad_days)))

    # 基因：白名单 ∪ 本轮查到的基因。编造基因是最高危幻觉，必须拦
    bad_genes = facts.genes - (GENE_WHITELIST | evidence.genes)
    if bad_genes:
        problems.append("基因" + "、".join(sorted(bad_genes)))

    bad_variants = facts.variants - evidence.variants
    if bad_variants:
        problems.append("变异" + "、".join(sorted(bad_variants)))

    if not problems:
        return None
    return UNSUPPORTED_FACT, "回复中的业务事实缺少工具支持：" + "；".join(problems)


def _check_status_consistency(text: str, evidence: Evidence) -> tuple[str, str] | None:
    """
    V3 补强①：逐条核对"委托单 N / 样本号 + 节点"是否与证据一致（防跨单串节点）。

    Returns:
        (错误码, 错误描述) 或 None（无法归属时放过，不误杀）。

    # 为什么光有集合比对不够？
    # evidence.statuses 是"该用户全部委托单节点的并集"。
    # 一个用户有 7 张单（覆盖 9 档节点）时，模型把委托单 1 说成"已发报告"
    # 也能通过集合比对 —— 这个词确实在证据里，只是属于委托单 2。
    # 这与本项目早期"把 A 单的样本金额安到 B 单"是同一类错误，
    # 只有把节点重新绑定到单号/样本号才能拦住。
    # 反过来，多单/多节点的句子无法归属，宁可放过也不能误杀，
    # 所以只在"恰好一个锚点并恰好一个节点词"时下结论。
    """
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        ids = {int(m.group(1)) for m in ORDER_ID_RE.finditer(clause)}
        samples = set(SAMPLE_NO_RE.findall(clause))
        words = {w for w in STATUS_WORDS if w in clause}
        if len(words) != 1 or (len(ids) + len(samples)) != 1:
            continue

        word = next(iter(words))
        if ids:
            order_id = next(iter(ids))
            actual = evidence.order_status.get(order_id)
            anchor = f"委托单{order_id}"
        else:
            sample_no = next(iter(samples))
            actual = evidence.sample_status.get(sample_no)
            anchor = f"样本{sample_no}"
        if actual is not None and actual != STATUS_WORDS[word]:
            return UNSUPPORTED_FACT, f"{anchor}的真实节点是 {actual}，回复里写成了「{word}」"
    return None


def _check_tier_consistency(text: str, evidence: Evidence) -> tuple[str, str] | None:
    """
    V3 补强②：逐条核对"变异 + Tier"是否与证据一致（Tier III 说成 Tier I 必须单列拦截）。

    Returns:
        (TIER_MISMATCH, 错误描述) 或 None。

    # 为什么 Tier 错报要单列错误码？
    # UNSUPPORTED_FACT 是"查无实据"（编了个不存在的变异），
    # TIER_MISMATCH 是"实据存在但说错了等级"—— 在医疗语境里后者更危险：
    # Tier I 和 Tier III 对医生的意义完全不同，说错等级比编造位点更隐蔽。
    # 单列错误码后，agent_runs 里能直接统计"分级错报率"这个独有指标。
    # 同样遵循"恰好一个变异 + 恰好一个分级"才下结论，避免误杀。
    """
    for clause in _CLAUSE_SPLIT_RE.split(text or ""):
        variants = _variant_tokens(clause)
        tiers = _tier_values(clause)
        if len(variants) != 1 or len(tiers) != 1:
            continue
        variant = next(iter(variants))
        actual = evidence.variant_tier.get(variant)
        if actual is not None and actual != tiers[0]:
            return (
                TIER_MISMATCH,
                f"变异 {variant} 的真实分级是 Tier {_TIER_ROMAN[actual]}，回复里写成了 Tier {_TIER_ROMAN[tiers[0]]}",
            )
    return _check_tier_whole_text(text, evidence)


def _check_tier_whole_text(text: str, evidence: Evidence) -> tuple[str, str] | None:
    """
    Tier 比对的整句兜底：全文恰好一个变异 + 一个分级时也能锚定。

    # 为什么状态比对不需要这道兜底而 Tier 需要？
    # 复述节点的回答天然是"委托单1已接收，委托单2已发报告"式的多锚点句，
    # 只能分句归属；而"这个变异是 Tier 几"的回答通常全文只有一条变异 ——
    # 模型爱把分级写在括号里（EGFR L858R（Tier I）），也爱用逗号隔开写
    # （检出 EGFR L858R，分级为 Tier I）。后者整句被逗号切开就两头落空，
    # 用整句兜底补上，恰好一变异一分级的精度足够不误杀。
    """
    variants = _variant_tokens(text or "")
    tiers = _tier_values(text or "")
    if len(variants) != 1 or len(tiers) != 1:
        return None
    variant = next(iter(variants))
    actual = evidence.variant_tier.get(variant)
    if actual is not None and actual != tiers[0]:
        return (
            TIER_MISMATCH,
            f"变异 {variant} 的真实分级是 Tier {_TIER_ROMAN[actual]}，回复里写成了 Tier {_TIER_ROMAN[tiers[0]]}",
        )
    return None


_TIER_ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV"}


def _tier_values(clause: str) -> list[int]:
    """把一个分句里的 Tier 写法全部转成数字分级值。"""
    return [_TIER_MAP[m.group(1).upper()] for m in TIER_RE.finditer(clause)]


def _check_action(output: AgentOutput, evidence: Evidence) -> str | None:
    """
    V4：动作校验（查表 + 集合，零正则）。按三个动作码各自的放行条件核对。

    Returns:
        错误描述（None 表示通过）。没有 action 时直接通过。

    # 为什么三个动作的放行条件要在这里"再写一遍"而不是复用一个表？
    # 加急看"未上机"、复检看"qc_failed 或发报告 7 日内"、退单看"未建库"——
    # 三条条件的形状完全不同（集合 vs 或组合 vs 带时间窗），硬抽成一个通用
    # 数据结构要么表达不了时间窗，要么把条件拆散到没法读。
    # 各写四行直白代码，可读性就是最好的架构。
    """
    action = output.action
    if action is None:
        return None

    if action.code not in ALLOWED_ACTION_CODES:
        return f"动作码 {action.code} 不在白名单内（仅支持 APPLY_URGENT / APPLY_RETEST / APPLY_CANCEL）"

    if action.order_id not in evidence.order_ids:
        return f"按钮指向的委托单{action.order_id}未经本次工具查询确认（先查后引导）"

    status = evidence.order_status.get(action.order_id)

    if action.code == "APPLY_URGENT":
        if status not in URGENT_ALLOWED_STATUS:
            return f"委托单{action.order_id}节点为 {status}，只有未上机（已接收/质检合格/建库完成）的单才能加急"
        if evidence.order_is_urgent.get(action.order_id):
            return f"委托单{action.order_id}已申请过加急，不能重复引导"

    elif action.code == "APPLY_RETEST":
        if status == "qc_failed":
            return None  # 质检不合格：无条件放行（复检是补救手段）
        if status != "report_issued":
            return f"委托单{action.order_id}节点为 {status}，质检不合格或已发报告 7 日内的单才能复检"
        issued = evidence.order_report_issued.get(action.order_id)
        if issued is None or _as_utc(issued) < datetime.now(timezone.utc) - timedelta(days=RETEST_WINDOW_DAYS):
            return f"委托单{action.order_id}的报告签发已超过 {RETEST_WINDOW_DAYS} 日复检窗口，不能引导复检"

    else:  # APPLY_CANCEL
        if status not in CANCEL_ALLOWED_STATUS:
            return f"委托单{action.order_id}节点为 {status}，只有未建库（已接收/质检合格/质检不合格）的单才能退单"

    return None


def _as_utc(dt: Any) -> datetime:
    """把报告签发时间统一成带 UTC 时区的 datetime（种子里的 timestamptz 本就带时区）。"""
    if isinstance(dt, str):  # 工具下发的是 isoformat 字符串（tools.py 统一转过了）
        dt = datetime.fromisoformat(dt)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt
