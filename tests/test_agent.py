"""
test_agent.py —— 编排层测试

真正的模型调用一律 monkeypatch 掉，用"假模型"精确控制每一轮的返回，
这样能稳定复现这些平时很难碰到的分支：
    - 模型要工具 → 工具执行 → 结果回填 → 再答（多轮循环）
    - 工具轮数耗尽时强制收口
    - 模型彻底失败 → 兜底话术（而不是 500）
    - 结构化输出裹了 ```json 代码块 / 带前缀废话，仍然能解析
    - 结构化两次都失败 → 兜底

这些都是线上真实会发生的路径，但靠真模型跑一次碰不到、也测不稳。
"""

import json

import pytest

from app import agent, refine
from app.llm import LLMError, LLMReply, ToolCallRequest


def _reply(content=None, tool_calls=None):
    """造一个假模型回复。"""
    return LLMReply(content=content, tool_calls=tool_calls or [], model="fake")


def _tool_call(name, arguments, call_id="call_1"):
    return ToolCallRequest(id=call_id, name=name, arguments=arguments)


class FakeChat:
    """
    按预设剧本逐次返回的假 chat()。

    用法：FakeChat([第一轮回复, 第二轮回复, ...])
    剧本用完还在被调用 → 直接抛 AssertionError，避免"代码多调了一次模型"被悄悄放过。
    """

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []  # 记录每次调用收到的 messages / tools，供断言

    def __call__(self, messages, tools=None, json_mode=False):
        self.calls.append({"messages": messages, "tools": tools, "json_mode": json_mode})
        if not self.replies:
            raise AssertionError(f"模型被多调用了一次（第 {len(self.calls)} 次），剧本已用完")
        return self.replies.pop(0)


@pytest.fixture
def patch_chat(monkeypatch):
    """
    同时替换 agent 和 refine 两个模块里的 chat。

    # 为什么必须替换两处？
    # 工具循环在 agent.py、结构化调用在 refine.py，两边写的都是
    # `from app.llm import chat`——函数对象在各自模块加载时就绑定了。
    # 只 patch 其中一个，另一个照样去调真模型（被 _no_real_llm 拦下报错），
    # 而且报错信息会误导你去查 mock，而不是想到还有第二个绑定。
    """

    def _install(replies):
        fake = FakeChat(replies)

        def _shared(*args, **kwargs):
            return fake(*args, **kwargs)

        monkeypatch.setattr(agent, "chat", _shared)
        monkeypatch.setattr(refine, "chat", _shared)
        return fake

    return _install


# 正常路径


def test_plain_answer_without_tool(patch_chat):
    """模型直接回答、不调工具 → 结构化后得到 ANSWER。"""
    patch_chat(
        [
            _reply("您好，请问有什么可以帮您？"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "您好，请问有什么可以帮您？", "action": None})),
        ]
    )

    run = agent.run("你好", [], user_id=1, nickname="陈医生")

    assert run.error is None
    assert run.output.reply_type.value == "ANSWER"
    assert run.output.content == "您好，请问有什么可以帮您？"
    assert run.output.action is None
    assert run.tool_trace == []


def test_tool_loop_then_answer(patch_chat, db_isolation):
    """
    模型先要工具、拿到真实数据后再回答 —— 这是主流程。

    断言的重点不是模型说了什么，而是：
        工具真的被执行了、结果真的回填进上下文、trace 里留下了证据。
    """
    fake = patch_chat(
        [
            _reply(tool_calls=[_tool_call("get_user_orders", "{}")]),
            _reply("您有 7 张委托单，其中委托单 1 已接收。"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "您有 7 张委托单，其中委托单 1 已接收。", "action": None})),
        ]
    )

    run = agent.run("我有哪些委托单", [], user_id=1)

    # 三轮：调工具 → 收口作答 → 结构化
    assert len(fake.calls) == 3
    # 工具真的执行成功，且 trace 里有记录
    assert len(run.tool_trace) == 1
    assert run.tool_trace[0].tool_name == "get_user_orders"
    assert run.tool_trace[0].success is True
    assert len(run.tool_trace[0].result["orders"]) == 7

    # 工具结果以 role=tool 回填，且是 indent=2 展开的（防止跨委托单串数据的那个修复）
    tool_msgs = [m for m in run.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "\n  " in tool_msgs[0]["content"], "工具结果应为缩进展开的 JSON，而不是压成一行"

    # 第一轮带工具定义，结构化那轮不带
    assert fake.calls[0]["tools"] is not None
    assert fake.calls[2]["tools"] is None
    assert fake.calls[2]["json_mode"] is True


def test_action_guide_with_urgent(patch_chat, db_isolation):
    """模型给出加急建议 → 解析出 action，order_id 必须是真实的。"""
    patch_chat(
        [
            _reply(tool_calls=[_tool_call("get_user_orders", "{}")]),
            _reply("委托单 1 是已接收状态，可以加急，请点击下方按钮确认。"),
            _reply(
                json.dumps(
                    {
                        "reply_type": "ACTION_GUIDE",
                        "content": "委托单 1 是已接收状态，可以加急，请点击下方按钮确认。",
                        "action": {"code": "APPLY_URGENT", "order_id": 1},
                    }
                )
            ),
        ]
    )

    run = agent.run("帮我加急委托单1", [], user_id=1)

    assert run.output.reply_type.value == "ACTION_GUIDE"
    assert run.output.action is not None
    assert run.output.action.code == "APPLY_URGENT"
    assert run.output.action.order_id == 1


def test_history_is_trimmed_and_ordered(patch_chat):
    """
    历史只取最近 N 条，且顺序不能被破坏。

    # 历史取反了（新的在前、旧的在后）模型就会把因果弄反，
    # 表现是"回答上一轮的问题"，而且这种错非常隐蔽 —— 单轮测试永远测不出来。
    """
    patch_chat(
        [
            _reply("好的"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "好的", "action": None})),
        ]
    )

    history = [{"role": "user", "content": f"第{i}问"} for i in range(1, 31)]
    run = agent.run("最后一问", history, user_id=1)

    sent = run.messages
    assert sent[0]["role"] == "system"
    assert sent[-1] == {"role": "user", "content": "最后一问"}
    # 中间的历史条数受 agent_history_limit 限制
    from app.config import settings

    assert len(sent) - 2 <= settings.agent_history_limit
    # 保留下来的必须是"最近的"，即末尾是第 30 问而不是第 1 问
    assert sent[-2]["content"] == "第30问"


def test_history_excludes_previous_nickname_leak(patch_chat):
    """昵称只写进 system prompt，不额外插入一条 user 消息。"""
    patch_chat(
        [
            _reply("在的"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "在的", "action": None})),
        ]
    )

    run = agent.run("在吗", [], user_id=1, nickname="陈医生")

    assert "陈医生" in run.messages[0]["content"]
    assert len(run.messages) == 2  # system + user


# 工具轮数耗尽


def test_tool_round_limit_forces_answer(patch_chat, monkeypatch):
    """
    模型一直要工具 → 到达上限后撤掉工具，逼它按已有信息作答。

    # 这里绝不能抛异常。抛了就是用户白等十几秒还拿不到回复。
    # 正确行为是"用现有证据给一个可能不完美但可信的答复"，剩下的交给 harness 拦。
    """
    from app.config import settings

    monkeypatch.setattr(settings, "agent_max_tool_rounds", 2)

    # 2 轮都请求工具，第 3 次调用是"撤掉工具后的收口"
    fake = patch_chat(
        [
            _reply(tool_calls=[_tool_call("get_user_orders", "{}", "c1")]),
            _reply(tool_calls=[_tool_call("search_products", '{"query":"肺癌"}', "c2")]),
            _reply("根据已查到的信息，您有 7 张委托单。"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "根据已查到的信息，您有 7 张委托单。", "action": None})),
        ]
    )

    run = agent.run("帮我查点东西", [], user_id=1)

    assert run.error is None
    assert run.output.reply_type.value == "ANSWER"
    # 最后一轮必须不带工具
    assert fake.calls[2]["tools"] is None
    # 两轮工具都执行了
    assert len(run.tool_trace) == 2


# 失败与兜底


def test_llm_error_returns_fallback(monkeypatch):
    """
    模型彻底失败 → 兜底话术 + REQUEST_HANDOFF，且 error 里带原因。

    # 为什么兜底是"转人工"而不是"请稍后重试"？
    # 因为此刻系统一条真实信息都没有，说"稍后重试"等于让用户自己猜；
    # 而转人工至少有个人能接手。这也是 harness 降级矩阵里的第一条。
    """

    def boom(*args, **kwargs):
        raise LLMError("主模型失败；备用模型也失败")

    monkeypatch.setattr(agent, "chat", boom)
    run = agent.run("你好", [], user_id=1)

    assert run.output.reply_type.value == "REQUEST_HANDOFF"
    assert run.output.content == agent.FALLBACK_CONTENT
    assert run.output.action is None
    assert run.error.startswith("LLM_ERROR:")


def test_invalid_json_twice_returns_fallback(patch_chat):
    """结构化两次都拿不到合法 JSON → 兜底，并标记 MODEL_OUTPUT_INVALID。"""
    patch_chat(
        [
            _reply("您有 7 张委托单。"),
            _reply("我不是 JSON，我是一段散文。"),
            _reply("我还是散文。"),
        ]
    )

    run = agent.run("我有哪些委托单", [], user_id=1)

    assert run.output.reply_type.value == "REQUEST_HANDOFF"
    assert run.error.startswith("MODEL_OUTPUT_INVALID:")
    # 草稿要留下来，方便排查"模型到底说了什么"
    assert run.draft == "您有 7 张委托单。"


def test_invalid_json_then_fixed(patch_chat):
    """第一次结构化失败、第二次成功 → 正常返回，不兜底。"""
    fake = patch_chat(
        [
            _reply("您有 7 张委托单。"),
            _reply("不是 JSON"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "您有 7 张委托单。", "action": None})),
        ]
    )

    run = agent.run("我有哪些委托单", [], user_id=1)

    assert run.error is None
    assert run.output.content == "您有 7 张委托单。"
    # 第二次请求里必须带上错误原因，否则模型会犯同样的错
    retry_prompt = fake.calls[2]["messages"][-1]["content"]
    assert "未通过服务端校验" in retry_prompt
    assert "不是合法 JSON" in retry_prompt   # 具体错误也要带上


def test_schema_violation_treated_as_invalid(patch_chat):
    """JSON 合法但字段不合契约（reply_type 瞎写）→ 也算失败，走兜底。"""
    patch_chat(
        [
            _reply("好的"),
            _reply(json.dumps({"reply_type": "WHATEVER", "content": "好的", "action": None})),
            _reply(json.dumps({"reply_type": "WHATEVER", "content": "好的", "action": None})),
        ]
    )

    run = agent.run("你好", [], user_id=1)

    assert run.error.startswith("MODEL_OUTPUT_INVALID:")


# 阶段② ：harness 收口（补查 / 纠错 / 拒绝）


def test_forced_refill_saves_action_without_evidence(patch_chat, db_isolation):
    """
    模型不查就给按钮 → harness 判"证据不足" → 强制补查 → 复验通过。

    这是对实测缺陷的对症测试：用户追问"帮我加急委托单1"时模型不调工具、
    直接沿用历史作答。没有补查机制的话，这次回复会被无端拒绝。
    """
    fake = patch_chat(
        [
            _reply("委托单 1 可以加急，请确认。"),   # 草稿：没调工具
            _reply(  # 结构化：直接给了按钮
                json.dumps(
                    {
                        "reply_type": "ACTION_GUIDE",
                        "content": "委托单 1 是已接收状态，可以加急。",
                        "action": {"code": "APPLY_URGENT", "order_id": 1},
                    }
                )
            ),
        ]
    )

    run = agent.run("帮我加急委托单1", [], user_id=1)

    # 只调了 2 次模型（草稿 + 结构化），没有为纠错再花钱
    assert len(fake.calls) == 2
    # 强制补查发生了：trace 里有一条成功的 get_user_orders
    forced = [t for t in run.tool_trace if t.tool_name == "get_user_orders"]
    assert len(forced) == 1 and forced[0].success is True
    # 复验通过：按钮保留，没有误伤用户
    assert run.error is None
    assert run.output.reply_type.value == "ACTION_GUIDE"
    assert run.output.action.order_id == 1
    assert run.validation is not None and run.validation.passed


def test_correction_rewrite_recovers(patch_chat, db_isolation):
    """首次校验不过（编了委托单9）→ 补查也没用 → 反馈错误重写 → 通过且 corrected=True。"""
    patch_chat(
        [
            _reply("您的委托单9已发报告。"),
            _reply(json.dumps({"reply_type": "ANSWER",
                               "content": "您的委托单 9 已发报告。",
                               "action": None})),
            _reply(json.dumps({"reply_type": "ANSWER",
                               "content": "抱歉，我这边查不到委托单 9 的信息。",
                               "action": None})),
        ]
    )

    run = agent.run("我的委托单9到哪了", [], user_id=1)

    assert run.error is None
    assert run.corrected is True
    assert run.validation is not None and run.validation.passed


def test_rejection_falls_back_after_two_failures(patch_chat, db_isolation):
    """补查 + 重写都救不回来 → 固定兜底，error 带 UNSUPPORTED_FACT 码。"""
    bad = json.dumps({"reply_type": "ANSWER", "content": "您的委托单 9 已发报告。", "action": None})
    patch_chat(
        [
            _reply("委托单9已发报告"),
            _reply(bad),   # 首次结构化：编造
            _reply(bad),   # 重写：仍编造
        ]
    )

    run = agent.run("委托单9到哪了", [], user_id=1)

    assert run.error is not None and run.error.startswith("UNSUPPORTED_FACT:")
    assert run.output.reply_type.value == "REQUEST_HANDOFF"
    assert run.output.content == agent.FALLBACK_CONTENT
    assert run.validation is not None
    assert run.validation.passed is False


def test_validation_summary_exposed_on_run(patch_chat, db_isolation):
    """AgentRun.validation / corrected 字段被正确填充（api_chat 和测试都依赖它们）。"""
    patch_chat(
        [
            _reply("好的"),
            _reply(json.dumps({"reply_type": "ANSWER", "content": "好的", "action": None})),
        ]
    )

    run = agent.run("你好", [], user_id=1)

    assert run.validation is not None
    assert run.validation.passed is True
    assert run.corrected is False
    assert run.error is None


# 转人工快速通道


def test_wants_human_positive():
    """点名要人工的各种说法都要认出来（漏认的后果就是用户被 AI 拒在门外）。"""
    for text in ("转人工", "我要转人工客服", "帮我找人工", "能不能转接人工", "给我换人工", "要真人客服"):
        assert refine.wants_human(text) is True, text


def test_wants_human_negation():
    """带否定的说法不能误判 —— 用户说"先不用转人工"时弹按钮是骚扰。"""
    for text in ("先不用转人工", "不要转人工", "别给我转人工", "我不需要人工客服", "不用人工"):
        assert refine.wants_human(text) is False, text


def test_wants_human_asking_about_staff():
    """打听人工客服的作息/联系方式，是在提问，不是要转接。"""
    for text in ("人工客服几点上班", "你们人工客服电话多少", "转人工要收费吗"):
        assert refine.wants_human(text) is False, text


def test_explicit_handoff_skips_model(patch_chat):
    """
    用户说"转人工" → 不调模型，直接给 REQUEST_HANDOFF + 不含"无法转接"的应答。

    # 这条用例为什么这么重要？
    # 实测同一句"转人工"，模型这次答 REQUEST_HANDOFF、下次答 DECLINE
    # "我无法为您转接人工支持" —— 这类采样随机性没法靠调提示词根除。
    # 用例把"必须转接"钉成硬性行为：模型一次都不会被调用（剧本为空，一调就报错）。
    """
    fake = patch_chat([])   # 空剧本：只要调模型就会抛 AssertionError

    run = agent.run("转人工", [], user_id=1, nickname="陈医生")

    assert fake.calls == []                                   # 真的没调模型
    assert run.output.reply_type.value == "REQUEST_HANDOFF"
    assert "无法" not in run.output.content                    # 不能出现"我无法为您转接"
    assert "转接人工支持" in run.output.content                 # 话术与前端按钮文案一致
    # 没有业务事实 → 过 V1，观测表里记 passed 而不是 degraded_model
    assert run.validation is not None and run.validation.passed is True
    assert run.error is None


# 全量盘点快速通道（2026-11-15 新增）


def test_wants_inventory_positive():
    """数量/全量清单类问法都要命中，漏一个就会退回"沿用历史旧答案"的老路。"""
    for text in (
        "我一共有几个样本",
        "我总共有几张委托单",
        "把所有样本列出来",
        "我有多少个样本",
        "一共多少张单",
    ):
        assert refine.wants_inventory(text) is True, text


def test_wants_inventory_negative():
    """普通问法不进快速通道 —— 泛问法模型本来就会自己调工具，不必多一次 SQL。"""
    for text in ("你好", "我有哪些委托单", "委托单9到哪了", "帮我加急委托单1", "转人工"):
        assert refine.wants_inventory(text) is False, text


def test_inventory_prefetches_orders_before_model(patch_chat, db_isolation):
    """
    盘点类问题：模型第一轮就带着本轮真实数据，即使它不主动要工具也有证据。

    # 为什么用"模型直接作答、工具=0"的假剧本？
    # 这正是线上出问题的那条路径 —— 模型复述历史、一次工具都不调。
    # 只要预取发生在调模型之前，光靠注入的证据就能让 harness 通过，
    # 不需要模型配合，也不依赖 refine 的补查（补查只补证据、不改正文）。
    """
    draft = "已根据最新查询结果为您整理委托单与样本清单。"
    fake = patch_chat(
        [
            _reply(draft),
            _reply(json.dumps({"reply_type": "ANSWER", "content": draft, "action": None})),
        ]
    )

    run = agent.run("我一共有几个样本", [], user_id=1)

    assert run.error is None
    assert len(run.tool_trace) == 1
    assert run.tool_trace[0].tool_name == "get_user_orders"
    assert run.tool_trace[0].success is True
    assert len(run.tool_trace[0].result["orders"]) == 7   # 测试库是基础种子 7 张

    # 第一轮模型请求里已经带着 tool 结果 + "历史作废"提示
    first = fake.calls[0]["messages"]
    tool_msgs = [m for m in first if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and "orders" in tool_msgs[0]["content"]
    assert tool_msgs[0]["tool_call_id"] == "prefetch_orders"
    assert first[-1]["role"] == "user" and "一律作废" in first[-1]["content"]


def test_inventory_prefetch_beats_stale_history(patch_chat, db_isolation):
    """历史里已有"6 个样本"的旧答案时，预取仍把本轮最新数据挂在最后。"""
    stale = [
        {"role": "user", "content": "我一共有几个样本"},
        {"role": "assistant", "content": "您现在有6个样本在检测中：HP-2026-00902 等。"},
    ]
    draft = "已根据最新查询结果为您整理委托单与样本清单。"
    patch_chat(
        [
            _reply(draft),
            _reply(json.dumps({"reply_type": "ANSWER", "content": draft, "action": None})),
        ]
    )

    run = agent.run("我一共有几个样本", stale, user_id=1)

    assert run.error is None
    assert run.tool_trace[0].success is True
    tool_msgs = [m for m in run.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1 and "orders" in tool_msgs[0]["content"]
    # 旧答案在前、本轮证据与"历史作废"提示在后，模型最后看到的依据是真数据
    assert run.messages[-1]["role"] == "user" and "一律作废" in run.messages[-1]["content"]
