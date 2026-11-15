"""
test_json_repair.py —— 畸形 JSON 的修复测试

这些用例全部来自真实踩坑（电商时代抓到的线上故障，换皮不换病灶）：
模型把 content 写成 "委托单1目前是"已接收"状态..."，正文里的引号没转义，
整段 JSON 语法非法，解析器直接判失败，用户看到的是"转人工"兜底。
这是高频现象（回答里带引号就会复现），必须专门测。
现在的 STRUCTURE_PROMPT 已要求正文引号改用「」，实测不再触发修复 ——
但防线保留：模型采样永远不保证 100% 听话。
"""

import json

import pytest

from app.json_repair import escape_inner_quotes, extract_json


# 真实失败样本


REAL_BROKEN = """{
  "reply_type": "ACTION_GUIDE",
  "content": "委托单1目前是"已接收"状态，可以帮您加急。请点击"确认加急"按钮提交申请。",
  "action": {
    "code": "APPLY_URGENT",
    "order_id": 1
  }
}"""


def test_real_broken_sample_now_parses():
    """
    线上真实抓到的坏样本（换检测域措辞，病灶相同）：必须能解析出来，且字段一个不差。

    # 这个用例的价值不在"测解析器"，而在于把一次线上故障固化成回归测试。
    # 以后任何人改 extract_json，只要把这条弄坏了，测试立刻红。
    """
    data = extract_json(REAL_BROKEN)

    assert data is not None
    assert data["reply_type"] == "ACTION_GUIDE"
    assert data["action"]["code"] == "APPLY_URGENT"
    assert data["action"]["order_id"] == 1
    # 正文里的引号要原样保留（不能变成 \" 或消失）
    assert "已接收" in data["content"]
    assert "确认加急" in data["content"]


def test_repair_keeps_structural_quotes_intact():
    """修复不能把结构性的引号也转义掉 —— 那样就从"缺转义"变成"多了转义"。"""
    raw = '{"a": "x", "b": "y"}'

    assert json.loads(escape_inner_quotes(raw)) == {"a": "x", "b": "y"}


def test_repair_leaves_valid_json_untouched():
    """本来合法的 JSON，修复函数应当原样返回。"""
    raw = '{"content": "正常内容，没有引号", "n": 1}'

    assert escape_inner_quotes(raw) == raw


def test_already_escaped_quotes_not_double_escaped():
    """模型正确写了 \\" 的时候，不能再补一层反斜杠。"""
    raw = '{"content": "他说\\"你好\\"就走了"}'

    data = json.loads(escape_inner_quotes(raw))
    assert data["content"] == '他说"你好"就走了'


def test_broken_json_with_fence_and_preamble():
    """三重难度叠加：代码块包裹 + 前面有废话 + 正文裸引号。"""
    raw = '好的，结果如下：\n```json\n{"reply_type": "ANSWER", "content": "这是"重点"内容", "action": null}\n```'

    data = extract_json(raw)

    assert data is not None
    assert data["content"] == '这是"重点"内容'


# 不能误伤


@pytest.mark.parametrize(
    "raw",
    ["完全不是 JSON", "", "[1, 2, 3]", "{不是键值对}"],
    ids=["纯文本", "空串", "数组", "坏结构"],
)
def test_still_returns_none_for_garbage(raw):
    """修复能力不能被无限放大 —— 修不出来就得老老实实返回 None，走兜底。"""
    assert extract_json(raw) is None


def test_nested_object_values_still_parse():
    """嵌套对象 + 中文内容，修复逻辑不能把嵌套的引号搞坏。"""
    raw = '{"action": {"code": "APPLY_URGENT", "order_id": 2}, "content": "委托单2"已发报告"了"}'

    data = extract_json(raw)

    assert data["action"]["order_id"] == 2
    assert data["content"] == '委托单2"已发报告"了'
