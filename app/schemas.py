"""
schemas.py —— 全项目共用的数据结构

职责：
    定义"跨文件传递"的数据形状：
      - 模型输出协议 AgentOutput（agent 产生，harness 校验，前端渲染）
      - 工具返回 ToolResult（tools 产生，agent 回填给模型）
      - 工具轨迹 ToolCallRecord（agent 记录，harness 当证据用）
      - HTTP 请求/响应体

不负责：
    校验逻辑。字段级类型/枚举约束由 Pydantic 在解析时天然完成；
    但"跨字段契约"（例如 ACTION_GUIDE 必须携带 action）属于 harness 的 V1 关卡，
    写在 harness.py —— 这样"校验"这个功能完整住在一个文件里，不散落。

    唯一的例外是 ChatRequest 的"去空白 + 拒绝空串"：它是入口处的数据清洗，
    属于"这个字段本身长什么样"，不是业务契约，放在这里才不会让入口裸奔。

为什么这几个结构要单文件：
    它们被 agent / tools / harness / api_* 同时引用，是规范里允许的两处公共依赖之一
    （另一处是 config.py）。
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# 模型输出协议


class ReplyType(str, Enum):
    """
    模型这次回复的"意图类型"。前端按它决定渲染成什么卡片，
    harness 按它决定走哪条校验分支。

    # 为什么不用自由文本 + 关键字匹配？
    # 因为"这句话算不算建议转人工"用关键词判断必然误判（模型可能说"不需要转人工"）。
    # 让模型在结构化字段里显式声明意图，前端和校验器就都有确定的依据。
    """

    ANSWER = "ANSWER"                    # 正常回答，内容有工具/知识依据
    DECLINE = "DECLINE"                  # 拒答：闲聊、与本店业务无关的问题
    REQUEST_HANDOFF = "REQUEST_HANDOFF"  # 建议转人工（含知识库 0 命中）
    ACTION_GUIDE = "ACTION_GUIDE"        # 写操作引导，必须携带 action


# 动作码白名单。模型只能从这里选，写了别的码一律被 V1/V4 拦掉。
# 三个写操作的语义（放行条件在 harness.py 的 V4 规则里逐条核对）：
#     APPLY_URGENT  申请加急：未上机（received/qc_passed/library_built）且未加急过
#     APPLY_RETEST  申请复检：质检不合格，或报告签发 7 日内
#     APPLY_CANCEL  申请退单：未建库（received/qc_passed/qc_failed）
# 用 frozenset 而不是 list：判断是 O(1)，且防止被意外修改。
ALLOWED_ACTION_CODES: frozenset[str] = frozenset(
    {"APPLY_URGENT", "APPLY_RETEST", "APPLY_CANCEL"}
)


class PageAction(BaseModel):
    """
    前端要渲染的"动作按钮"描述。

    模型只能描述"建议用户点哪个按钮"，它没有执行权 ——
    真正的改库发生在用户点击后，由后端重新校验归属和状态再执行。
    """

    code: str      # 动作码，必须在 ALLOWED_ACTION_CODES 里（APPLY_URGENT / APPLY_RETEST / APPLY_CANCEL）
    order_id: int  # 目标委托单主键（orders.id，不是"样本编号"——样本号在 order_items 里）


class AgentOutput(BaseModel):
    """
    模型最终输出的契约。这是 harness 校验的输入，也是接口返回给前端的 reply。

    reply_type / content / action 三者的关系（跨字段规则在 harness V1 校验）：
        ACTION_GUIDE    → action 必须有，content 是引导语
        其余三种        → action 必须为空
        content         → 任何类型都不能是空字符串
    """

    reply_type: ReplyType
    content: str
    action: PageAction | None = None


# 工具子系统


class ToolResult(BaseModel):
    """
    工具的统一返回结构。

    # 工具失败为什么不抛异常？
    # 因为异常会把整个请求打断，模型永远看不到"你查的东西不存在"这件事，
    # 于是它只能编一个答案（这正是幻觉的来源）。
    # 把失败变成结构化的 success=False + code + message 交回给模型，
    # 模型才有机会说"抱歉，没查到您的订单"，harness 也能据此判断证据不足。
    """

    success: bool
    code: str                      # 稳定错误码，如 ORDERS_EMPTY / KNOWLEDGE_NOT_FOUND
    message: str                   # 给模型看的失败说明（中文，可直接进 prompt）
    data: Any = None               # 成功时的业务数据

    @classmethod
    def ok(cls, data: Any = None, code: str = "OK", message: str = "") -> "ToolResult":
        """构造一个成功结果。"""
        return cls(success=True, code=code, message=message, data=data)

    @classmethod
    def fail(cls, code: str, message: str, data: Any = None) -> "ToolResult":
        """构造一个业务失败结果（不是崩溃，是"查了但没有"）。"""
        return cls(success=False, code=code, message=message, data=data)


@dataclass
class ToolCallRecord:
    """
    一次工具调用的轨迹记录。

    只在单次请求的内存里存活（不落库），因为 harness 的校验语义是
    "本次运行中成功的工具结果才算证据" —— 跨请求的旧证据不该被引用。
    """

    tool_name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    success: bool = False


# HTTP 请求 / 响应体


class LoginRequest(BaseModel):
    """登录请求体。"""

    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=100)


class UserInfo(BaseModel):
    """返回给前端的用户信息（绝不包含 password_hash）。"""

    id: int
    username: str
    nickname: str | None = None
    role: str


class LoginResponse(BaseModel):
    """登录成功后的响应：token + 用户信息。"""

    access_token: str
    token_type: str = "bearer"
    user: UserInfo


class ConversationInfo(BaseModel):
    """一个会话的基本信息。status 取值见会话状态机：AI / QUEUED / HUMAN / CLOSED。"""

    id: int
    status: str
    created_at: datetime
    last_active_at: datetime


class MessageItem(BaseModel):
    """一条聊天消息。role 取值：user / ai / human。"""

    id: int
    role: str
    content: str
    created_at: datetime


class ChatRequest(BaseModel):
    """发消息的请求体。"""

    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def _strip_and_reject_blank(cls, value: str) -> str:
        """
        去掉首尾空白，并拒绝"只有空白"的内容。

        # min_length=1 拦不住 "   "。
        # 三个空格能通过长度校验，然后一路走到 agent → 白调 2~3 次大模型，
        # 花掉几秒和一个请求配额，最后模型回一句"您好，请问需要什么帮助"。
        # 在入口把空白串挡掉，是最便宜的一处优化。
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("消息内容不能为空")
        return stripped


class UserMessageAck(BaseModel):
    """
    人工接待模式下用户发消息的响应。

    # 为什么不复用 ChatResponse？
    # ChatResponse 的 reply 是必填的 AgentOutput —— 那是"AI 回答了"的形状。
    # 人工模式下没有 AI 回复，硬塞一个空的 AgentOutput 会让前端
    # 误以为有气泡要渲染。类型上区分开，前端 if 的时候才不会猜。
    """

    conversation_id: int
    user_message_id: int
    status: str  # 当前会话状态（QUEUED / HUMAN），前端据此渲染"转接中/人工服务中"


class ChatResponse(BaseModel):
    """
    发消息的完整响应。

    reply 是模型输出的结构化结果（含 reply_type 和 action），
    前端据此决定渲染普通气泡、转人工按钮还是"确认加急/复检/退单"按钮。

    validation 是 harness 五关卡的结论摘要（阶段② 起填充）：
        status     passed / corrected / rejected / degraded_tools / degraded_model
        passed     本次回复是否通过了校验（降级也算 True，因为内容是系统控制的）
        corrected  首次未过、模型重写后通过 —— 前端可以打"已纠错"标记
        error_codes  触发的错误码列表（MODEL_OUTPUT_INVALID 等）
    """

    conversation_id: int
    user_message_id: int
    ai_message_id: int
    reply: AgentOutput
    validation: dict[str, Any] | None = None


# 阶段④：写操作确认 / 转人工 / 坐席端


class OrderActionRequest(BaseModel):
    """
    写操作确认接口（加急/复检/退单共用）的请求体。

    conversation_id 是可选的：前端带上它，操作成功后后端会把确认消息
    写进这个会话（用户在聊天页能立刻看到"已登记加急"）；
    不带也能执行，只是聊天记录里不落确认消息。
    """

    conversation_id: int | None = None


class OrderActionResponse(BaseModel):
    """一次写操作（加急/复检/退单）的结果：成功/失败都返回 200 语义由前端提示。"""

    order_id: int
    success: bool
    message: str  # 给用户看的中文说明（"已登记加急"/"该委托单当前节点不支持复检"）


class HandoffRequest(BaseModel):
    """请求转人工的请求体（当前无字段，留作扩展）。"""

    reason: str | None = None  # 仅记日志用，不落库


class HandoffResponse(BaseModel):
    """转人工请求的结果：status 是会话的最新状态，前端据此渲染。"""

    conversation_id: int
    status: str  # QUEUED / HUMAN（重复请求时直接返回当前值）


class QueuedConversation(BaseModel):
    """坐席工作台"待接管列表"里的一行。"""

    conversation_id: int
    user_nickname: str
    last_message: str | None  # 最近一条消息的截断预览，帮助坐席决定先接谁
    queued_since: datetime  # 排队时间（created/updated at）


class ActiveConversation(BaseModel):
    """
    坐席工作台"接待中列表"里的一行（status='HUMAN' 的会话）。

    # 为什么不能只查 QUEUED？
    # 会话一旦被接管就离开排队队列，坐席点开别的会话或刷新页面后，
    # 就再也没有入口回到刚才那个会话 —— 用户那边横幅写着"人工客服正在为您服务"，
    # 实际永远等不到回复。所以必须有第二个列表把 HUMAN 的会话留在视野里。
    """

    conversation_id: int
    user_nickname: str
    last_message: str | None      # 最近一条消息预览，用来判断用户是否还在等
    last_active_at: datetime      # 最近活跃时间，超过一定时长就该收尾或转回 AI
    claimed_by_me: bool           # 是否由当前坐席接管（多坐席时用于区分归属）


class AgentMessageRequest(BaseModel):
    """坐席回复的请求体，与用户的 ChatRequest 同一套清洗规则。"""

    content: str = Field(min_length=1, max_length=2000)

    @field_validator("content")
    @classmethod
    def _strip_and_reject_blank(cls, value: str) -> str:
        """与 ChatRequest 相同：拦住纯空白串，见那边的说明。"""
        stripped = value.strip()
        if not stripped:
            raise ValueError("消息内容不能为空")
        return stripped


class CloseConversationRequest(BaseModel):
    """坐席结束服务的请求体：back=ai（回到 AI 接待）或 closed（终态）。"""

    back: str = Field(pattern="^(ai|closed)$")  # 只允许这两个值
