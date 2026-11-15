# 肿瘤 NGS 送检全流程智能助手（Agent + Harness）

面向肿瘤 NGS 送检医生与科研客户的智能服务助手：多轮对话 + 工具调用 Agent +
harness 五关卡事实校验 + 混合检索 RAG（稠密 + 稀疏 + RRF）+ 写操作四段式闭环
（申请加急 / 申请复检 / 退单）+ 转人工（技术支持/生信工程师）+ agent_runs 观测落库
+ WebSocket 实时推送 + 前端双视图 + 全量盘点确定性查库通道（防止沿用历史旧清单）。

技术栈：FastAPI + PostgreSQL 16 / pgvector + 智谱 GLM（OpenAI 兼容协议）
+ 本地 bge-m3 双头编码（稠密 + 稀疏，GPU）+ 原生 HTML/JS 前端（无构建步骤）。

> **免责声明**：本项目仅提供信息查询与流程支持，所有账号、委托单、样本与报告
> 变异均为虚构演示数据，不构成医学建议，也不用于任何诊断或治疗决策。

---

## 登录账号（种子数据）

| 用户名   | 密码    | 昵称        | 委托单                                        |
|----------|---------|-------------|----------------------------------------------|
| chen     | 123456  | 陈医生      | 7 张，覆盖 received / report_issued / cancelled / analyzing / report_review / qc_passed |
| lin      | 123456  | 林教授      | 2 张（sequencing + 阴性报告单），演示数据隔离  |
| wang     | 123456  | 王医生      | 1 张 qc_failed（DNA 浓度不足），演示免费复检   |
| zhao     | 123456  | 赵同学      | 0 张，演示空结果降级                           |
| zhou     | 123456  | 销售小周    | 1 张 library_built                             |
| tech001  | 123456  | 技术支持老徐 | 技术支持坐席（转人工接管）                     |

**退单演示扩展数据**（两个脚本，均只灌开发库、可重复执行，演示后重跑即复位）：

- `script/seed_extra_cancel_orders.sql` —— 3 张单
  （101 已接收 / 102 质检不合格 / 103 建库完成），覆盖退单白名单与「已建库不可退」边界。
- `script/seed_extra_received_orders.sql` —— 6 张全
  `received` 单（104~109），各配 1 款在售产品，供批量退单演示。

两份都灌完后陈医生共 16 张单、9 张可退单候选；`get_user_orders` 的查询上限已
从 10 放宽到 30，保证全部单据都进得了工具证据（改代码后需重启后端）。

---

## 快速开始

所有命令都在**项目根目录**执行。Python 3.12，主要依赖 fastapi、uvicorn、psycopg、
pydantic-settings、FlagEmbedding、torch（torch 请按本机 CUDA 版本自行安装）。

```powershell
# 1. 配置：复制模板并填写 DATABASE_URL / LLM_API_KEY / JWT_SECRET
cp .env.example .env

# 2. 建库（PostgreSQL 需已装 pgvector 扩展）
#    用任意 PG 客户端在开发库依次执行：
#      script/ngs_schema.sql   12 张表 + vector 扩展
#      script/seed_ngs.sql     6 个账号 / 11 张委托单 / 报告变异等种子
python tests/setup_test_db.py     # 重建独立测试库（跑 pytest 前执行一次）

# 3. 语料入库：扫描 document/docs/*.md 切块并做双路编码（首次加载 bge-m3 到 GPU）
python -m app.ingest

# 4. 启动（两个窗口）
python run.py                                  # 后端 API + WebSocket，端口 8010
python -m http.server 8080 --directory web     # 前端静态页，端口 8080
```

浏览器打开 <http://127.0.0.1:8080/login.html>。

`.env` 已被 `.gitignore` 排除；`tests/setup_test_db.py` 与 `tests/smoke.py` 都从 `.env`
派生数据库连接串，代码里不含明文口令。测试跑在独立的 `ngs_service_assistant_test` 库，
不会动开发库。表结构 SQL、种子数据与 8 份语料都随代码一起版本化，clone 后即可复现。

## 启动前端

先启动后端（PyCharm 运行 `run.py` 亦可），再开一个命令行窗口起静态服务器：

```powershell
python -m http.server 8080 --directory web
```

改完代码想快速验一遍整体链路，后端启动后另开命令行跑端到端冒烟（约 1 分钟，真调模型）：

```powershell
python tests/smoke.py
```

31 项检查覆盖健康检查、鉴权、真实 LLM 问答、写操作三重条件、转人工状态机、库内一致性；
脚本跑完会复位聊天记录、会话状态与扩展单据，不污染演示数据。首次知识库检索要冷加载
bge-m3 到 GPU，可能耗时 1~2 分钟（热后约 6 秒），脚本已单独放宽该用例超时。

页面说明（两个文件，无 node、无构建）：

- `login.html` —— 登录页。送检客户用 chen / lin / wang，技术支持用 tech001。
- `chat.html` —— 一个页面两副面孔，按登录角色自动切换：
  - **客户视图**：聊天气泡；AI 建议加急/复检/退单时出现确认按钮（点击后条件 UPDATE 真实落库）；
    AI 建议转人工或点【转接人工支持】后，状态横幅提示"转接中/人工服务中"，
    人工回复经 WebSocket 实时到达（WS 断线自动降级为 3 秒轮询）。
  - **坐席视图**：左侧「接待中 / 待接管」两栏（含用户昵称和最近消息预览），
    点【接管】后右侧展开完整聊天记录，可回复、可"结束并转回 AI / 关闭会话"。

## 核心机制速览

- **Agentic loop**：模型自主决定调 4 个只读工具（查委托单/查产品/查报告变异/检索 SOP），
  多轮循环、轮数耗尽强制收口；工具失败不抛异常，统一降级。
- **Harness 五关卡**：V1 结构 → V2 工具失败降级 → V3 事实比对（委托单号/样本号/金额/
  节点/周期/**基因白名单与别名、变异位点、Tier 分级**）→ V4 写操作放行 →
  V5 补查/重写/兜底（refine.py）。证据不足先强制补查一次，不无端拒绝用户。
- **写操作四段式**：AI 只出建议 → harness 校验 → 前端确认按钮 → 条件原子 UPDATE
  （加急未上机 / 复检 qc_failed 或发报告 7 日内 / 退单未建库）。
- **混合检索**：bge-m3 双头一次前向出稠密向量 + 稀疏词项权重，pgvector 余弦
  + 倒排表点积双路召回，RRF(k=60) 融合；稀疏路失败降级纯稠密。

## 目录说明

```
app/            后端全部代码，一个功能一个文件，平铺不建子目录
  main.py       启动装配、挂路由、健康检查
  config.py     读 .env
  db.py         连接池（不含任何业务 SQL）
  security.py   密码哈希、JWT、当前用户依赖
  llm.py        调智谱、重试、模型降级
  prompts.py    提示词与兜底话术（改话术只动这一个文件）
  embedding.py  bge-m3 懒加载单例 + 稠密/稀疏/双头编码入口
  rag.py        混合检索（<=> 余弦 + unnest 倒排点积 + RRF 融合）
  ingest.py     命令行回填双路索引（embedding + 稀疏词项，可重复跑）
  tools.py      4 个只读工具的 SQL + schema + 分发
  agent.py      一次问答的编排（工具循环）
  refine.py     收口：结构化 → 五关卡校验 → 补查 → 重写 → 兜底
  json_repair.py 模型输出 JSON 的容错解析（代码块/废话/裸引号）
  harness.py    五关卡校验（V1~V4 判断 + 证据集合，纯函数，含基因白名单）
  run_log.py    agent_runs 观测表写入 + 校验摘要
  schemas.py    AgentOutput、ToolResult、接口 DTO、动作码白名单
  api_auth.py   登录
  api_chat.py   会话、发消息（人工模式分流）
  api_actions.py 加急/复检/退单确认（条件原子 UPDATE）
  api_handoff.py 转人工 + 坐席端接口（队列/接管/回复/结束）
  ws.py         WebSocket 连接管理与实时推送
  log_setup.py  日志配置（唯一初始化日志的地方）
web/            前端（两个页面，纯原生 HTML/JS）
  login.html    登录页
  chat.html     客户聊天 + 坐席工作台（按角色切换）
run.py          启动入口
tests/          pytest 用例（233 个，跑独立测试库 ngs_service_assistant_test）
  smoke.py      端到端冒烟（31 项，需后端已启动）
script/         表结构与种子数据，随代码一起版本化
  ngs_schema.sql          12 张表 + pgvector 扩展
  seed_ngs.sql            基础种子（6 账号 / 11 委托单 / 报告变异）
  seed_extra_*.sql        演示用的扩展委托单
document/docs/  知识库语料（8 份 md：产品说明 / SOP / 报告解读指南 / FAQ）
.env.example    配置模板（复制为 .env 后填写，.env 不入库）
logs/app.log    运行日志（5MB 轮转，留 3 份，不入库）
```
