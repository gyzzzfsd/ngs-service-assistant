# 肿瘤 NGS 送检全流程智能助手（Agent + Harness）

面向肿瘤 NGS 送检医生与科研客户的智能服务助手：多轮对话 + 工具调用 Agent +
harness 五关卡事实校验 + 混合检索 RAG（稠密 + 稀疏 + RRF）+ 写操作四段式闭环
（申请加急 / 申请复检 / 退单）+ 转人工（技术支持/生信工程师）+ agent_runs 观测落库
+ WebSocket 实时推送 + 前端双视图 + 全量盘点确定性查库通道（防止沿用历史旧清单）。

数据库 hadoop100（PostgreSQL 16 + pgvector，`ngs_service_assistant`），大模型用智谱 GLM，
embedding 用本地 bge-m3 双头（稠密 + 稀疏，GPU）。

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

- `../ngs_service_assistant_data/script/seed_extra_cancel_orders.sql` —— 3 张单
  （101 已接收 / 102 质检不合格 / 103 建库完成），覆盖退单白名单与「已建库不可退」边界。
- `../ngs_service_assistant_data/script/seed_extra_received_orders.sql` —— 6 张全
  `received` 单（104~109），各配 1 款在售产品，供批量退单演示。

两份都灌完后陈医生共 16 张单、9 张可退单候选；`get_user_orders` 的查询上限已
从 10 放宽到 30，保证全部单据都进得了工具证据（改代码后需重启后端）。

---

## 启动前端

先启动后端（PyCharm 运行 run.py），再开一个命令行窗口起静态服务器：

```powershell
cd D:\allCode\AI_Model\project\Project_2_NGS_Assistant\ngs_service_assistant
D:\app\anaconda\envs\llm1\python.exe -m http.server 8080 --directory web
```

浏览器打开 <http://127.0.0.1:8080/login.html>。

改完代码想快速验一遍整体链路，后端启动后另开命令行跑端到端冒烟（约 1 分钟，真调模型）：

```powershell
D:\app\anaconda\envs\llm1\python.exe tests/smoke.py
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
run.py          PyCharm 启动入口
tests/          pytest 用例（221 个，跑独立测试库 ngs_service_assistant_test）
logs/app.log    运行日志（5MB 轮转，留 3 份）
script/         设计文档、表结构 SQL 与种子数据（伴生数据目录）
document/       知识库语料原文与切块脚本（伴生数据目录）
```
