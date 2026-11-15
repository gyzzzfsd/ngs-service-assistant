"""
AI 智能客服后端包。

目录约定（一个功能一个文件，不按 controller/service/repository 分层）：
    main.py       启动装配
    config.py     读 .env
    db.py         连接池
    security.py   密码哈希 + JWT + 当前用户依赖
    llm.py        调大模型（含工具调用、重试、备用模型）
    embedding.py  bge-m3 文本转向量
    rag.py        切块 / 入库 / 检索
    tools.py      3 个只读工具 + 1 个写操作建议工具
    agent.py      一次问答的编排
    harness.py    五关卡校验
    schemas.py    共用数据结构
    api_auth.py   登录接口
    api_chat.py   聊天 / WebSocket / 转人工
    ingest.py     命令行回填向量
"""
