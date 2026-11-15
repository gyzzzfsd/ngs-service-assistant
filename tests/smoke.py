"""
smoke.py —— 端到端冒烟测试（需后端已启动）

用法：
    D:/app/anaconda/envs/llm1/python.exe tests/smoke.py

覆盖六组：
    A 基础设施   /health、静态页、WebSocket、数据库连通
    B 认证       登录 / 错误密码 / me / 无 token
    C 真实问答   全量盘点、跨单据查询、产品、知识库、写操作引导（真调 LLM）
    D 写操作     退单 / 重复退单 / 不可退 / 加急 / 复检 / 越权
    E 转人工     QUEUED → HUMAN → 关闭 的完整状态机
    F 数据一致性 表齐全、金额对账、孤儿数据、序列

为什么单独一个脚本而不写进 pytest：
    这些用例打的是"正在运行的服务"、真调大模型、真的改开发库，
    每次跑 pytest 都执行会又慢又花钱，还会把演示数据改乱。
    冒烟是发版/改代码后手动跑一次的动作，独立脚本更合适。

# 脚本会往开发库写消息和改单据状态，必须在 finally 里复位。
# 记录冒烟开始前的 messages/agent_runs 最大 id，结束时只删这之后、
# 且属于本次会话的记录；单据状态用两份幂等 seed 脚本重建。
# 这样跑完冒烟，聊天历史与业务数据跟没跑过一样。
#
# 已知边界：若某条请求客户端读超时、服务端仍在线程池里跑完并落库，
# 写入会晚于 finally 的复位（冷启动首次实测遇到过，迟到约 30 秒）。
# 遇到这种情况按 id 手工删一次即可；正常路径不会产生残留。
"""

import sys
from pathlib import Path

import psycopg
import requests
from websockets.sync.client import connect

API = "http://127.0.0.1:8010"
WEB = "http://127.0.0.1:8080"
DB = "host=192.168.218.100 port=5432 user=ai_cs password=ai_cs_pwd_2026 dbname=ngs_service_assistant"
SEED_DIR = Path(__file__).resolve().parent.parent.parent / "ngs_service_assistant_data" / "script"

RESULTS: list[tuple[str, bool, str]] = []
STATE: dict = {}

# 开发机装了本地代理（HTTP_PROXY 指向 127.0.0.1:23889），
# requests 默认会把发往 127.0.0.1:8010 的请求也绕去代理，既多一跳、
# 又会在代理抖动时误报 timeout。trust_env=False 让冒烟直连本机服务。
SESSION = requests.Session()
SESSION.trust_env = False


def check(name: str, ok: bool, detail: str = "") -> None:
    """记一条结果并即时打印，失败时把实际值带上，方便定位。"""
    RESULTS.append((name, bool(ok), detail))
    no = len(RESULTS)
    print(f"[{'PASS' if ok else 'FAIL'}] {no:02d} {name}" + (f" | {detail}" if detail else ""), flush=True)


def http(method: str, path: str, token: str | None = None, body: dict | None = None,
         base: str = API, timeout: int = 90) -> tuple[int, object]:
    """发一个 HTTP 请求，返回 (状态码, JSON 或截断文本)。4xx 不当异常抛。"""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    resp = SESSION.request(method, base + path, json=body, headers=headers, timeout=timeout)
    try:
        return resp.status_code, resp.json()
    except ValueError:
        return resp.status_code, resp.text[:200]


def login(username: str, password: str = "123456") -> tuple[int, str | None, object]:
    """登录，返回 (状态码, token, 原始响应)。"""
    status, data = http("POST", "/api/auth/login", body={"username": username, "password": password})
    token = data.get("access_token") if isinstance(data, dict) else None
    return status, token, data


def chat(text: str, timeout: int = 90) -> tuple[int, dict]:
    """往当前会话发一句话，拿到完整 ChatResponse（真调 LLM，约 3~8 秒）。"""
    status, data = http(
        "POST", f"/api/conversations/{STATE['conv_id']}/messages",
        token=STATE["chen_token"], body={"content": text}, timeout=timeout,
    )
    return status, data if isinstance(data, dict) else {}


def group_a() -> None:
    """A 组：服务活着、静态页能开、WS 能连、库能查。"""
    status, data = http("GET", "/health")
    check("A1 /health 200 且 status=ok", status == 200 and data.get("status") == "ok", f"status={status} body={data}")

    for page in ("login.html", "chat.html"):
        st, _ = http("GET", f"/{page}", base=WEB, timeout=10)
        check(f"A2 静态页 {page}", st == 200, f"status={st}")

    # WS 无效 token 也会被 accept 后以 1008 关闭，必须拿真 token 才能验证"真连上"
    _, ws_token, _ = login("chen")
    try:
        with connect(f"ws://127.0.0.1:8010/ws?token={ws_token}", open_timeout=3) as ws:
            ws.ping()
        check("A3 WebSocket 可连接", True)
    except Exception as exc:
        check("A3 WebSocket 可连接", False, f"{type(exc).__name__}: {exc}")

    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("select count(*) from information_schema.tables where table_schema='public'")
        n = cur.fetchone()[0]
    check("A4 数据库连通", n >= 12, f"public 表数={n}")


def group_b() -> None:
    """B 组：登录、错误口令、身份读取、鉴权拦截。"""
    status, token, data = login("chen")
    STATE["chen_token"] = token
    check("B1 chen 登录", status == 200 and bool(token) and data["user"]["role"] == "customer",
          f"status={status} role={data.get('user', {}).get('role')}")

    st, _, _ = login("chen", "wrong_pwd")
    check("B2 错误密码被拒", st in (401, 400, 403), f"status={st}")

    st, me = http("GET", "/api/auth/me", token=token)
    check("B3 /me 身份正确", st == 200 and me.get("id") == 1, f"status={st} id={me.get('id')}")

    st, _ = http("GET", "/api/auth/me")
    check("B4 无 token 被拒", st in (401, 403), f"status={st}")

    _, tech_token, _ = login("tech001")
    STATE["tech_token"] = tech_token
    _, lin_token, _ = login("lin")
    STATE["lin_token"] = lin_token
    check("B5 坐席/隔离账号可登录", bool(tech_token) and bool(lin_token))


def group_c() -> None:
    """C 组：真实 LLM 链路，重点验证 LIMIT 30 与 harness 没被 16 张单压垮。"""
    st, conv = http("POST", "/api/conversations", token=STATE["chen_token"])
    STATE["conv_id"] = conv.get("id") if isinstance(conv, dict) else None
    check("C1 会话创建/复用", st == 200 and bool(STATE["conv_id"]), f"status={st} conv={conv}")

    st, msgs = http("GET", f"/api/conversations/{STATE['conv_id']}/messages", token=STATE["chen_token"])
    check("C2 历史消息可读", st == 200 and isinstance(msgs, list), f"status={st} 条数={len(msgs) if isinstance(msgs, list) else '?'}")

    st, r = chat("我一共有几个样本")
    content = r.get("reply", {}).get("content", "")
    passed = (r.get("validation") or {}).get("passed")
    hits = [s for s in ("01014", "01019", "00902", "00850") if s in content]
    check("C3 全量盘点覆盖新旧单据", st == 200 and r.get("reply", {}).get("reply_type") == "ANSWER"
          and passed is True and len(hits) >= 3,
          f"type={r.get('reply', {}).get('reply_type')} passed={passed} 命中={hits}")

    st, r = chat("委托单2 现在是什么状态")
    content = r.get("reply", {}).get("content", "")
    check("C4 老单据仍在证据内（LIMIT 30）", st == 200 and "已发报告" in content,
          f"片段={content[:80]}")

    st, r = chat("肺癌11基因检测多少钱")
    content = r.get("reply", {}).get("content", "")
    check("C5 产品金额问答", st == 200 and "3600" in content, f"片段={content[:80]}")

    # 首次知识库检索要冷加载 bge-m3（2.27GB 到 GPU），实测可到 90 秒以上；
    # 热了以后只要 6 秒。这里单独放宽，避免把一次性冷启动误报成故障。
    st, r = chat("FFPE 组织样本采集有什么要求", timeout=180)
    content = r.get("reply", {}).get("content", "")
    check("C6 知识库 RAG 问答", st == 200 and len(content) > 20
          and r.get("reply", {}).get("reply_type") not in ("REQUEST_HANDOFF",),
          f"片段={content[:80]}")

    st, r = chat("帮我退掉委托单104")
    reply = r.get("reply", {})
    code = (reply.get("action") or {}).get("code")
    check("C7 退单引导四段式", st == 200 and reply.get("reply_type") == "ACTION_GUIDE"
          and code == "APPLY_CANCEL",
          f"type={reply.get('reply_type')} code={code}")


def group_d() -> None:
    """D 组：写操作三重条件与越权防护（直接打接口，不经过模型）。"""
    tok = STATE["chen_token"]

    st, r = http("POST", "/api/orders/104/cancel", token=tok, body={})
    check("D1 已接收单可退单", st == 200 and r.get("success") is True, f"status={st} body={r}")

    st, r = http("POST", "/api/orders/104/cancel", token=tok, body={})
    check("D2 重复退单被拒", st == 200 and r.get("success") is False, f"status={st} body={r}")

    st, r = http("POST", "/api/orders/103/cancel", token=tok, body={})
    check("D3 已建库单不可退", st == 200 and r.get("success") is False, f"status={st} body={r}")

    st, r = http("POST", "/api/orders/105/urgent", token=tok, body={})
    check("D4 未上机单可加急", st == 200 and r.get("success") is True, f"status={st} body={r}")

    st, r = http("POST", "/api/orders/102/retest", token=tok, body={})
    check("D5 质检不合格可复检", st == 200 and r.get("success") is True, f"status={st} body={r}")

    st, r = http("POST", "/api/orders/104/cancel", token=STATE["lin_token"], body={})
    ok = st in (404, 403) or (isinstance(r, dict) and r.get("success") is False)
    check("D6 越权访问他人单据被拒", ok, f"status={st} body={r}")


def group_e() -> None:
    """E 组：转人工状态机 QUEUED → HUMAN → 关闭。"""
    tok, tech = STATE["chen_token"], STATE["tech_token"]
    conv = STATE["conv_id"]

    st, rows = http("GET", "/api/agent/queued", token=tech)
    check("E1 坐席待接管列表", st == 200 and isinstance(rows, list), f"status={st} 条数={len(rows) if isinstance(rows, list) else '?'}")

    st, r = http("POST", f"/api/conversations/{conv}/handoff", token=tok, body={})
    check("E2 转人工 → QUEUED", st == 200 and r.get("status") == "QUEUED", f"status={st} body={r}")

    st, r = http("POST", f"/api/agent/conversations/{conv}/accept", token=tech, body={})
    check("E3 坐席接管 → HUMAN", st == 200 and r.get("status") == "HUMAN", f"status={st} body={r}")

    st, rows = http("GET", "/api/agent/active", token=tech)
    check("E4 接待中列表可见", st == 200 and isinstance(rows, list), f"status={st} 条数={len(rows) if isinstance(rows, list) else '?'}")

    st, r = http("POST", f"/api/agent/conversations/{conv}/close", token=tech, body={"back": "ai"})
    check("E5 关闭会话回到 AI", st == 200 and r.get("status") == "AI", f"status={st} body={r}")


def group_f() -> None:
    """F 组：库内一致性与序列，冒烟改过数据也应对得上。"""
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("""select count(*) from orders o join order_items i on i.order_id=o.id
                       group by o.id, o.total_amount
                       having o.total_amount <> sum(i.unit_price*i.quantity)""")
        mismatch = cur.fetchall()
        check("F1 全库金额对账", len(mismatch) == 0, f"不平单号={mismatch}")

        cur.execute("select count(*) from order_items i left join orders o on o.id=i.order_id where o.id is null")
        orphan_items = cur.fetchone()[0]
        cur.execute("select count(*) from report_variants v left join reports r on r.id=v.report_id where r.id is null")
        orphan_variants = cur.fetchone()[0]
        check("F2 无孤儿子表数据", orphan_items == 0 and orphan_variants == 0,
              f"items={orphan_items} variants={orphan_variants}")

        cur.execute("select (select last_value from orders_id_seq) >= (select max(id) from orders), "
                    "(select last_value from order_items_id_seq) >= (select max(id) from order_items)")
        seq_ok = cur.fetchone()
        check("F3 序列未落后于显式 ID", all(seq_ok), f"orders/items={seq_ok}")


def snapshot() -> None:
    """冒烟前记下基线，供 cleanup 精确回滚。"""
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        cur.execute("select coalesce(max(id),0) from messages"); STATE["max_msg"] = cur.fetchone()[0]
        cur.execute("select coalesce(max(id),0) from agent_runs"); STATE["max_run"] = cur.fetchone()[0]


def cleanup() -> None:
    """复位：删本次新增的消息/观测，恢复会话状态，重建两份扩展单。"""
    conv = STATE.get("conv_id")
    with psycopg.connect(DB) as conn, conn.cursor() as cur:
        if conv:
            cur.execute("delete from messages where conversation_id=%s and id>%s", (conv, STATE["max_msg"]))
            cur.execute("delete from agent_runs where conversation_id=%s and id>%s", (conv, STATE["max_run"]))
            cur.execute("update conversations set status='AI', claimed_by=null where id=%s", (conv,))
    with psycopg.connect(DB, autocommit=True) as conn, conn.cursor() as cur:
        for name in ("seed_extra_cancel_orders.sql", "seed_extra_received_orders.sql"):
            path = SEED_DIR / name
            if not path.exists():
                print(f"[WARN] 复位脚本不存在：{path}", flush=True)
                continue
            cur.execute(path.read_text(encoding="utf-8"))
            while cur.nextset():
                pass
    print("[CLEAN] 聊天记录、会话状态、扩展单据已复位", flush=True)


def run_group(name: str, func) -> None:
    """跑一组；整组异常也只记一条 FAIL，不中断后续组。"""
    print(f"\n--- {name} ---", flush=True)
    try:
        func()
    except Exception as exc:
        check(f"{name} 组执行", False, f"{type(exc).__name__}: {exc}")


def main() -> int:
    """按 A→F 顺序跑，finally 保证复位，最后输出汇总与退出码。"""
    snapshot()
    try:
        run_group("A 基础设施", group_a)
        run_group("B 认证", group_b)
        run_group("C 真实问答", group_c)
        run_group("D 写操作", group_d)
        run_group("E 转人工", group_e)
        run_group("F 数据一致性", group_f)
    finally:
        try:
            cleanup()
        except Exception as exc:
            check("复位清理", False, f"{type(exc).__name__}: {exc}")

    failed = [r for r in RESULTS if not r[1]]
    print(f"\n===== 汇总：{len(RESULTS) - len(failed)}/{len(RESULTS)} 通过 =====")
    for name, _, detail in failed:
        print(f"  FAIL  {name} | {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
