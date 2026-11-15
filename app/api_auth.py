"""
api_auth.py —— 登录与当前用户

职责：
    只提供两个接口：登录（换 token）、查当前登录用户（给前端刷新页面时用）。

不负责：
    密码怎么比对、token 怎么签发（security.py）、用户数据怎么维护（本项目不做注册，用户由种子数据预置）。
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from psycopg.rows import dict_row

from app.db import get_conn
from app.schemas import LoginRequest, LoginResponse, UserInfo
from app.security import CurrentUser, create_token, get_current_user, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["认证"])


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest) -> LoginResponse:
    """
    账号密码登录，成功返回 JWT。

    Args:
        body: 用户名 + 明文密码。

    Returns:
        LoginResponse（access_token + 用户基本信息）。

    Raises:
        HTTPException 401: 用户名不存在或密码错误。

    # 用户不存在和密码错误必须返回同一句提示。
    # 如果分开提示"用户不存在"，攻击者就能拿这个接口批量枚举有效用户名，
    # 拿到用户名后再去撞库。统一成"用户名或密码错误"，把信息泄露降到最低。
    """
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, username, password_hash, nickname, role FROM users WHERE username = %s",
            (body.username.strip(),),
        )
        row = cur.fetchone()

    if row is None or not verify_password(body.password, row["password_hash"]):
        # 日志里区分真实原因（便于排查），但返回给前端的提示必须一样（防用户名枚举）。
        # 注意：日志绝不记录密码本身，只记用户名。
        logger.warning(
            "登录失败 | username=%s 原因=%s",
            body.username,
            "用户不存在" if row is None else "密码错误",
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
        )

    user = CurrentUser(user_id=row["id"], username=row["username"], role=row["role"])
    logger.info("登录成功 | username=%s user_id=%s role=%s", user.username, user.user_id, user.role)
    return LoginResponse(
        access_token=create_token(user),
        user=UserInfo(
            id=row["id"],
            username=row["username"],
            nickname=row["nickname"],
            role=row["role"],
        ),
    )


@router.get("/me", response_model=UserInfo)
def me(current: CurrentUser = Depends(get_current_user)) -> UserInfo:
    """
    返回当前 token 对应的用户。

    前端刷新页面后先调这个接口：能成功说明 token 还有效，
    失败会自动跳回登录页。比让前端自己解析 JWT 的 exp 靠谱。

    Args:
        current: 由 get_current_user 依赖注入的当前用户。

    Returns:
        UserInfo。

    Raises:
        HTTPException 401: token 无效或过期（由依赖抛出）。
    """
    # token 里没有 nickname（那是展示字段，不该塞进凭证里增加体积），
    # 所以这里回库里取一次。
    with get_conn() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id, username, nickname, role FROM users WHERE id = %s", (current.user_id,))
        row = cur.fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")

    return UserInfo(id=row["id"], username=row["username"], nickname=row["nickname"], role=row["role"])
