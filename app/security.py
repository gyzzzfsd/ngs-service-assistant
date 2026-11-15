"""
security.py —— 密码哈希与登录凭证

职责：
    1. 密码的哈希与比对
    2. JWT 的签发与解析
    3. FastAPI 的 get_current_user 依赖（从请求头取 token → 解出当前用户）

不负责：
    用户注册（本项目不提供）、用户表的读写（登录查库在 api_auth.py）。

为什么认证单独一个文件：
    HTTP 接口和 WebSocket 都要验身份，两处共用同一套 token 逻辑。
    散在两个文件里最典型的后果是：改了一处的过期策略，另一处忘了改。
"""

import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings


@dataclass
class CurrentUser:
    """当前登录用户的身份信息，从 JWT 解析而来，不查库。"""

    user_id: int
    username: str
    role: str  # customer / agent


def hash_password(password: str) -> str:
    """
    计算密码的 SHA-256 十六进制摘要。

    Args:
        password: 明文密码。

    Returns:
        64 位小写十六进制字符串，例如 sha256('123456') 的结果。

    # 难点（安全性取舍）：这里故意用无盐 SHA-256，因为 seed_data.sql 里三个预置账号
    # 存的就是 sha256('123456')。换成 bcrypt 会让已入库的种子数据全部登录失败。
    # 生产环境必须换 bcrypt / argon2（每用户独立盐 + 慢哈希），否则彩虹表可直接反查；
    # 为保持种子数据开箱可用，沿用 sha256，并在 README 标注生产应改 bcrypt。
    """
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def verify_password(plain: str, password_hash: str) -> bool:
    """
    比对明文密码与库里的哈希值。

    Args:
        plain: 用户提交的明文密码。
        password_hash: users.password_hash 字段的值。

    Returns:
        True 表示密码正确。

    为什么用 hmac.compare_digest 而不是 ==：
        == 遇到第一个不同字符就返回，比较耗时会随"前多少个字符相同"变化。
        攻击者可以通过测量响应时间逐位猜出哈希（时序攻击）。
        compare_digest 是常量时间比较，无论差在哪一位耗时都一样。
    """
    return hmac.compare_digest(hash_password(plain), password_hash)


def create_token(user: CurrentUser) -> str:
    """
    签发 JWT。

    Args:
        user: 已通过密码校验的用户。

    Returns:
        签好的 token 字符串，前端放在 Authorization: Bearer <token> 里。

    # sub 必须是字符串。
    # PyJWT 2.10 起强制要求 "sub" 为 str，传 int 会在 decode 时抛
    # InvalidSubjectError（错误信息还很不直观）。所以这里 str(user_id)，
    # 取出来时再 int() 转回去。
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.user_id),
        "username": user.username,
        "role": user.role,
        "iat": now,
        "exp": now + timedelta(minutes=settings.jwt_expire_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> CurrentUser:
    """
    解析并校验 JWT，返回当前用户。

    Args:
        token: 前端传来的 token（不含 "Bearer " 前缀）。

    Returns:
        CurrentUser。HTTP 接口和 WebSocket 握手都用这个函数。

    Raises:
        jwt.PyJWTError: token 无效、签名不对或已过期。由调用方决定怎么处理
            （HTTP 转 401，WebSocket 直接关连接）。
    """
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    return CurrentUser(
        user_id=int(payload["sub"]),
        username=payload.get("username", ""),
        role=payload.get("role", "customer"),
    )


# auto_error=False：没带 token 时返回 None 而不是直接抛 403，
# 这样可以自定义成 401 + 更清楚的中文提示（前端好区分"没登录"和"没权限"）。
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> CurrentUser:
    """
    FastAPI 依赖：从请求头取出 token 并解出当前用户。

    用法：
        @router.get("/api/xxx")
        def handler(user: CurrentUser = Depends(get_current_user)):
            ...

    Raises:
        HTTPException 401: 没带 token、token 格式不对、过期或签名错误。
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="未登录，请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        return decode_token(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="登录凭证无效",
            headers={"WWW-Authenticate": "Bearer"},
        )
