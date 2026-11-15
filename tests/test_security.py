"""
test_security.py —— 密码哈希与 JWT 测试

这些用例在验证一件很朴素的事：凭证这块不能有任何"看起来能用但实际不对"的地方。
"""

from datetime import datetime, timedelta, timezone

import jwt
import pytest

from app.config import settings
from app.security import CurrentUser, create_token, decode_token, hash_password, verify_password

# sha256('123456') 的标准结果，也是 seed_data.sql 里三个预置账号的密码哈希。
# 写死这个值来断言，等于同时锁住了"哈希算法没被悄悄换掉"和"跟种子数据兼容"。
KNOWN_SHA256_123456 = "8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92"


def test_hash_matches_seed_data():
    """哈希必须和种子数据一致，否则三个预置账号全部登录失败。"""
    assert hash_password("123456") == KNOWN_SHA256_123456


def test_hash_is_stable():
    """同一密码两次哈希必须相同（无盐方案的前提，也是它的缺点）。"""
    assert hash_password("abc") == hash_password("abc")


def test_verify_password_ok():
    assert verify_password("123456", KNOWN_SHA256_123456) is True


def test_verify_password_wrong():
    assert verify_password("1234567", KNOWN_SHA256_123456) is False


def test_token_roundtrip():
    """签出来的 token 必须能解回同一个用户，三个字段都不能丢。"""
    user = CurrentUser(user_id=7, username="zhangsan", role="customer")
    token = create_token(user)

    decoded = decode_token(token)
    assert decoded.user_id == 7
    assert decoded.username == "zhangsan"
    assert decoded.role == "customer"


def test_token_sub_is_string():
    """
    PyJWT 2.10+ 要求 sub 必须是字符串，传 int 会在 decode 时抛 InvalidSubjectError。
    这条用例锁住"签发时转了 str"这个细节，避免以后有人"优化"掉 int()。
    """
    token = create_token(CurrentUser(user_id=1, username="a", role="customer"))
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    assert isinstance(payload["sub"], str)
    assert payload["sub"] == "1"


def test_expired_token_rejected():
    """过期 token 必须抛 ExpiredSignatureError，不能在业务层悄悄放行。"""
    payload = {
        "sub": "1",
        "username": "zhangsan",
        "role": "customer",
        "exp": datetime.now(timezone.utc) - timedelta(minutes=1),
    }
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_token(token)


def test_wrong_signature_rejected():
    """换一个密钥签的 token 必须被拒 —— 这是"用户不能自己伪造身份"的底线。"""
    token = jwt.encode(
        {"sub": "1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        "attacker-secret-not-the-real-one",
        algorithm="HS256",
    )
    with pytest.raises(jwt.InvalidSignatureError):
        decode_token(token)


def test_garbage_token_rejected():
    with pytest.raises(jwt.PyJWTError):
        decode_token("this.is.not.a.jwt")
