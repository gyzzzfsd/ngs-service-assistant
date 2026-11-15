"""
test_api_auth.py —— 登录接口测试

除了"能登录"，重点验证两条安全语义：
    - 密码错和用户不存在，对外提示必须完全一致（防用户名枚举）
    - 没有 token 时不能访问受保护接口
"""


def test_login_success(client):
    resp = client.post("/api/auth/login", json={"username": "chen", "password": "123456"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["access_token"]
    assert data["token_type"] == "bearer"
    assert data["user"]["username"] == "chen"
    assert data["user"]["nickname"] == "陈医生"
    assert data["user"]["role"] == "customer"


def test_login_response_never_leaks_password_hash(client):
    """响应体里绝不能出现 password_hash 字段 —— 一次疏忽就是全站账号泄露。"""
    resp = client.post("/api/auth/login", json={"username": "chen", "password": "123456"})

    body = resp.text
    assert "password_hash" not in body
    assert "8d969eef" not in body


def test_login_wrong_password(client):
    resp = client.post("/api/auth/login", json={"username": "chen", "password": "wrong"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == "用户名或密码错误"


def test_login_unknown_user_same_message(client):
    """
    用户不存在时，提示必须和"密码错误"一字不差。

    # 如果这里返回"用户不存在"，攻击者就能用这个接口批量试用户名，
    # 拿到有效用户名清单后再去撞库。两条路径的响应必须无法区分。
    """
    resp = client.post("/api/auth/login", json={"username": "no_such_user", "password": "123456"})

    assert resp.status_code == 401
    assert resp.json()["detail"] == "用户名或密码错误"


def test_login_empty_password_rejected_by_validation(client):
    """空密码在 Pydantic 层就被拦下（422），不会走到数据库查询。"""
    resp = client.post("/api/auth/login", json={"username": "chen", "password": ""})

    assert resp.status_code == 422


def test_me_requires_token(client):
    resp = client.get("/api/auth/me")

    assert resp.status_code == 401
    assert "未登录" in resp.json()["detail"]


def test_me_returns_current_user(client, auth_headers):
    resp = client.get("/api/auth/me", headers=auth_headers)

    assert resp.status_code == 200
    assert resp.json()["username"] == "chen"


def test_me_rejects_garbage_token(client):
    resp = client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-real-token"})

    assert resp.status_code == 401
    assert "无效" in resp.json()["detail"]
