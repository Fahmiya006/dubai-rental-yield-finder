import json, pathlib, sys, threading, urllib.request, urllib.error
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import pytest
import server


@pytest.fixture()
def app(tmp_path, monkeypatch):
    server.CFG["db"] = tmp_path / "t.db"
    server.CFG["iters"] = 1000
    server.FAILS.clear(); server.HITS.clear()
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    srv = server.make_server("127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


class Client:
    def __init__(self, base):
        self.base, self.cookie = base, None

    def call(self, method, path, body=None, csrf=True, raw=False):
        h = {"content-type": "application/json"}
        if csrf:
            h["x-requested-with"] = "ryf"
        if self.cookie:
            h["cookie"] = self.cookie
        req = urllib.request.Request(self.base + path, method=method, headers=h,
                                     data=None if body is None else json.dumps(body).encode())
        try:
            r = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            r = e
        sc = r.headers.get("set-cookie")
        if sc:
            tok = sc.split(";")[0]
            self.cookie = None if tok.endswith("=") else tok
        data = r.read()
        return r.status if hasattr(r, "status") else r.code, (data if raw else json.loads(data))


def signup(c, email="a@b.co", pw="secret123", name="Ana"):
    return c.call("POST", "/api/signup", {"email": email, "password": pw, "name": name})


def test_signup_login_logout_roundtrip(app):
    c = Client(app)
    assert c.call("GET", "/api/me")[1]["user"] is None
    st, j = signup(c)
    assert st == 201 and j["user"]["name"] == "Ana"
    assert c.call("GET", "/api/me")[1]["user"]["email"] == "a@b.co"
    c.call("POST", "/api/logout", {})
    assert c.call("GET", "/api/me")[1]["user"] is None
    assert c.call("POST", "/api/login", {"email": "a@b.co", "password": "wrongpass"})[0] == 401
    assert c.call("POST", "/api/login", {"email": "A@B.co", "password": "secret123"})[0] == 200


def test_validation_and_duplicates(app):
    c = Client(app)
    assert signup(c, email="nope")[0] == 400
    assert signup(c, pw="short")[0] == 400
    assert signup(c)[0] == 201
    assert signup(Client(app))[0] == 409


def test_password_is_hashed_in_db(app):
    signup(Client(app), pw="supersecret1")
    import sqlite3
    row = sqlite3.connect(server.CFG["db"]).execute("select pw from users").fetchone()[0]
    assert "supersecret1" not in row and row.startswith("pbkdf2$")


def test_data_persists_across_sessions(app):
    c = Client(app)
    signup(c)
    assert c.call("PUT", "/api/data/watch", {"value": ["Arjan|Studio"]})[0] == 200
    assert c.call("PUT", "/api/data/prefs", {"value": {"inr": True, "fx": 24.5}})[0] == 200
    c.call("POST", "/api/logout", {})
    c2 = Client(app)  # a brand new device
    c2.call("POST", "/api/login", {"email": "a@b.co", "password": "secret123"})
    d = c2.call("GET", "/api/data")[1]["data"]
    assert d["watch"] == ["Arjan|Studio"] and d["prefs"]["fx"] == 24.5
    c2.call("DELETE", "/api/data/watch")
    assert "watch" not in c2.call("GET", "/api/data")[1]["data"]


def test_users_cannot_see_each_others_data(app):
    a, b = Client(app), Client(app)
    signup(a); signup(b, email="x@y.co")
    a.call("PUT", "/api/data/watch", {"value": ["secret"]})
    assert b.call("GET", "/api/data")[1]["data"] == {}


def test_auth_and_csrf_guards(app):
    c = Client(app)
    assert c.call("GET", "/api/data")[0] == 401
    assert c.call("PUT", "/api/data/watch", {"value": []})[0] == 401
    assert c.call("POST", "/api/agent", {"messages": [], "tools": []})[0] == 401
    assert c.call("POST", "/api/signup", {"email": "a@b.co", "password": "secret123"}, csrf=False)[0] == 403
    signup(c)
    assert c.call("PUT", "/api/data/watch", {"value": []}, csrf=False)[0] == 403
    assert c.call("PUT", "/api/data/evil", {"value": 1})[0] == 404
    assert c.call("PUT", "/api/data/watch", {"value": "x" * 60_000})[0] == 413


def test_login_throttle(app):
    c = Client(app)
    signup(c)
    c.call("POST", "/api/logout", {})
    codes = [c.call("POST", "/api/login", {"email": "a@b.co", "password": "badbadbad"})[0] for _ in range(10)]
    assert codes[:8] == [401] * 8 and codes[8] == 429
    assert c.call("POST", "/api/login", {"email": "a@b.co", "password": "secret123"})[0] == 429


def test_change_password_signs_out_other_devices(app):
    a, b = Client(app), Client(app)
    signup(a)
    b.call("POST", "/api/login", {"email": "a@b.co", "password": "secret123"})
    assert a.call("POST", "/api/password", {"old": "wrong", "new": "newsecret1"})[0] == 403
    assert a.call("POST", "/api/password", {"old": "secret123", "new": "newsecret1"})[0] == 200
    assert a.call("GET", "/api/me")[1]["user"] is not None
    assert b.call("GET", "/api/me")[1]["user"] is None
    assert Client(app).call("POST", "/api/login", {"email": "a@b.co", "password": "newsecret1"})[0] == 200


def test_profile_and_delete_account(app):
    c = Client(app)
    signup(c)
    assert c.call("PUT", "/api/profile", {"name": "  New   Name "})[1]["user"]["name"] == "New Name"
    c.call("PUT", "/api/data/watch", {"value": [1]})
    assert c.call("DELETE", "/api/account", {"password": "nope"})[0] == 403
    assert c.call("DELETE", "/api/account", {"password": "secret123"})[0] == 200
    assert c.call("POST", "/api/login", {"email": "a@b.co", "password": "secret123"})[0] == 401
    import sqlite3
    con = sqlite3.connect(server.CFG["db"])
    assert con.execute("select count(*) from user_data").fetchone()[0] == 0  # cascade


def test_agent_without_key_is_clear_error(app):
    c = Client(app)
    signup(c)
    st, j = c.call("POST", "/api/agent", {"messages": [{"role": "user", "content": "hi"}], "tools": []})
    assert st == 503 and "ANTHROPIC_API_KEY" in j["error"]


def test_serves_app_and_manifest(app):
    c = Client(app)
    st, html = c.call("GET", "/", raw=True)
    assert st == 200 and b"manifest.json" in html and b"Dubai Yield Finder" in html
    assert c.call("GET", "/manifest.json")[1]["display"] == "standalone"


def test_serves_png_icons_and_rejects_other_assets(app):
    c = Client(app)
    for name in ("icon-192.png", "icon-512.png", "icon-maskable-512.png", "apple-touch-icon.png"):
        st, data = c.call("GET", "/assets/" + name, raw=True)
        assert st == 200 and data[:8] == b"\x89PNG\r\n\x1a\n"
    assert c.call("GET", "/assets/logo-source.jpg", raw=True)[0] == 404
    assert c.call("GET", "/assets/../server.py", raw=True)[0] == 404
    assert all(i["type"] == "image/png" for i in c.call("GET", "/manifest.json")[1]["icons"])


def test_minigpt_requires_login_and_validates_prompt(app):
    c = Client(app)
    assert c.call("POST", "/api/minigpt", {"prompt": "x"})[0] == 401
    signup(c)
    assert c.call("POST", "/api/minigpt", {"prompt": "hello"})[0] in (400, 503)
    assert "minigpt" in c.call("GET", "/api/health")[1]


def test_forwarded_ip_only_trusted_behind_a_proxy(app, monkeypatch):
    # without TRUST_PROXY a spoofed header must not change the limits: 11 sign-ups from one address still hit the cap
    c = Client(app)
    codes = []
    for i in range(12):
        req = urllib.request.Request(app + "/api/signup", method="POST", data=json.dumps(
            {"email": f"u{i}@b.co", "password": "secret123", "name": "U"}).encode(),
            headers={"content-type": "application/json", "x-requested-with": "ryf", "x-forwarded-for": f"9.9.9.{i}"})
        try:
            codes.append(urllib.request.urlopen(req).status)
        except urllib.error.HTTPError as e:
            codes.append(e.code)
    assert 429 in codes
