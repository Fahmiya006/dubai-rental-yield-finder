"""Dubai Yield Finder server: accounts, saved data (SQLite) and the AI agent relay.

    export ANTHROPIC_API_KEY=sk-ant-...     # optional: enables the LLM agent
    python server.py                        # http://localhost:8000

Stdlib only. Data lives in data/app.db (override with RYF_DB).
Env: PORT, HOST (0.0.0.0 to reach it from a phone on your Wi-Fi), COOKIE_SECURE=1 (set when served
over HTTPS), TRUST_PROXY=1 (behind a hosting proxy), ANTHROPIC_MODEL, ANTHROPIC_BASE_URL, RYF_DB.

Security model
- Passwords: PBKDF2-HMAC-SHA256, 600k iterations, random salt per user. Never stored in plain text.
- Sessions: random 256-bit token in an HttpOnly, SameSite=Lax cookie. Only its SHA-256 is stored.
- Every write needs the X-Requested-With header and a same-origin Origin (basic CSRF defence).
- Login attempts are throttled. All SQL is parameterised. The API key never reaches the browser.
"""
from __future__ import annotations
import hashlib, hmac, json, os, re, secrets, sqlite3, sys, time, urllib.error, urllib.request
from contextlib import contextmanager
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
CFG = {
    "db": Path(os.environ.get("RYF_DB", ROOT / "data" / "app.db")),
    "secure": os.environ.get("COOKIE_SECURE") == "1",
    "iters": 600_000,
    "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    "base": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
    "max_body": 2_500_000,
}
SESSION_SECONDS = 30 * 24 * 3600
KEY_LIMITS = {"prefs": 50_000, "watch": 50_000, "calc": 50_000, "listing": 50_000,
              "chat": 200_000, "dataset": 2_000_000}
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
RULES = (
    'You are the assistant inside "Dubai Yield Finder", an app that helps people (often first-time buyers, '
    "expats and overseas Indian investors) judge Dubai rental properties. Use only the DATA and COST "
    "ASSUMPTIONS provided for any number; if the data is synthetic, say the figures are sample numbers. "
    "Never invent prices, laws or visa rules; for legal, visa, tax or mortgage-eligibility questions give "
    "general orientation and tell the user to verify with official sources (DLD, a licensed broker, a bank). "
    "Do not promise returns. Keep answers under 150 words, plain English, key number first."
)
SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
  pw TEXT NOT NULL, created INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  expires INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS user_data(
  user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE, k TEXT NOT NULL, v TEXT NOT NULL,
  updated INTEGER NOT NULL, PRIMARY KEY(user_id, k));
"""
FAILS: dict[str, list[float]] = {}
HITS: dict[str, list[float]] = {}


class ApiError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg); self.code, self.msg = code, msg


# ---------- database ----------
@contextmanager
def db():
    c = sqlite3.connect(CFG["db"], timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()


def init_db():
    Path(CFG["db"]).parent.mkdir(parents=True, exist_ok=True)
    with db() as c:
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)


# ---------- passwords and sessions ----------
def hash_pw(pw: str, salt: bytes | None = None, iters: int | None = None) -> str:
    salt = salt or os.urandom(16)
    iters = iters or CFG["iters"]
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, iters)
    return f"pbkdf2${iters}${salt.hex()}${dk.hex()}"


def verify_pw(pw: str, stored: str) -> bool:
    try:
        _, iters, salt, dk = stored.split("$")
        test = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters))
        return hmac.compare_digest(test.hex(), dk)
    except Exception:  # noqa: BLE001
        return False


def sha(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def throttle(bucket: dict, key: str, limit: int, window: int):
    now = time.time()
    bucket[key] = [t for t in bucket.get(key, []) if now - t < window]
    if len(bucket[key]) >= limit:
        raise ApiError(429, "Too many attempts. Wait a few minutes and try again.")


def note(bucket: dict, key: str):
    bucket.setdefault(key, []).append(time.time())


def new_session(c, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    c.execute("INSERT INTO sessions VALUES(?,?,?)", (sha(token), user_id, int(time.time()) + SESSION_SECONDS))
    c.execute("DELETE FROM sessions WHERE expires<?", (int(time.time()),))
    return token


def cookie_header(token: str, max_age: int) -> tuple[str, str]:
    v = f"ryf_session={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}"
    return ("set-cookie", v + ("; Secure" if CFG["secure"] else ""))


def clean_name(n, email):
    n = re.sub(r"\s+", " ", str(n or "")).strip()[:60]
    return n or email.split("@")[0]


# ---------- LLM relay ----------
def _post_llm(body):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ApiError(503, "The AI is not configured on this server (no ANTHROPIC_API_KEY).")
    req = urllib.request.Request(CFG["base"] + "/v1/messages", data=json.dumps(body).encode(), method="POST",
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def call_llm(messages, context):
    data = _post_llm({"model": CFG["model"], "max_tokens": 500, "system": RULES + "\n\n" + context[:12000],
                      "messages": messages[-12:]})
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def call_agent(system, messages, tools):
    """One agent turn. The browser runs the tools (the data lives there) and calls again with results."""
    safe = [{"name": str(t["name"])[:64], "description": str(t.get("description", ""))[:600],
             "input_schema": t.get("input_schema", {"type": "object"})} for t in tools[:15]]
    data = _post_llm({"model": CFG["model"], "max_tokens": 700, "system": RULES + "\n\n" + str(system)[:6000],
                      "messages": messages[-40:], "tools": safe})
    return {"content": data.get("content", []), "stop_reason": data.get("stop_reason")}


def clean_msgs(msgs):
    out = []
    for m in msgs if isinstance(msgs, list) else []:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"].strip():
            if out and out[-1]["role"] == m["role"]:
                out[-1]["content"] += "\n" + m["content"][:2000]
            else:
                out.append({"role": m["role"], "content": m["content"][:2000]})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


# ---------- static ----------
HEAD = (b'<!doctype html><html lang="en"><meta charset="utf-8">'
        b'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
        b'<meta name="theme-color" content="#FFFFFF"><meta name="apple-mobile-web-app-capable" content="yes">'
        b'<meta name="mobile-web-app-capable" content="yes">'
        b'<link rel="manifest" href="/manifest.json"><link rel="icon" href="/assets/icon-192.png" type="image/png">'
        b'<link rel="apple-touch-icon" href="/assets/apple-touch-icon.png">')
MANIFEST = json.dumps({"name": "Dubai Yield Finder", "short_name": "Yield Finder", "start_url": "/", "display": "standalone",
                       "background_color": "#FFFFFF", "theme_color": "#FFFFFF",
                       "icons": [{"src": "/assets/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
                                 {"src": "/assets/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any"},
                                 {"src": "/assets/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}]}).encode()


MINI = {"model": None, "tried": False}
MINI_PATH = ROOT / "llm" / "model.npz"


def mini_model():
    """Loads the from-scratch mini GPT once. Needs numpy; the rest of the server stays stdlib-only."""
    if not MINI["tried"]:
        MINI["tried"] = True
        try:
            sys.path.insert(0, str(ROOT))
            from llm import tinygpt
            MINI["model"] = (tinygpt, *tinygpt.load(MINI_PATH)) if MINI_PATH.exists() else None
        except Exception as e:  # numpy missing, bad file, ...
            print("mini-GPT not available:", e)
            MINI["model"] = None
    return MINI["model"]


ASSETS = {"icon-192.png", "icon-512.png", "icon-maskable-512.png", "apple-touch-icon.png", "logo.png"}


class H(BaseHTTPRequestHandler):
    server_version = "RYF"

    def _send(self, code, payload, ctype="application/json", headers=()):
        raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(raw)))
        self.send_header("x-content-type-options", "nosniff")
        self.send_header("x-frame-options", "DENY")
        if ctype == "application/json":
            self.send_header("cache-control", "no-store")
        for k, v in headers:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(raw)

    def _body(self):
        n = int(self.headers.get("content-length") or 0)
        if n > CFG["max_body"]:
            raise ApiError(413, "Request too large")
        try:
            d = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise ApiError(400, "Invalid JSON")
        if not isinstance(d, dict):
            raise ApiError(400, "Invalid JSON")
        return d

    def _csrf(self):
        if self.headers.get("x-requested-with") != "ryf":
            raise ApiError(403, "Blocked request")
        origin = self.headers.get("origin")
        if origin and urlparse(origin).netloc != self.headers.get("host"):
            raise ApiError(403, "Blocked request")

    def _token(self):
        raw = self.headers.get("cookie")
        if not raw:
            return None
        try:
            c = SimpleCookie(raw)
            return c["ryf_session"].value if "ryf_session" in c else None
        except Exception:  # noqa: BLE001
            return None

    def _user(self):
        t = self._token()
        if not t:
            return None
        with db() as c:
            return c.execute("SELECT u.* FROM sessions s JOIN users u ON u.id=s.user_id "
                             "WHERE s.token_hash=? AND s.expires>?", (sha(t), int(time.time()))).fetchone()

    def _need(self):
        u = self._user()
        if not u:
            raise ApiError(401, "Please log in")
        return u

    def _ip(self):
        # Behind a hosting proxy (Render, Fly) every request arrives from the proxy's address, so the per-IP limits
        # would be shared by all visitors. With TRUST_PROXY=1 use the client address the proxy reports.
        if os.environ.get("TRUST_PROXY") == "1":
            fwd = self.headers.get("x-forwarded-for", "").split(",")[0].strip()
            if fwd:
                return fwd[:64]
        return self.client_address[0]

    def _public(self, u):
        return {"email": u["email"], "name": u["name"]}

    # ---- routing ----
    def _dispatch(self, method):
        path = urlparse(self.path).path
        try:
            if method != "GET" and path.startswith("/api/"):
                self._csrf()
            if method == "GET":
                return self._get(path)
            if method == "POST":
                return self._post(path)
            if method == "PUT":
                return self._put(path)
            if method == "DELETE":
                return self._delete(path)
        except ApiError as e:
            self._send(e.code, {"error": e.msg})
        except urllib.error.HTTPError as e:
            self._send(502, {"error": f"The AI service returned an error ({e.code})."})
        except (urllib.error.URLError, TimeoutError):
            self._send(502, {"error": "Could not reach the AI service."})
        except Exception:  # noqa: BLE001
            self._send(500, {"error": "Server error"})

    def do_GET(self): self._dispatch("GET")
    def do_POST(self): self._dispatch("POST")
    def do_PUT(self): self._dispatch("PUT")
    def do_DELETE(self): self._dispatch("DELETE")

    def _get(self, path):
        if path == "/api/health":
            return self._send(200, {"llm": bool(os.environ.get("ANTHROPIC_API_KEY")), "model": CFG["model"],
                                    "minigpt": mini_model() is not None})
        if path == "/api/me":
            u = self._user()
            return self._send(200, {"user": self._public(u) if u else None})
        if path == "/api/data":
            u = self._need()
            with db() as c:
                rows = c.execute("SELECT k,v FROM user_data WHERE user_id=?", (u["id"],)).fetchall()
            return self._send(200, {"data": {r["k"]: json.loads(r["v"]) for r in rows}})
        if path in ("/", "/index.html"):
            return self._send(200, HEAD + (ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/manifest.json":
            return self._send(200, MANIFEST, "application/manifest+json")
        if path.startswith("/assets/") and path.count("/") == 2 and path[8:] in ASSETS:
            return self._send(200, (ROOT / "app" / "assets" / path[8:]).read_bytes(), "image/png",
                              [("cache-control", "public, max-age=86400")])
        self._send(404, {"error": "Not found"})

    def _post(self, path):
        b = self._body()
        if path == "/api/signup":
            throttle(HITS, "signup|" + self._ip(), 10, 3600)
            email = str(b.get("email", "")).strip().lower()
            pw = str(b.get("password", ""))
            if not EMAIL_RE.match(email) or len(email) > 254:
                raise ApiError(400, "Enter a valid email address.")
            if len(pw) < 8 or len(pw) > 200:
                raise ApiError(400, "Password needs at least 8 characters.")
            note(HITS, "signup|" + self._ip())
            with db() as c:
                if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
                    raise ApiError(409, "This email already has an account. Log in instead.")
                name = clean_name(b.get("name"), email)
                uid = c.execute("INSERT INTO users(email,name,pw,created) VALUES(?,?,?,?)",
                                (email, name, hash_pw(pw), int(time.time()))).lastrowid
                token = new_session(c, uid)
            return self._send(201, {"user": {"email": email, "name": name}}, headers=[cookie_header(token, SESSION_SECONDS)])
        if path == "/api/login":
            email = str(b.get("email", "")).strip().lower()
            pw = str(b.get("password", ""))[:200]
            key = self._ip() + "|" + email
            throttle(FAILS, key, 8, 600)
            with db() as c:
                u = c.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
                ok = verify_pw(pw, u["pw"] if u else DUMMY)
                if not (u and ok):
                    note(FAILS, key)
                    raise ApiError(401, "Email or password is wrong.")
                token = new_session(c, u["id"])
            FAILS.pop(key, None)
            return self._send(200, {"user": self._public(u)}, headers=[cookie_header(token, SESSION_SECONDS)])
        if path == "/api/logout":
            t = self._token()
            if t:
                with db() as c:
                    c.execute("DELETE FROM sessions WHERE token_hash=?", (sha(t),))
            return self._send(200, {"ok": True}, headers=[cookie_header("", 0)])
        if path == "/api/password":
            u = self._need()
            new = str(b.get("new", ""))
            if len(new) < 8 or len(new) > 200:
                raise ApiError(400, "New password needs at least 8 characters.")
            if not verify_pw(str(b.get("old", ""))[:200], u["pw"]):
                raise ApiError(403, "Current password is wrong.")
            with db() as c:
                c.execute("UPDATE users SET pw=? WHERE id=?", (hash_pw(new), u["id"]))
                c.execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))  # sign out everywhere else
                token = new_session(c, u["id"])
            return self._send(200, {"ok": True}, headers=[cookie_header(token, SESSION_SECONDS)])
        if path == "/api/minigpt":
            u = self._need()
            throttle(HITS, f"mini|{u['id']}", 30, 60)
            note(HITS, f"mini|{u['id']}")
            m = mini_model()
            if m is None:
                raise ApiError(503, "The mini GPT is not trained yet. Run: python3 -m llm.train")
            prompt = str(b.get("prompt", ""))
            if not prompt.startswith("F: ") or "\nQ: " not in prompt or len(prompt) > 600 or not prompt.endswith("\nA: "):
                raise ApiError(400, "Bad prompt")
            tg, params, cfg = m
            return self._send(200, {"text": tg.generate_fast(params, cfg, prompt, max_new=200).strip()})
        if path in ("/api/chat", "/api/agent"):
            u = self._need()
            throttle(HITS, f"ai|{u['id']}", 20, 60)
            note(HITS, f"ai|{u['id']}")
            if path == "/api/agent":
                if not isinstance(b.get("messages"), list) or not isinstance(b.get("tools"), list):
                    raise ApiError(400, "messages and tools required")
                return self._send(200, call_agent(b.get("system", ""), b["messages"], b["tools"]))
            msgs = clean_msgs(b.get("messages"))
            if not msgs:
                raise ApiError(400, "No question")
            return self._send(200, {"text": call_llm(msgs, str(b.get("context", "")))})
        self._send(404, {"error": "Not found"})

    def _put(self, path):
        u = self._need()
        b = self._body()
        if path == "/api/profile":
            name = clean_name(b.get("name"), u["email"])
            with db() as c:
                c.execute("UPDATE users SET name=? WHERE id=?", (name, u["id"]))
            return self._send(200, {"user": {"email": u["email"], "name": name}})
        m = re.fullmatch(r"/api/data/(\w+)", path)
        if m and m.group(1) in KEY_LIMITS:
            k = m.group(1)
            if "value" not in b:
                raise ApiError(400, "value required")
            v = json.dumps(b["value"], separators=(",", ":"))
            if len(v) > KEY_LIMITS[k]:
                raise ApiError(413, "Too much data for " + k)
            with db() as c:
                c.execute("INSERT INTO user_data(user_id,k,v,updated) VALUES(?,?,?,?) "
                          "ON CONFLICT(user_id,k) DO UPDATE SET v=excluded.v, updated=excluded.updated",
                          (u["id"], k, v, int(time.time())))
            return self._send(200, {"ok": True})
        self._send(404, {"error": "Not found"})

    def _delete(self, path):
        u = self._need()
        m = re.fullmatch(r"/api/data/(\w+)", path)
        if m and m.group(1) in KEY_LIMITS:
            with db() as c:
                c.execute("DELETE FROM user_data WHERE user_id=? AND k=?", (u["id"], m.group(1)))
            return self._send(200, {"ok": True})
        if path == "/api/account":
            b = self._body()
            if not verify_pw(str(b.get("password", ""))[:200], u["pw"]):
                raise ApiError(403, "Password is wrong.")
            with db() as c:
                c.execute("DELETE FROM users WHERE id=?", (u["id"],))
            return self._send(200, {"ok": True}, headers=[cookie_header("", 0)])
        self._send(404, {"error": "Not found"})

    def log_message(self, *a):
        pass


DUMMY = hash_pw("dummy-password-for-timing")


def make_server(host="127.0.0.1", port=8000):
    init_db()
    return ThreadingHTTPServer((host, port), H)


if __name__ == "__main__":
    host, port = os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", 8000))
    srv = make_server(host, port)
    print(f"http://{'localhost' if host == '127.0.0.1' else host}:{port}  db={CFG['db']}  "
          f"llm={'on' if os.environ.get('ANTHROPIC_API_KEY') else 'OFF (set ANTHROPIC_API_KEY)'}  model={CFG['model']}")
    srv.serve_forever()
