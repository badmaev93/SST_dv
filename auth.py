import os, sqlite3, secrets, string, logging, time
from collections import defaultdict
from urllib.parse import quote
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt as _bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

logger    = logging.getLogger(__name__)

_DB_PATH   = os.path.join(os.environ.get("SST_DATA_DIR", "data"), "users.db")
_SECRET    = os.environ.get("JWT_SECRET", secrets.token_hex(32))
_ALG       = "HS256"
_EXP_DAYS  = int(os.environ.get("JWT_EXP_DAYS", "30"))
AUTH_ON    = os.environ.get("AUTH_REQUIRED", "0") == "1"

def _hash_pw(pw: str) -> str:
    return _bcrypt.hashpw(pw.encode(), _bcrypt.gensalt()).decode()

def _verify_pw(pw: str, hashed: str) -> bool:
    return _bcrypt.checkpw(pw.encode(), hashed.encode())

_bearer = HTTPBearer(auto_error=False)

_failed: dict = defaultdict(list)  # ip -> [timestamps]
_locked: dict = {} # ip -> unlock_timestamp
_MAX_ATTEMPTS = 5
_WINDOW        = 300   # 5 мин — окно подсчёта
_LOCKOUT       = 900   # 15 мин — блокировка

# SECURITY: Enforce a strict lockout period to prevent brute-force attacks
def _check_brute(ip: str):
    now = time.time()
    if ip in _locked:
        if now < _locked[ip]:
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Слишком много попыток. Подождите 15 минут.")
        else:
            _locked.pop(ip, None)

    _failed[ip] = [t for t in _failed[ip] if now - t < _WINDOW]
    if len(_failed[ip]) >= _MAX_ATTEMPTS:
        _locked[ip] = now + _LOCKOUT
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Слишком много попыток. Подождите 15 минут.")

def _record_fail(ip: str):
    _failed[ip].append(time.time())


@contextmanager
def _db():
    con = sqlite3.connect(_DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.commit()
        con.close()


def init_db():
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with _db() as con:
        con.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT    UNIQUE NOT NULL,
                pw_hash    TEXT    NOT NULL,
                is_admin   INTEGER NOT NULL DEFAULT 0,
                active     INTEGER NOT NULL DEFAULT 1,
                created_at TEXT    NOT NULL
            );
            CREATE TABLE IF NOT EXISTS invites (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                email      TEXT    NOT NULL,
                token      TEXT    UNIQUE NOT NULL,
                used       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL
            );
        """)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_password(n: int = 12) -> str:
    abc = string.ascii_letters + string.digits
    return "".join(secrets.choice(abc) for _ in range(n))


def _make_token(email: str, is_admin: bool) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=_EXP_DAYS)
    return jwt.encode({"sub": email, "admin": is_admin, "exp": exp}, _SECRET, algorithm=_ALG)


def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, _SECRET, algorithms=[_ALG])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Токен истёк")
    except jwt.InvalidTokenError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный токен")


def _get_user(email: str) -> Optional[sqlite3.Row]:
    with _db() as con:
        return con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()


def get_current_user(creds: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not AUTH_ON:
        return {"email": "local", "is_admin": True}
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Требуется авторизация")
    payload = _decode_token(creds.credentials)
    user    = _get_user(payload["sub"])
    if not user or not user["active"]:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Пользователь не найден")
    return {"email": user["email"], "is_admin": bool(user["is_admin"])}


def require_admin(user: dict = Depends(get_current_user)):
    if not user["is_admin"]:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Нет прав")
    return user


def _make_mailto(to_email: str, password: str) -> str:
    url_win     = os.environ.get("APP_URL_WIN", "")
    url_mac     = os.environ.get("APP_URL_MAC", "")
    url_android = os.environ.get("APP_URL_ANDROID", "")

    links = []
    if url_win:     links.append(f"  Windows:  {url_win}")
    if url_mac:     links.append(f"  macOS:    {url_mac}")
    if url_android: links.append(f"  Android:  {url_android}")
    links_block = "\n".join(links) if links else "  (ссылки будут добавлены позже)"

    body = (
        f"Вы приглашены в ERA5 ДВ.\n\n"
        f"Email:  {to_email}\n"
        f"Пароль: {password}\n\n"
        f"Скачать приложение:\n{links_block}\n\n"
    )
    subject = quote("Приглашение в ERA5 ДВ")
    body_enc = quote(body)
    return f"mailto:{to_email}?subject={subject}&body={body_enc}"


# ── Pydantic models ───────────────────────────────────────────────────────────

class LoginReq(BaseModel):
    email:    EmailStr
    password: str

class InviteReq(BaseModel):
    email: EmailStr

class ChangePwReq(BaseModel):
    old_password: str
    new_password: str


# ── Route handlers (подключаются в app.py через router) ──────────────────────

from fastapi import APIRouter
router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login")
def login(req: LoginReq, request: Request):
    ip = request.client.host
    # Use X-Forwarded-For if available to handle proxies
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # SECURITY: Use the right-most IP to prevent spoofing by the client
        ip = forwarded.split(",")[-1].strip()

    _check_brute(ip)

    user = _get_user(req.email)

    # Timing attack prevention: always compute hash
    is_valid = False
    if user and user["active"]:
        is_valid = _verify_pw(req.password, user["pw_hash"])
    else:
        # Compute a dummy hash to match the time of _verify_pw
        # We need a valid bcrypt hash format for checkpw to actually do the work
        _verify_pw(req.password, "$2b$12$NqLzOYY1X5tK6XmYg1qY.OyqG/Qo6/V66.Q66Z6q6v.Z66.Z66.Z6")

    if not is_valid:
        _record_fail(ip)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Неверный email или пароль")

    _failed.pop(ip, None)
    return {"token": _make_token(req.email, bool(user["is_admin"])), "email": req.email}


@router.post("/invite")
def invite(req: InviteReq, _admin=Depends(require_admin)):
    # SECURITY: Restrict invites to admins only to prevent unauthorized account creation
    if _get_user(req.email):
        raise HTTPException(400, "Пользователь уже существует")
    password = _gen_password()
    pw_hash  = _hash_pw(password)
    with _db() as con:
        con.execute(
            "INSERT INTO users (email, pw_hash, is_admin, active, created_at) VALUES (?,?,0,1,?)",
            (req.email, pw_hash, _now()))
    mailto = _make_mailto(req.email, password)
    return {"ok": True, "email": req.email, "mailto": mailto}


@router.get("/me")
def me(user=Depends(get_current_user)):
    return user


@router.post("/change_password")
def change_password(req: ChangePwReq, user=Depends(get_current_user)):
    row = _get_user(user["email"])
    if not row or not _verify_pw(req.old_password, row["pw_hash"]):
        raise HTTPException(400, "Неверный текущий пароль")
    new_hash = _hash_pw(req.new_password)
    with _db() as con:
        con.execute("UPDATE users SET pw_hash=? WHERE email=?", (new_hash, user["email"]))
    return {"ok": True}


@router.get("/users")
def list_users(_admin=Depends(require_admin)):
    with _db() as con:
        rows = con.execute(
            "SELECT id, email, is_admin, active, created_at FROM users ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/deactivate")
def deactivate(email: str, _admin=Depends(require_admin)):
    with _db() as con:
        con.execute("UPDATE users SET active=0 WHERE email=?", (email,))
    return {"ok": True}


def create_admin(email: str, password: str):
    """Запустить один раз: python -c \"from auth import create_admin; create_admin('x@x.ru','pass')\" """
    init_db()
    pw_hash = _hash_pw(password)
    with _db() as con:
        con.execute(
            "INSERT OR REPLACE INTO users (email, pw_hash, is_admin, active, created_at) VALUES (?,?,1,1,?)",
            (email, pw_hash, _now()))
    print(f"Admin created: {email}")
