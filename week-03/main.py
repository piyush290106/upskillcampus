import os
import io
import re
import hmac
import json
import base64
import hashlib
import string
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends, status, Form, Header, Response
from fastapi.responses import RedirectResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlmodel import SQLModel, Field, create_engine, Session, select
from pydantic import BaseModel, HttpUrl

# Optional deps (QR + Redis)
try:
    import qrcode
except Exception:
    qrcode = None

try:
    from redis import asyncio as aioredis  # redis>=5
except Exception:
    aioredis = None

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./shortener.db")
BASE_URL = os.getenv("BASE_URL")  # e.g. https://sho.rt
CODE_LENGTH_MIN = int(os.getenv("CODE_LENGTH_MIN", 4))
CODE_LENGTH_MAX = int(os.getenv("CODE_LENGTH_MAX", 10))
REDIS_URL = os.getenv("REDIS_URL")
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
TOKEN_TTL_SECONDS = int(os.getenv("TOKEN_TTL_SECONDS", "86400"))  # 24h

# --------------------------------------------------------------------------------------
# App + dirs
# --------------------------------------------------------------------------------------
app = FastAPI(title="TinyLink API", version="3.0")
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# --------------------------------------------------------------------------------------
# DB models
# --------------------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, echo=False, future=True)

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class Url(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    long_url: str
    short_code: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    clicks: int = Field(default=0)
    expires_at: Optional[datetime] = None
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")

class ClickLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url_id: int = Field(index=True, foreign_key="url.id")
    ts: datetime = Field(default_factory=datetime.utcnow)
    referrer: Optional[str] = None
    user_agent: Optional[str] = None
    ip: Optional[str] = None

def init_db():
    SQLModel.metadata.create_all(engine)

# --------------------------------------------------------------------------------------
# Redis (optional)
# --------------------------------------------------------------------------------------
redis_client = None
if REDIS_URL and aioredis is not None:
    redis_client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
BASE62 = string.digits + string.ascii_lowercase + string.ascii_uppercase
URL_REGEX = re.compile(r"^https?://", re.IGNORECASE)
ALIAS_REGEX = re.compile(r"^[a-zA-Z0-9]{4,16}$")

def base62_encode(n: int) -> str:
    if n == 0:
        return BASE62[0]
    s = []
    while n > 0:
        n, r = divmod(n, 62)
        s.append(BASE62[r])
    return "".join(reversed(s))

def is_valid_url(u: str) -> bool:
    return bool(u and URL_REGEX.match(u))

def is_valid_alias(a: str) -> bool:
    if not a:
        return False
    if len(a) < CODE_LENGTH_MIN or len(a) > min(CODE_LENGTH_MAX, 16):
        return False
    return bool(ALIAS_REGEX.match(a))

def full_short(code: str, request: Optional[Request] = None) -> str:
    if BASE_URL:
        return f"{BASE_URL.rstrip('/')}/{code}"
    if request:
        base = str(request.base_url).rstrip("/")
        return f"{base}/{code}"
    return f"//{code}"

async def cache_get(code: str) -> Optional[str]:
    if not redis_client:
        return None
    return await redis_client.get(f"url:{code}")

async def cache_set(code: str, long_url: str, ttl: Optional[int] = None) -> None:
    if not redis_client:
        return
    key = f"url:{code}"
    if ttl and ttl > 0:
        await redis_client.setex(key, ttl, long_url)
    else:
        await redis_client.set(key, long_url)

async def cache_del(code: str) -> None:
    if not redis_client:
        return
    await redis_client.delete(f"url:{code}")

def remaining_ttl(expires_at: Optional[datetime]) -> Optional[int]:
    if not expires_at:
        return None
    delta = int((expires_at - datetime.utcnow()).total_seconds())
    return delta if delta > 0 else None

def jlog(event: str, **kw):
    print(json.dumps({"event": event, "ts": datetime.utcnow().isoformat() + "Z", **kw}))

# --------------------------------------------------------------------------------------
# Auth (PBKDF2 + HMAC token) + cookie support
# --------------------------------------------------------------------------------------
def hash_password(password: str, *, salt_bytes: int = 16, iters: int = 100_000) -> str:
    salt = os.urandom(salt_bytes)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
    return base64.b64encode(salt + dk).decode()

def verify_password(password: str, stored_hash_b64: str, iters: int = 100_000) -> bool:
    raw = base64.b64decode(stored_hash_b64.encode())
    salt, dk = raw[:16], raw[16:]
    test = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iters)
    return hmac.compare_digest(dk, test)

def _sign(payload: bytes) -> str:
    sig = hmac.new(SECRET_KEY.encode(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(sig).decode().rstrip("=")

def make_token(user_id: int, ttl_seconds: int = TOKEN_TTL_SECONDS) -> str:
    exp = int((datetime.utcnow() + timedelta(seconds=ttl_seconds)).timestamp())
    body = json.dumps({"uid": user_id, "exp": exp}).encode()
    body_b64 = base64.urlsafe_b64encode(body).decode().rstrip("=")
    sig = _sign(body_b64.encode())
    return f"{body_b64}.{sig}"

def verify_token(token: str) -> Optional[int]:
    try:
        body_b64, sig = token.split(".", 1)
        if not hmac.compare_digest(sig, _sign(body_b64.encode())):
            return None
        body = json.loads(base64.urlsafe_b64decode(body_b64 + "=="))
        if int(body["exp"]) < int(datetime.utcnow().timestamp()):
            return None
        return int(body["uid"])
    except Exception:
        return None

def get_session():
    with Session(engine) as session:
        yield session

def _get_user_by_token(token: str, session: Session) -> Optional[User]:
    uid = verify_token(token)
    if not uid:
        return None
    return session.get(User, uid)

def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(None),
    session: Session = Depends(get_session),
) -> User:
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization.split(" ", 1)[1].strip()
    elif "auth" in request.cookies:
        token = request.cookies.get("auth")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = _get_user_by_token(token, session)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user

# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------
class ShortenRequest(BaseModel):
    long_url: HttpUrl
    alias: Optional[str] = None
    expires_in_days: Optional[int] = None

class UrlOut(BaseModel):
    id: int
    long_url: str
    short_code: str
    short_url: str
    created_at: datetime
    clicks: int
    expires_at: Optional[datetime]

class SignupIn(BaseModel):
    username: str
    password: str

class LoginIn(BaseModel):
    username: str
    password: str

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class StatsOut(BaseModel):
    code: str
    total_clicks: int
    last_30_days: List[Dict[str, Any]]

# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/signup", response_class=HTMLResponse)
def signup_page(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@app.get("/logout")
def logout(response: Response):
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie("auth")
    return response

@app.get("/admin", response_class=HTMLResponse)
def admin_page(
    request: Request,
    session: Session = Depends(get_session),
    user: User = Depends(get_current_user),
    q: Optional[str] = None,
    page: int = 1,
    size: int = 20,
):
    page = max(1, page)
    size = min(100, max(5, size))

    stmt = select(Url).where(Url.owner_id == user.id)
    if q:
        like = f"%{q}%"
        stmt = stmt.where(
            (Url.short_code.like(like)) |
            (Url.long_url.like(like)) |
            (Url.tags.like(like))
        )

    all_rows = session.exec(stmt.order_by(Url.created_at.desc())).all()
    total = len(all_rows)
    start, end = (page - 1) * size, (page) * size
    rows = all_rows[start:end]

    return templates.TemplateResponse(
        "admin.html",
        {
            "request": request,
            "urls": rows,
            "me": user,
            "q": q or "",
            "page": page,
            "size": size,
            "total": total,
            "pages": max(1, (total + size - 1) // size),
        },
    )

# --------------------------------------------------------------------------------------
# Auth endpoints (API + form)
# --------------------------------------------------------------------------------------
@app.post("/auth/signup", response_model=TokenOut)
def auth_signup(payload: SignupIn, session: Session = Depends(get_session)):
    existing = session.exec(select(User).where(User.username == payload.username)).first()
    if existing:
        raise HTTPException(400, "Username already taken")
    u = User(username=payload.username, password_hash=hash_password(payload.password))
    session.add(u)
    session.commit()
    token = make_token(u.id)
    return TokenOut(access_token=token)

@app.post("/auth/login", response_model=TokenOut)
def auth_login(payload: LoginIn, session: Session = Depends(get_session)):
    u = session.exec(select(User).where(User.username == payload.username)).first()
    if not u or not verify_password(payload.password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    return TokenOut(access_token=make_token(u.id))

# Browser form handlers -> set cookie and redirect
@app.post("/auth/login_form")
def auth_login_form(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    u = session.exec(select(User).where(User.username == username)).first()
    if not u or not verify_password(password, u.password_hash):
        raise HTTPException(401, "Invalid credentials")
    token = make_token(u.id)
    resp = RedirectResponse(url="/admin", status_code=302)
    # HttpOnly=False so client JS can read (if needed); set True for stronger security in prod
    resp.set_cookie("auth", token, max_age=TOKEN_TTL_SECONDS, secure=False, httponly=True, samesite="lax")
    return resp

@app.post("/auth/signup_form")
def auth_signup_form(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
):
    existing = session.exec(select(User).where(User.username == username)).first()
    if existing:
        raise HTTPException(400, "Username already taken")
    u = User(username=username, password_hash=hash_password(password))
    session.add(u)
    session.commit()
    token = make_token(u.id)
    resp = RedirectResponse(url="/admin", status_code=302)
    resp.set_cookie("auth", token, max_age=TOKEN_TTL_SECONDS, secure=False, httponly=True, samesite="lax")
    return resp

# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
@app.post("/api/shorten", response_model=UrlOut, status_code=status.HTTP_201_CREATED)
def api_shorten(
    payload: ShortenRequest,
    request: Request,
    session: Session = Depends(get_session),
    user: Optional[User] = Depends(lambda request: None),  # optional auth; we’ll read cookie manually
):
    # Optional: read user via cookie to own the URL if logged in
    token = request.cookies.get("auth")
    if token:
        with Session(engine) as s2:
            user = _get_user_by_token(token, s2)

    long_url = str(payload.long_url).strip()
    alias = (payload.alias or "").strip() or None
    expires_at = None

    if not is_valid_url(long_url):
        raise HTTPException(400, "Invalid URL. Must start with http(s)://")

    if alias and not is_valid_alias(alias):
        raise HTTPException(400, "Invalid alias. Use 4-16 alphanumeric characters.")

    if payload.expires_in_days:
        try:
            d = int(payload.expires_in_days)
            if 0 < d <= 365:
                expires_at = datetime.utcnow() + timedelta(days=d)
        except Exception:
            pass

    # Idempotent (per-owner) if no alias/expiry
    if not alias and not expires_at:
        q = select(Url).where(Url.long_url == long_url, Url.expires_at.is_(None))
        if user:
            q = q.where(Url.owner_id == user.id)
        existing = session.exec(q).first()
        if existing:
            return UrlOut(
                id=existing.id,
                long_url=existing.long_url,
                short_code=existing.short_code,
                short_url=full_short(existing.short_code, request),
                created_at=existing.created_at,
                clicks=existing.clicks,
                expires_at=existing.expires_at,
            )

    # Alias path
    if alias:
        clash = session.exec(select(Url).where(Url.short_code == alias)).first()
        if clash and clash.long_url.lower() != long_url.lower():
            raise HTTPException(400, "Alias already in use")
        if clash:
            if expires_at:
                if user and clash.owner_id and clash.owner_id != user.id:
                    raise HTTPException(403, "Not your link")
                clash.expires_at = expires_at
                session.add(clash)
                session.commit()
            return UrlOut(
                id=clash.id,
                long_url=clash.long_url,
                short_code=clash.short_code,
                short_url=full_short(clash.short_code, request),
                created_at=clash.created_at,
                clicks=clash.clicks,
                expires_at=clash.expires_at,
            )

        rec = Url(
            long_url=long_url,
            short_code=alias,
            expires_at=expires_at,
            owner_id=(user.id if user else None),
        )
        session.add(rec)
        session.commit()
        jlog("shorten", code=rec.short_code, alias=True, owner_id=rec.owner_id)
        return UrlOut(
            id=rec.id,
            long_url=rec.long_url,
            short_code=rec.short_code,
            short_url=full_short(rec.short_code, request),
            created_at=rec.created_at,
            clicks=rec.clicks,
            expires_at=rec.expires_at,
        )

    # Auto-generate code
    rec = Url(
        long_url=long_url,
        short_code="tmp",
        expires_at=expires_at,
        owner_id=(user.id if user else None),
    )
    session.add(rec)
    session.flush()
    code = base62_encode(rec.id)
    if len(code) < CODE_LENGTH_MIN:
        code = code.rjust(CODE_LENGTH_MIN, "0")
    if len(code) > CODE_LENGTH_MAX:
        code = code[:CODE_LENGTH_MAX]
    rec.short_code = code
    session.add(rec)
    session.commit()
    jlog("shorten", code=rec.short_code, alias=False, owner_id=rec.owner_id)

    return UrlOut(
        id=rec.id,
        long_url=rec.long_url,
        short_code=rec.short_code,
        short_url=full_short(rec.short_code, request),
        created_at=rec.created_at,
        clicks=rec.clicks,
        expires_at=rec.expires_at,
    )

@app.get("/{code}")
async def resolve(
    code: str,
    request: Request,
    session: Session = Depends(get_session),
):
    if not code:
        raise HTTPException(404)

    # Cache first
    cached = await cache_get(code)
    if cached:
        url = session.exec(select(Url).where(Url.short_code == code)).first()
        if not url:
            await cache_del(code)
            raise HTTPException(404)
        if url.expires_at and datetime.utcnow() >= url.expires_at:
            await cache_del(code)
            raise HTTPException(status_code=410, detail="Link expired")

        _log_click(session, url, request)
        url.clicks += 1
        session.add(url)
        session.commit()

        jlog("redirect", code=code, cached=True)
        return RedirectResponse(cached, status_code=302)

    url = session.exec(select(Url).where(Url.short_code == code)).first()
    if not url:
        raise HTTPException(404)
    if url.expires_at and datetime.utcnow() >= url.expires_at:
        raise HTTPException(status_code=410, detail="Link expired")

    _log_click(session, url, request)
    url.clicks += 1
    session.add(url)
    session.commit()

    ttl = remaining_ttl(url.expires_at)
    await cache_set(code, url.long_url, ttl)

    jlog("redirect", code=code, cached=False)
    return RedirectResponse(url.long_url, status_code=302)

def _log_click(session: Session, url: Url, request: Request) -> None:
    try:
        ref = request.headers.get("referer")
        ua = request.headers.get("user-agent")
        ip = (request.client.host if request.client else None)
        session.add(ClickLog(url_id=url.id, referrer=ref, user_agent=ua, ip=ip))
        # no separate commit
    except Exception as e:
        jlog("clicklog_error", error=str(e))

@app.get("/qrcode/{code}.png")
def qrcode_png(code: str, request: Request, session: Session = Depends(get_session)):
    if qrcode is None:
        raise HTTPException(500, "qrcode package not installed")

    url = session.exec(select(Url).where(Url.short_code == code)).first()
    if not url:
        raise HTTPException(404)
    if url.expires_at and datetime.utcnow() >= url.expires_at:
        raise HTTPException(status_code=410, detail="Link expired")

    img = qrcode.make(full_short(code, request))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="{code}.png"'}
    )

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat() + "Z"}

# Public stats API
@app.get("/api/stats/{code}", response_model=StatsOut)
def stats(code: str, session: Session = Depends(get_session)):
    url = session.exec(select(Url).where(Url.short_code == code)).first()
    if not url:
        raise HTTPException(404, "Unknown short code")

    since = datetime.utcnow() - timedelta(days=29)
    logs = session.exec(
        select(ClickLog).where(ClickLog.url_id == url.id, ClickLog.ts >= since)
    ).all()

    counts: Dict[str, int] = {}
    for i in range(30):
        day = (since + timedelta(days=i)).date().isoformat()
        counts[day] = 0
    for cl in logs:
        d = cl.ts.date().isoformat()
        if d in counts:
            counts[d] += 1

    series = [{"date": d, "count": counts[d]} for d in sorted(counts.keys())]
    return StatsOut(code=code, total_clicks=url.clicks, last_30_days=series)

# HTML form shorten (kept)
@app.post("/shorten", response_class=HTMLResponse)
def shorten_form(
    request: Request,
    long_url: str = Form(...),
    alias: Optional[str] = Form(None),
    expiry_days: Optional[int] = Form(None),
    session: Session = Depends(get_session),
):
    expires_in_days = expiry_days if expiry_days and expiry_days > 0 else None
    payload = ShortenRequest(long_url=long_url, alias=alias or None, expires_in_days=expires_in_days)
    result = api_shorten(payload, request, session)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "short_url": result.short_url,
            "long_url": result.long_url,
            "code": result.short_code,
        },
    )

# --------------------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    init_db()

"""
Run locally:
    uvicorn main:app --reload

Production example:
    gunicorn -k uvicorn.workers.UvicornWorker -w 2 main:app

Env Vars:
    SECRET_KEY=change_me
    TOKEN_TTL_SECONDS=86400
    DATABASE_URL=...
    REDIS_URL=...
    BASE_URL=https://sho.rt
"""
