import os
import io
import re
import json
import string
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Depends, status, Form
from fastapi.responses import RedirectResponse, JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sqlmodel import SQLModel, Field, create_engine, Session, select
from pydantic import BaseModel, HttpUrl

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
BASE_URL = os.getenv("BASE_URL")  # e.g. https://sho.rt (optional)
CODE_LENGTH_MIN = int(os.getenv("CODE_LENGTH_MIN", 4))
CODE_LENGTH_MAX = int(os.getenv("CODE_LENGTH_MAX", 10))
REDIS_URL = os.getenv("REDIS_URL")  # e.g. redis://localhost:6379/0


# --------------------------------------------------------------------------------------
# App (create first, then mount)
# --------------------------------------------------------------------------------------
app = FastAPI(title="TinyLink API", version="2.0")

# Robust paths for static/templates
BASE_DIR = Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# Auto-create if missing so Starlette doesn't crash
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static only if exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --------------------------------------------------------------------------------------
# DB setup
# --------------------------------------------------------------------------------------
engine = create_engine(DATABASE_URL, echo=False, future=True)

class Url(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    long_url: str
    short_code: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    clicks: int = Field(default=0)
    expires_at: Optional[datetime] = None

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

def base62_encode(n: int) -> str:
    if n == 0:
        return BASE62[0]
    s = []
    while n > 0:
        n, r = divmod(n, 62)
        s.append(BASE62[r])
    return "".join(reversed(s))

URL_REGEX = re.compile(r"^https?://", re.IGNORECASE)
ALIAS_REGEX = re.compile(r"^[a-zA-Z0-9]{4,16}$")

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
    return f"/{code}"

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


# --------------------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------------------
def get_session():
    with Session(engine) as session:
        yield session


# --------------------------------------------------------------------------------------
# Pages (Form + Admin)
# --------------------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, session: Session = Depends(get_session)):
    urls = session.exec(select(Url).order_by(Url.clicks.desc()).limit(50)).all()
    return templates.TemplateResponse("admin.html", {"request": request, "urls": urls})


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
@app.post("/api/shorten", response_model=UrlOut, status_code=status.HTTP_201_CREATED)
def api_shorten(payload: ShortenRequest, request: Request, session: Session = Depends(get_session)):
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

    # If no alias/expiry requested, return existing if present (idempotent)
    if not alias and not expires_at:
        existing = session.exec(
            select(Url).where(Url.long_url == long_url, Url.expires_at.is_(None))
        ).first()
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
            if expires_at:  # update expiry if provided
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
        rec = Url(long_url=long_url, short_code=alias, expires_at=expires_at)
        session.add(rec)
        session.commit()
        jlog("shorten", code=rec.short_code, alias=True)
        return UrlOut(
            id=rec.id,
            long_url=rec.long_url,
            short_code=rec.short_code,
            short_url=full_short(rec.short_code, request),
            created_at=rec.created_at,
            clicks=rec.clicks,
            expires_at=rec.expires_at,
        )

    # Auto-generate code from ID
    rec = Url(long_url=long_url, short_code="tmp", expires_at=expires_at)
    session.add(rec)
    session.flush()  # get ID
    code = base62_encode(rec.id)

    # fit to bounds
    if len(code) < CODE_LENGTH_MIN:
        code = code.rjust(CODE_LENGTH_MIN, "0")
    if len(code) > CODE_LENGTH_MAX:
        code = code[:CODE_LENGTH_MAX]

    rec.short_code = code
    session.add(rec)
    session.commit()
    jlog("shorten", code=rec.short_code, alias=False)

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
async def resolve(code: str, session: Session = Depends(get_session)):
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

    url.clicks += 1
    session.add(url)
    session.commit()

    # Cache with TTL if expiry exists
    ttl = remaining_ttl(url.expires_at)
    await cache_set(code, url.long_url, ttl)

    jlog("redirect", code=code, cached=False)
    return RedirectResponse(url.long_url, status_code=302)


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


# --------------------------------------------------------------------------------------
# Simple HTML form support (POST)
# --------------------------------------------------------------------------------------
@app.post("/shorten", response_class=HTMLResponse)
def shorten_form(
    request: Request,
    long_url: str = Form(...),
    alias: Optional[str] = Form(None),
    expiry_days: Optional[int] = Form(None),
    session: Session = Depends(get_session),
):
    # reuse API logic by constructing payload
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
