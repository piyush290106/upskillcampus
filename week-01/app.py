
import os
import re
import string
from datetime import datetime
from urllib.parse import urlparse

from flask import Flask, request, redirect, render_template_string, url_for, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func


DB_URL = os.getenv("DATABASE_URL", "sqlite:///shortener.db")
BASE_URL = os.getenv("BASE_URL")  # e.g., https://sho.rt (optional, used in responses)
CODE_LENGTH_MIN = int(os.getenv("CODE_LENGTH_MIN", 4))
CODE_LENGTH_MAX = int(os.getenv("CODE_LENGTH_MAX", 10))

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = DB_URL
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


BASE62_ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase
BASE = len(BASE62_ALPHABET)

def base62_encode(num: int) -> str:
    if num == 0:
        return BASE62_ALPHABET[0]
    s = []
    n = num
    while n > 0:
        n, r = divmod(n, BASE)
        s.append(BASE62_ALPHABET[r])
    return "".join(reversed(s))


    __tablename__ = "urls"

    id = db.Column(db.Integer, primary_key=True)
    long_url = db.Column(db.Text, nullable=False)
    short_code = db.Column(db.String(16), unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    clicks = db.Column(db.Integer, default=0, nullable=False)

    def to_dict(self):
        short = full_short_url(self.short_code)
        return {
            "id": self.id,
            "long_url": self.long_url,
            "short_code": self.short_code,
            "short_url": short,
            "created_at": self.created_at.isoformat() + "Z",
            "clicks": self.clicks,
        }


ALLOWED_SCHEMES = {"http", "https"}

URL_REGEX = re.compile(r"^https?://.+", re.IGNORECASE)

def is_valid_url(u: str) -> bool:
    if not u:
        return False
    if not URL_REGEX.match(u):
        return False
    p = urlparse(u)
    return p.scheme in ALLOWED_SCHEMES and bool(p.netloc)


def full_short_url(code: str) -> str:
    if BASE_URL:
        return f"{BASE_URL.rstrip('/')}/{code}"
    # Fallback to building from request if available
    try:
        return url_for("resolve", code=code, _external=True)
    except RuntimeError:
        # outside request context
        return f"/{code}"


INDEX_HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>URL Shortener</title>
    <style>
      body{font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, sans-serif; margin: 40px;}
      .card{max-width: 720px; margin: 0 auto; padding: 24px; border: 1px solid #e5e7eb; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06)}
      h1{margin-top:0}
      input[type=url]{width:100%; padding:12px; font-size:16px; border:1px solid #d1d5db; border-radius:8px}
      button{margin-top:12px; padding:10px 14px; font-size:15px; border-radius:8px; border:0; background:#111827; color:white; cursor:pointer}
      .result{margin-top:18px; padding:12px; background:#f9fafb; border:1px dashed #e5e7eb; border-radius:8px}
      code{padding:2px 6px; background:#111827; color:white; border-radius:6px}
      .stats{margin-top:12px; color:#6b7280}
    </style>
  </head>
  <body>
    <div class="card">
      <h1>🔗 TinyLink</h1>
      <p>Paste a long URL and get a tidy short link.</p>
      <form method="post" action="/shorten">
        <input type="url" name="long_url" placeholder="https://example.com/very/long/path?with=params" required>
        <button type="submit">Shorten</button>
      </form>
      {% if short_url %}
      <div class="result">
        <div>Short URL: <a href="{{ short_url }}">{{ short_url }}</a></div>
        <div class="stats">Original: {{ long_url }}</div>
      </div>
      {% endif %}
      <p class="stats">Try the JSON API: <code>POST /api/shorten {"long_url":"https://..."}</code></p>
    </div>
  </body>
</html>
"""


@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML)


@app.route("/shorten", methods=["POST"])
def shorten_form():
    long_url = request.form.get("long_url", "").strip()
    if not is_valid_url(long_url):
        return render_template_string(
            INDEX_HTML + "<p style='color:#b91c1c'>Invalid URL. Must start with http(s) and include a host.</p>"
        )
    url = get_or_create_short(long_url)
    return render_template_string(
        INDEX_HTML, short_url=full_short_url(url.short_code), long_url=url.long_url
    )


@app.route("/api/shorten", methods=["POST"])
def api_shorten():
    payload = request.get_json(silent=True) or {}
    long_url = (payload.get("long_url") or "").strip()
    if not is_valid_url(long_url):
        return jsonify({"error": "Invalid URL. Provide http(s)://host..."}), 400
    url = get_or_create_short(long_url)
    return jsonify(url.to_dict()), 201

class Url(db.Model):
    __tablename__ = "urls"
    id = db.Column(db.Integer, primary_key=True)
    long_url = db.Column(db.String(512), nullable=False)
    short_code = db.Column(db.String(16), unique=True, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    clicks = db.Column(db.Integer, default=0)


@app.route("/<code>")
def resolve(code: str):
    code = (code or "").strip()
    if not code:
        abort(404)
    url = Url.query.filter_by(short_code=code).first()
    if not url:
        abort(404)
    Url.query.filter_by(id=url.id).update({Url.clicks: Url.clicks + 1})
    db.session.commit()
    return redirect(url.long_url, code=302)


def get_or_create_short(long_url: str) -> Url:
    existing = Url.query.filter(func.lower(Url.long_url) == long_url.lower()).first()
    if existing:
        return existing

    url = Url(long_url=long_url)
    db.session.add(url)
    db.session.flush()  

    code = base62_encode(url.id)
    if len(code) < CODE_LENGTH_MIN:
        code = code.rjust(CODE_LENGTH_MIN, BASE62_ALPHABET[0])

    url.short_code = code

    try:
        db.session.commit()
    except IntegrityError:
        db.session.rollback()
        for i in range(1, 5):
            alt = f"{code}{BASE62_ALPHABET[i]}"
            url.short_code = alt
            try:
                db.session.commit()
                break
            except IntegrityError:
                db.session.rollback()
        else:
            raise

    return url


@app.cli.command("init-db")
def init_db():
    """Initialize the database tables."""
    db.create_all()
    print("Initialized the database.")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
