"""
InflectionAI Signal Hunter — Backend Server
============================================
FastAPI backend with:
  - Signal generation (yfinance + EMA strategy)
  - Stripe subscription payments (card / Apple Pay / Google Pay / Alipay)
  - PayPal subscription payments
  - NOWPayments USDT/crypto payments
  - SQLite user/subscription database
  - API key auth for protected routes

Env vars (set in Render dashboard or .env):
  ── Stripe (card / Apple Pay / Google Pay / Alipay) ──
  STRIPE_SECRET_KEY       sk_live_...  (or sk_test_...)
  STRIPE_WEBHOOK_SECRET   whsec_...
  STRIPE_PRICE_WEEKLY     price_...
  STRIPE_PRICE_MONTHLY    price_...
  STRIPE_PRICE_QUARTERLY  price_...
  STRIPE_PRICE_YEARLY     price_...

  ── PayPal ──────────────────────────────────────────
  PAYPAL_CLIENT_ID        (from developer.paypal.com)
  PAYPAL_CLIENT_SECRET    (from developer.paypal.com)
  PAYPAL_MODE             live | sandbox  (default: sandbox)

  ── NOWPayments (USDT / crypto) ──────────────────────
  NOWPAYMENTS_API_KEY     (from app.nowpayments.io)
  NOWPAYMENTS_IPN_SECRET  (IPN secret for webhook verification)

  ── General ─────────────────────────────────────────
  FRONTEND_URL            https://your-domain.com
  ADMIN_KEY               (random secret for admin endpoints)

Run locally:
  pip install -r requirements.txt
  python server.py
"""

import os, sys, json, sqlite3, secrets, hashlib, time, hmac, threading
from datetime import datetime, timezone, timedelta
from typing import Optional
from pathlib import Path
from contextlib import contextmanager
import urllib.request, urllib.parse

# ── In-memory data cache (TTL-based, zero extra dependencies) ────
_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()
# TTL seconds per timeframe — daily data valid 6 h, hourly 1 h
_CACHE_TTL = {"1h": 3600, "4h": 7200, "1d": 21600, "1w": 43200, "1M": 86400}

def _ck(sym, tf, limit):          return f"{sym}|{tf}|{limit}"
def _cache_get(key, tf):
    with _CACHE_LOCK:
        e = _CACHE.get(key)
        if e and time.time() - e["ts"] < _CACHE_TTL.get(tf, 3600):
            return e["payload"]
    return None
def _cache_set(key, payload):
    with _CACHE_LOCK:
        _CACHE[key] = {"payload": payload, "ts": time.time()}

# Symbols pre-warmed at startup (most commonly searched)
_WARMUP = ["AAPL","TSLA","NVDA","SPY","MSFT","AMZN","META","GOOGL"]

# ── FastAPI ──────────────────────────────────────────────────────
from fastapi import FastAPI, Query, HTTPException, Request, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import uvicorn

# ── Optional: load .env in local dev ────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Stripe ───────────────────────────────────────────────────────
try:
    import stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_OK = bool(stripe.api_key)
except ImportError:
    STRIPE_OK = False

# ── Signal engine (reuse inflection_hunter) ──────────────────────
SCRIPT_DIR = Path(__file__).parent
for p in [SCRIPT_DIR.parent / "inflection_hunter", SCRIPT_DIR.parent]:
    if p.is_dir():
        sys.path.insert(0, str(p))
        break

try:
    from data_provider import YFinanceProvider
    from strategy import EMACrossoverStrategy
    _ENGINE = "inflection_hunter"
except ImportError:
    # Inline fallback
    import pandas as pd, numpy as np
    _ENGINE = "inline"

    class YFinanceProvider:
        # yfinance interval → yf_interval
        # 1M: fetch weekly then resample to monthly (more reliable than 1mo API)
        # 1Y: fetch daily data, limit to ~252 bars = 1 trading year
        _TF = {
            "1m":"1m",  "5m":"5m",  "15m":"15m", "30m":"30m",
            "1h":"1h",  "4h":"1h",               # 4h → resample from 1h
            "1d":"1d",  "1w":"1wk",
            "1M":"1wk",                           # monthly → resample from weekly
            "1Y":"1d",                            # 1-year view → daily bars
        }
        _PD = {
            "1m":"7d",  "5m":"60d", "15m":"60d", "30m":"60d",
            "1h":"max", "4h":"max",
            "1d":"max", "1w":"max", "1M":"max",  "1Y":"max",
        }
        def get_ohlcv(self, symbol, timeframe="1d", start=None, end=None, limit=1260):
            import yfinance as yf
            import pandas as pd
            resample_4h = (timeframe == "4h")
            resample_1M = (timeframe == "1M")
            yf_tf  = self._TF.get(timeframe, "1d")
            period = self._PD.get(timeframe, "max")
            df = yf.Ticker(symbol).history(period=period, interval=yf_tf,
                                           auto_adjust=True)
            if df.empty: raise ValueError(f"No data for {symbol}")
            df.columns = [c.lower() for c in df.columns]
            if "volume" not in df.columns: df["volume"] = 0
            df = df[["open","high","low","close","volume"]].dropna()
            if df.index.tzinfo:
                try:    df.index = df.index.tz_localize(None)
                except: df.index = df.index.tz_convert(None)
            df = df.sort_index()
            if resample_4h:
                df = df.resample("4h").agg(
                    {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                ).dropna()
            if resample_1M:
                # resample weekly→monthly (pandas "ME" ≥2.2, fallback "M")
                rule = "ME"
                try:
                    df = df.resample(rule).agg(
                        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                    ).dropna()
                except Exception:
                    df = df.resample("M").agg(
                        {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
                    ).dropna()
            return df.tail(limit) if limit else df

    class EMACrossoverStrategy:
        def __init__(self, fast=8, slow=20, confirm_window=2):
            self.fast=fast; self.slow=slow; self.confirm_window=confirm_window
        def generate_signals(self, df):
            from dataclasses import dataclass
            @dataclass
            class Sig:
                time: object; side: str; price: float
                reason: str=""; regime: str=""; strength: float=0.9
            ema_f = df["close"].ewm(span=self.fast,adjust=False).mean()
            ema_s = df["close"].ewm(span=self.slow,adjust=False).mean()
            above = ema_f > ema_s
            prev  = above.shift(1).fillna(above)
            cu = above & ~prev.astype(bool)
            cd = ~above & prev.astype(bool)
            sigs=[]; peb=pes=None; last_dir=None
            for i in range(len(df)):
                ts=df.index[i]; px=float(df["close"].iloc[i])
                if cu.iloc[i] and last_dir != 'UP':
                    sigs.append(Sig(ts,"eB",px,"ema_fast_cross_up","UP",0.55))
                    peb=i; pes=None; last_dir='UP'
                elif cd.iloc[i] and last_dir != 'DOWN':
                    sigs.append(Sig(ts,"eS",px,"ema_fast_cross_down","DOWN",0.55))
                    pes=i; peb=None; last_dir='DOWN'
                if peb is not None and (i-peb)==self.confirm_window:
                    sigs.append(Sig(ts,"B",px,"confirm_window_2","UP",0.9)); peb=None
                if pes is not None and (i-pes)==self.confirm_window:
                    sigs.append(Sig(ts,"S",px,"confirm_window_2","DOWN",0.9)); pes=None
            return sigs


# ═══════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════
DB_PATH = SCRIPT_DIR / "inflection.db"

def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT    UNIQUE NOT NULL,
            api_key     TEXT    UNIQUE,
            created_at  TEXT    DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS subscriptions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email          TEXT    NOT NULL,
            stripe_customer_id  TEXT,
            stripe_sub_id       TEXT,
            stripe_session_id   TEXT,
            plan                TEXT,
            status              TEXT    DEFAULT 'pending',
            payment_method      TEXT    DEFAULT 'stripe',
            tx_hash             TEXT,
            current_period_end  TEXT,
            created_at          TEXT    DEFAULT (datetime('now')),
            updated_at          TEXT    DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_users_email  ON users(email);
        CREATE INDEX IF NOT EXISTS idx_subs_email   ON subscriptions(user_email);
        CREATE INDEX IF NOT EXISTS idx_subs_session ON subscriptions(stripe_session_id);
        """)
        # Add new columns to existing DB (migration — safe to run repeatedly)
        for col, defval in [("payment_method","'stripe'"), ("tx_hash","NULL")]:
            try:
                db.execute(f"ALTER TABLE subscriptions ADD COLUMN {col} TEXT DEFAULT {defval}")
            except Exception:
                pass  # column already exists
        db.commit()

def upsert_user(email: str) -> str:
    """Create user if not exists, return their api_key."""
    with get_db() as db:
        row = db.execute("SELECT api_key FROM users WHERE email=?", (email,)).fetchone()
        if row:
            return row["api_key"]
        key = "iai_" + secrets.token_urlsafe(32)
        db.execute("INSERT INTO users (email, api_key) VALUES (?,?)", (email, key))
        db.commit()
        return key

def get_sub_status(email: str) -> dict:
    with get_db() as db:
        row = db.execute(
            """SELECT status, plan, current_period_end FROM subscriptions
               WHERE user_email=? AND status='active'
               ORDER BY created_at DESC LIMIT 1""",
            (email,)
        ).fetchone()
        if row:
            return {"active": True, "plan": row["plan"], "expires": row["current_period_end"]}
        return {"active": False, "plan": None, "expires": None}

def verify_api_key(api_key: str) -> Optional[str]:
    """Returns email if valid key, else None."""
    with get_db() as db:
        row = db.execute("SELECT email FROM users WHERE api_key=?", (api_key,)).fetchone()
        return row["email"] if row else None


# ═══════════════════════════════════════════════════════════════
# APP
# ═══════════════════════════════════════════════════════════════
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8080")

app = FastAPI(
    title="InflectionAI API",
    description="Signal generation + subscription management",
    version="1.0.0",
    docs_url="/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten to your domain in production
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

def _build_snapshot_payload(sym, tf, limit, fast=8, slow=20, aux=50, confirm=2):
    """Fetch data + compute signals → return the full response dict (cacheable)."""
    prov = YFinanceProvider()
    df   = prov.get_ohlcv(sym, tf, limit=limit)
    strat = EMACrossoverStrategy(fast=fast, slow=slow, confirm_window=confirm)
    sigs  = strat.generate_signals(df)

    def to_unix(ts):
        try:    return int(ts.timestamp())
        except: return int(ts.value // 1_000_000_000)

    def ema_s(period):
        s = df["close"].ewm(span=period, adjust=False).mean()
        return [{"time": to_unix(t), "value": round(float(v), 6)} for t, v in s.items()]

    return {
        "bars": [
            {"time": to_unix(t), "open": round(float(r.open),6),
             "high": round(float(r.high),6), "low": round(float(r.low),6),
             "close": round(float(r.close),6), "volume": int(r.volume)}
            for t, r in df.iterrows()
        ],
        "signals": [
            {"time": to_unix(s.time),
             "side": s.side if isinstance(s.side, str) else s.side.value,
             "price": round(float(s.price), 6),
             "reason": getattr(s, "reason", ""),
             "regime": getattr(s, "regime", ""),
             "strength": round(float(getattr(s, "strength", 0.9)), 4)}
            for s in sigs
        ],
        "ema_series": ema_s(slow),
        "aux_series": ema_s(aux),
        "meta": {
            "symbol": sym, "tf": tf,
            "signal_fast": fast, "signal_slow": slow,
            "ema_period": slow, "aux_period": aux,
            "engine": _ENGINE,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "cached": False,
        },
    }


def _warmup():
    """Pre-load popular symbols into cache in background (non-blocking)."""
    time.sleep(3)  # wait for server to fully start
    for sym in _WARMUP:
        try:
            key = _ck(sym, "1d", 600)
            if _cache_get(key, "1d") is None:
                payload = _build_snapshot_payload(sym, "1d", 600)
                payload["meta"]["cached"] = True
                _cache_set(key, payload)
                print(f"  [warmup] ✓ {sym}")
        except Exception as ex:
            print(f"  [warmup] ✗ {sym}: {ex}")


@app.on_event("startup")
def startup():
    init_db()
    _paypal_ok = bool(os.getenv("PAYPAL_CLIENT_ID") and os.getenv("PAYPAL_CLIENT_SECRET"))
    _usdt_ok   = bool(os.getenv("USDT_WALLET_ADDRESS"))
    print(f"  DB: {DB_PATH}")
    print(f"  Stripe:      {'✓' if STRIPE_OK else '✗ (set STRIPE_SECRET_KEY)'}")
    print(f"  PayPal:      {'✓' if _paypal_ok else '✗ (set PAYPAL_CLIENT_ID + PAYPAL_CLIENT_SECRET)'}")
    print(f"  USDT wallet: {'✓' if _usdt_ok else '✗ (set USDT_WALLET_ADDRESS)'}")
    print(f"  Engine: {_ENGINE}")
    # Pre-warm cache in background so first users get instant responses
    threading.Thread(target=_warmup, daemon=True).start()


# ── Auth dependency ──────────────────────────────────────────────
def require_auth(x_api_key: Optional[str] = Header(None)) -> str:
    if not x_api_key:
        raise HTTPException(401, "Missing X-API-Key header")
    email = verify_api_key(x_api_key)
    if not email:
        raise HTTPException(401, "Invalid API key")
    sub = get_sub_status(email)
    if not sub["active"]:
        raise HTTPException(403, "No active subscription. Visit the website to subscribe.")
    return email


# ═══════════════════════════════════════════════════════════════
# PLAN CONFIG
# ═══════════════════════════════════════════════════════════════
PLANS = {
    "weekly":    {"name": "Weekly",    "price_usd": 490,   "interval": "week"},
    "monthly":   {"name": "Monthly",   "price_usd": 1990,  "interval": "month"},
    "quarterly": {"name": "Quarterly", "price_usd": 4990,  "interval": "quarter"},
    "yearly":    {"name": "Yearly",    "price_usd": 18900, "interval": "year"},
}

def get_stripe_price(plan: str) -> str:
    """Get Stripe Price ID from env vars."""
    key = f"STRIPE_PRICE_{plan.upper()}"
    pid = os.getenv(key, "")
    if not pid:
        raise HTTPException(500, f"Stripe price not configured for plan '{plan}'. "
                                 f"Set {key} in environment variables.")
    return pid


# ═══════════════════════════════════════════════════════════════
# PUBLIC ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/health")
@app.get("/health")   # short alias for keep-alive pings
def health():
    with _CACHE_LOCK:
        cache_size = len(_CACHE)
    return {
        "status": "ok",
        "engine": _ENGINE,
        "stripe": STRIPE_OK,
        "cache_entries": cache_size,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/plans")
def plans():
    return {"plans": PLANS}


# ── Signal snapshot — with TTL cache (daily=6h, hourly=1h) ───────
@app.get("/api/snapshot")
def snapshot(
    symbol:  str  = Query("TSLA"),
    tf:      str  = Query("1d"),
    limit:   int  = Query(600),
    fast:    int  = Query(8),
    slow:    int  = Query(20),
    aux:     int  = Query(50),
    confirm: int  = Query(2),
    preview: bool = Query(False),
    x_api_key: Optional[str] = Header(None),
):
    sym = symbol.strip().upper()

    # Only cache standard params to keep cache simple
    use_cache = (fast == 8 and slow == 20 and aux == 50 and confirm == 2)
    key = _ck(sym, tf, limit)

    if use_cache:
        cached = _cache_get(key, tf)
        if cached is not None:
            return cached          # ← instant return, zero yfinance calls

    try:
        payload = _build_snapshot_payload(sym, tf, limit, fast, slow, aux, confirm)
    except Exception as e:
        raise HTTPException(422, f"Data error: {e}")

    payload["meta"]["bars_count"]    = len(payload["bars"])
    payload["meta"]["signals_count"] = len(payload["signals"])
    payload["meta"]["cached"]        = False

    if use_cache:
        _cache_set(key, payload)

    return payload


# ═══════════════════════════════════════════════════════════════
# STRIPE — CHECKOUT
# ═══════════════════════════════════════════════════════════════

@app.post("/api/subscribe/create-checkout")
async def create_checkout(request: Request):
    """
    Create a Stripe Checkout session.
    Body: { "email": "user@email.com", "plan": "monthly" }
    Returns: { "url": "https://checkout.stripe.com/..." }
    """
    if not STRIPE_OK:
        raise HTTPException(503, "Payment system not configured. "
                                 "Set STRIPE_SECRET_KEY environment variable.")
    body = await request.json()
    email = body.get("email", "").strip().lower()
    plan  = body.get("plan", "monthly").lower()

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email required.")
    if plan not in PLANS:
        raise HTTPException(400, f"Unknown plan: {plan}. Valid: {list(PLANS.keys())}")

    price_id = get_stripe_price(plan)

    # Create or retrieve Stripe customer
    customers = stripe.Customer.list(email=email, limit=1)
    if customers.data:
        customer_id = customers.data[0].id
    else:
        cust = stripe.Customer.create(email=email)
        customer_id = cust.id

    # Create upsert user locally
    upsert_user(email)

    # Create checkout session
    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{"price": price_id, "quantity": 1}],
        mode="subscription",
        success_url=f"{FRONTEND_URL}/success.html?session_id={{CHECKOUT_SESSION_ID}}",
        cancel_url=f"{FRONTEND_URL}/?canceled=1",
        metadata={"plan": plan, "email": email},
    )

    # Record pending subscription
    with get_db() as db:
        db.execute(
            """INSERT INTO subscriptions
               (user_email, stripe_customer_id, stripe_session_id, plan, status)
               VALUES (?,?,?,?,'pending')""",
            (email, customer_id, session.id, plan)
        )
        db.commit()

    return {"url": session.url, "session_id": session.id}


# ── Stripe Webhook ───────────────────────────────────────────────
@app.post("/api/stripe/webhook")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig     = request.headers.get("stripe-signature", "")
    secret  = os.getenv("STRIPE_WEBHOOK_SECRET", "")

    if not secret:
        raise HTTPException(500, "STRIPE_WEBHOOK_SECRET not configured")

    try:
        event = stripe.Webhook.construct_event(payload, sig, secret)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(400, "Invalid webhook signature")

    etype = event["type"]
    data  = event["data"]["object"]

    if etype == "checkout.session.completed":
        session_id  = data["id"]
        sub_id      = data.get("subscription")
        customer_id = data.get("customer")
        email       = data.get("customer_details", {}).get("email") or \
                      data.get("metadata", {}).get("email", "")

        # Retrieve subscription end date
        period_end = None
        if sub_id:
            sub = stripe.Subscription.retrieve(sub_id)
            period_end = datetime.fromtimestamp(
                sub.current_period_end, tz=timezone.utc
            ).isoformat()

        with get_db() as db:
            db.execute(
                """UPDATE subscriptions
                   SET status='active', stripe_sub_id=?, current_period_end=?,
                       updated_at=datetime('now')
                   WHERE stripe_session_id=?""",
                (sub_id, period_end, session_id)
            )
            # Ensure user exists
            existing = db.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone()
            if not existing and email:
                key = "iai_" + secrets.token_urlsafe(32)
                db.execute("INSERT OR IGNORE INTO users (email, api_key) VALUES (?,?)",
                           (email, key))
            db.commit()

    elif etype in ("customer.subscription.deleted", "customer.subscription.updated"):
        sub_id = data["id"]
        status = "cancelled" if etype == "customer.subscription.deleted" else \
                 ("active" if data.get("status") == "active" else "expired")
        period_end = None
        if data.get("current_period_end"):
            period_end = datetime.fromtimestamp(
                data["current_period_end"], tz=timezone.utc
            ).isoformat()
        with get_db() as db:
            db.execute(
                """UPDATE subscriptions
                   SET status=?, current_period_end=?, updated_at=datetime('now')
                   WHERE stripe_sub_id=?""",
                (status, period_end, sub_id)
            )
            db.commit()

    return {"received": True, "type": etype}


# ═══════════════════════════════════════════════════════════════
# USER ROUTES
# ═══════════════════════════════════════════════════════════════

@app.get("/api/user/status")
def user_status(email: str = Query(...)):
    """Check subscription status and return API key for active subscribers."""
    email = email.strip().lower()
    sub   = get_sub_status(email)
    if not sub["active"]:
        return {"active": False, "message": "No active subscription found."}

    api_key = upsert_user(email)
    return {
        "active":  True,
        "plan":    sub["plan"],
        "expires": sub["expires"],
        "api_key": api_key,
        "message": "Subscription active.",
    }

@app.get("/api/user/verify")
def verify_user(x_api_key: str = Header(...)):
    """Verify an API key and return subscription info."""
    email = verify_api_key(x_api_key)
    if not email:
        raise HTTPException(401, "Invalid API key")
    sub = get_sub_status(email)
    return {"email": email, "subscription": sub}


# ═══════════════════════════════════════════════════════════════
# PROTECTED ROUTES (require active subscription)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/signals/full")
def full_signals(
    symbol:    str   = Query("TSLA"),
    tf:        str   = Query("1d"),
    limit:     int   = Query(600),
    email:     str   = Depends(require_auth),
):
    """Full signal data — requires active subscription."""
    return snapshot(symbol=symbol, tf=tf, limit=limit, x_api_key=None)


# ═══════════════════════════════════════════════════════════════
# PAYPAL PAYMENTS
# ═══════════════════════════════════════════════════════════════
# Requires: PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET, PAYPAL_MODE

PAYPAL_MODE        = os.getenv("PAYPAL_MODE", "sandbox")
PAYPAL_CLIENT_ID   = os.getenv("PAYPAL_CLIENT_ID", "")
PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET", "")
PAYPAL_BASE = ("https://api.paypal.com" if PAYPAL_MODE == "live"
               else "https://api.sandbox.paypal.com")

# Plan prices in USD (sync with PLANS dict)
PLAN_PRICES_USD = {
    "weekly": 4.90,
    "monthly": 19.90,
    "quarterly": 49.90,
    "yearly": 189.00,
}

def _paypal_token() -> str:
    """Fetch a PayPal OAuth2 bearer token."""
    if not PAYPAL_CLIENT_ID or not PAYPAL_CLIENT_SECRET:
        raise HTTPException(503, "PayPal not configured. Set PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET.")
    creds = f"{PAYPAL_CLIENT_ID}:{PAYPAL_CLIENT_SECRET}".encode()
    b64   = __import__("base64").b64encode(creds).decode()
    data  = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req   = urllib.request.Request(
        f"{PAYPAL_BASE}/v1/oauth2/token",
        data=data,
        headers={"Authorization": f"Basic {b64}", "Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())["access_token"]

def _paypal_post(path: str, body: dict, token: str) -> dict:
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{PAYPAL_BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Prefer":        "return=representation",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())

def _paypal_get(path: str, token: str) -> dict:
    req = urllib.request.Request(
        f"{PAYPAL_BASE}{path}",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


@app.post("/api/paypal/create-order")
async def paypal_create_order(request: Request):
    """
    Create a PayPal order for one-time payment.
    Body: { "email": "...", "plan": "monthly" }
    Returns: { "order_id": "...", "approval_url": "..." }
    """
    body  = await request.json()
    email = body.get("email", "").strip().lower()
    plan  = body.get("plan", "monthly").lower()

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email required.")
    if plan not in PLAN_PRICES_USD:
        raise HTTPException(400, f"Unknown plan: {plan}")

    amount = PLAN_PRICES_USD[plan]
    token  = _paypal_token()

    order  = _paypal_post("/v2/checkout/orders", {
        "intent": "CAPTURE",
        "purchase_units": [{
            "amount": {"currency_code": "USD", "value": f"{amount:.2f}"},
            "description": f"InflectionAI {plan.title()} Subscription",
            "custom_id": f"{email}|{plan}",
        }],
        "application_context": {
            "return_url": f"{FRONTEND_URL}/?paypal=success&plan={plan}&email={urllib.parse.quote(email)}",
            "cancel_url": f"{FRONTEND_URL}/?paypal=canceled",
            "brand_name": "InflectionAI",
            "user_action": "PAY_NOW",
        },
    }, token)

    approval_url = next(
        (l["href"] for l in order.get("links", []) if l["rel"] == "approve"),
        None,
    )
    upsert_user(email)
    return {"order_id": order["id"], "approval_url": approval_url, "status": order["status"]}


@app.post("/api/paypal/capture-order")
async def paypal_capture_order(request: Request):
    """
    Capture a PayPal order after user approval.
    Body: { "order_id": "...", "email": "...", "plan": "..." }
    Returns: { "success": true, ... }
    """
    body     = await request.json()
    order_id = body.get("order_id", "")
    email    = body.get("email", "").strip().lower()
    plan     = body.get("plan", "monthly").lower()

    token    = _paypal_token()
    result   = _paypal_post(f"/v2/checkout/orders/{order_id}/capture", {}, token)

    if result.get("status") == "COMPLETED":
        # Activate subscription in DB
        # PayPal one-time payments: grant access for period length
        from datetime import datetime, timezone, timedelta
        periods = {"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}
        days    = periods.get(plan, 30)
        expires = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        capture_id = result["purchase_units"][0]["payments"]["captures"][0]["id"]

        with get_db() as db:
            db.execute(
                """INSERT INTO subscriptions
                   (user_email, stripe_customer_id, stripe_sub_id, plan, status, current_period_end)
                   VALUES (?,?,?,?,'active',?)""",
                (email, "paypal", capture_id, plan, expires)
            )
            db.commit()

        return {"success": True, "plan": plan, "expires": expires, "capture_id": capture_id}

    raise HTTPException(400, f"PayPal capture failed: {result.get('status')}")


# ═══════════════════════════════════════════════════════════════
# NOWPAYMENTS — USDT / CRYPTO
# ═══════════════════════════════════════════════════════════════
# Requires: NOWPAYMENTS_API_KEY, NOWPAYMENTS_IPN_SECRET

NOWPAYMENTS_KEY    = os.getenv("NOWPAYMENTS_API_KEY", "")
NOWPAYMENTS_IPN    = os.getenv("NOWPAYMENTS_IPN_SECRET", "")
NOWPAYMENTS_BASE   = "https://api.nowpayments.io/v1"

def _now_post(path: str, body: dict) -> dict:
    if not NOWPAYMENTS_KEY:
        raise HTTPException(503, "Crypto payments not configured. Set NOWPAYMENTS_API_KEY.")
    data = json.dumps(body).encode()
    req  = urllib.request.Request(
        f"{NOWPAYMENTS_BASE}{path}",
        data=data,
        headers={"x-api-key": NOWPAYMENTS_KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


@app.post("/api/crypto/create-invoice")
async def crypto_create_invoice(request: Request):
    """
    Create a NOWPayments USDT invoice.
    Body: { "email": "...", "plan": "monthly", "currency": "USDT" }
    Returns: { "payment_id": "...", "pay_address": "...", "pay_amount": ..., "pay_currency": "usdttrc20" }
    """
    body     = await request.json()
    email    = body.get("email", "").strip().lower()
    plan     = body.get("plan", "monthly").lower()
    currency = body.get("currency", "USDT").lower()

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email required.")
    if plan not in PLAN_PRICES_USD:
        raise HTTPException(400, f"Unknown plan: {plan}")

    amount   = PLAN_PRICES_USD[plan]
    # Use TRC20 USDT by default (low fees)
    pay_curr = "usdttrc20" if currency in ("usdt", "usdttrc20") else "usdterc20"

    result = _now_post("/payment", {
        "price_amount":      amount,
        "price_currency":    "usd",
        "pay_currency":      pay_curr,
        "order_id":          f"{plan}_{email}_{int(time.time())}",
        "order_description": f"InflectionAI {plan.title()} — {email}",
        "ipn_callback_url":  f"{os.getenv('BACKEND_URL', 'http://localhost:8000')}/api/crypto/webhook",
    })

    # Record as pending subscription
    upsert_user(email)
    with get_db() as db:
        db.execute(
            """INSERT INTO subscriptions
               (user_email, stripe_customer_id, stripe_session_id, plan, status)
               VALUES (?,?,?,?,'pending')""",
            (email, "nowpayments", result.get("payment_id", ""), plan)
        )
        db.commit()

    return {
        "payment_id":    result.get("payment_id"),
        "pay_address":   result.get("pay_address"),
        "pay_amount":    result.get("pay_amount"),
        "pay_currency":  result.get("pay_currency"),
        "price_amount":  amount,
        "status":        result.get("payment_status"),
    }


@app.post("/api/crypto/webhook")
async def crypto_webhook(request: Request):
    """
    NOWPayments IPN webhook — called when a crypto payment is confirmed.
    NOWPayments sends payment info; we verify HMAC and activate subscription.
    """
    body_bytes = await request.body()
    payload    = json.loads(body_bytes)
    sig_header = request.headers.get("x-nowpayments-sig", "")

    # Verify IPN signature
    if NOWPAYMENTS_IPN:
        sorted_payload = json.dumps(
            {k: payload[k] for k in sorted(payload)},
            separators=(",", ":"),
        ).encode()
        expected = hmac.new(
            NOWPAYMENTS_IPN.encode(),
            sorted_payload,
            hashlib.sha512,
        ).hexdigest()
        if not hmac.compare_digest(expected, sig_header):
            raise HTTPException(400, "Invalid IPN signature")

    status     = payload.get("payment_status", "")
    payment_id = str(payload.get("payment_id", ""))
    order_id   = payload.get("order_id", "")

    if status in ("finished", "confirmed", "partially_paid"):
        # Extract email and plan from order_id: "plan_email_timestamp"
        parts = order_id.split("_", 2)
        plan  = parts[0] if parts else "monthly"
        email_ts = parts[1] if len(parts) > 1 else ""
        # email might have @ in it — find by payment_id in DB
        periods  = {"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}
        days     = periods.get(plan, 30)
        expires  = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

        with get_db() as db:
            db.execute(
                """UPDATE subscriptions
                   SET status='active', current_period_end=?, updated_at=datetime('now')
                   WHERE stripe_session_id=? AND status='pending'""",
                (expires, payment_id)
            )
            db.commit()

    return {"received": True, "status": status}


# ═══════════════════════════════════════════════════════════════
# PAYPAL — CLIENT ID ENDPOINT (for frontend JS SDK)
# ═══════════════════════════════════════════════════════════════

@app.get("/api/paypal/client-id")
def paypal_client_id_endpoint():
    """Return PayPal client ID so frontend can load the PayPal JS SDK."""
    return {
        "client_id": PAYPAL_CLIENT_ID,
        "mode": PAYPAL_MODE,
        "configured": bool(PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET),
    }


# ═══════════════════════════════════════════════════════════════
# USDT — MANUAL WALLET FLOW (no third-party service needed)
# ═══════════════════════════════════════════════════════════════
# Set env var USDT_WALLET_ADDRESS to your TRC20 USDT address.

USDT_WALLET = os.getenv("USDT_WALLET_ADDRESS", "")

@app.get("/api/usdt/wallet")
def usdt_wallet(plan: str = Query("monthly"), email: str = Query("")):
    """Return the USDT wallet address and amount for the chosen plan."""
    if not USDT_WALLET:
        raise HTTPException(503, "USDT wallet not configured. Set USDT_WALLET_ADDRESS.")
    amount = PLAN_PRICES_USD.get(plan, 19.90)
    email  = email.strip().lower()
    if email and "@" in email:
        upsert_user(email)
        with get_db() as db:
            db.execute(
                """INSERT INTO subscriptions
                   (user_email, plan, status, payment_method)
                   VALUES (?,?,?,?)""",
                (email, plan, "pending_usdt", "usdt"),
            )
            db.commit()
    return {
        "address": USDT_WALLET,
        "amount":  amount,
        "currency": "USDT",
        "network": "TRC20",
        "plan": plan,
    }

@app.post("/api/usdt/submit")
async def usdt_submit_tx(request: Request):
    """User submits their USDT TX hash after sending payment."""
    body    = await request.json()
    email   = body.get("email", "").strip().lower()
    tx_hash = body.get("tx_hash", "").strip()
    plan    = body.get("plan", "monthly").lower()

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email required.")
    if not tx_hash or len(tx_hash) < 20:
        raise HTTPException(400, "Valid TX hash required.")

    upsert_user(email)
    with get_db() as db:
        # Update most recent pending_usdt row, or insert new
        updated = db.execute(
            """UPDATE subscriptions SET tx_hash=?, updated_at=datetime('now')
               WHERE user_email=? AND status='pending_usdt'
               ORDER BY created_at DESC LIMIT 1""",
            (tx_hash, email)
        ).rowcount
        if not updated:
            db.execute(
                """INSERT INTO subscriptions
                   (user_email, plan, status, payment_method, tx_hash)
                   VALUES (?,?,?,?,?)""",
                (email, plan, "pending_usdt", "usdt", tx_hash),
            )
        db.commit()
    return {
        "success": True,
        "message": "TX hash received. We'll verify on-chain within 24h and email you upon activation.",
    }


# ═══════════════════════════════════════════════════════════════
# STRIPE — ALIPAY / WECHAT PAY (one-time payment, not subscription)
# ═══════════════════════════════════════════════════════════════

@app.post("/api/subscribe/alipay-checkout")
async def alipay_checkout(request: Request):
    """
    Stripe Checkout for Alipay or WeChat Pay (one-time payment).
    Body: { "email": "...", "plan": "monthly", "method": "alipay" | "wechat_pay" }
    """
    if not STRIPE_OK:
        raise HTTPException(503, "Stripe not configured. Set STRIPE_SECRET_KEY.")
    body   = await request.json()
    email  = body.get("email", "").strip().lower()
    plan   = body.get("plan", "monthly").lower()
    method = body.get("method", "alipay")   # "alipay" | "wechat_pay"

    if not email or "@" not in email:
        raise HTTPException(400, "Valid email required.")
    if plan not in PLANS:
        raise HTTPException(400, f"Unknown plan: {plan}")
    if method not in ("alipay", "wechat_pay"):
        raise HTTPException(400, "method must be 'alipay' or 'wechat_pay'")

    amount_cents = PLANS[plan]["price_usd"]   # already in cents
    plan_name    = PLANS[plan]["name"]

    upsert_user(email)

    session = stripe.checkout.Session.create(
        customer_email=email,
        payment_method_types=[method],
        line_items=[{
            "price_data": {
                "currency": "usd",
                "unit_amount": amount_cents,
                "product_data": {"name": f"InflectionAI {plan_name}"},
            },
            "quantity": 1,
        }],
        mode="payment",   # one-time (Alipay/WeChat don't support subscriptions)
        success_url=f"{FRONTEND_URL}/?stripe=success&plan={plan}&email={urllib.parse.quote(email)}",
        cancel_url=f"{FRONTEND_URL}/?stripe=canceled",
        metadata={"plan": plan, "email": email, "method": method},
    )

    # Record pending
    with get_db() as db:
        db.execute(
            """INSERT INTO subscriptions
               (user_email, stripe_session_id, plan, status, payment_method)
               VALUES (?,?,?,?,?)""",
            (email, session.id, plan, "pending", method),
        )
        db.commit()

    return {"url": session.url, "session_id": session.id}


# ── Admin: manually activate USDT payment ───────────────────────
@app.post("/api/admin/activate-usdt")
async def admin_activate_usdt(request: Request,
                               admin_key: Optional[str] = Header(None, alias="x-admin-key")):
    """Manually activate a USDT subscription after verifying TX on-chain."""
    if admin_key != os.getenv("ADMIN_KEY"):
        raise HTTPException(403, "Invalid admin key")
    body  = await request.json()
    email = body.get("email", "").strip().lower()
    plan  = body.get("plan", "monthly").lower()
    days  = {"weekly": 7, "monthly": 30, "quarterly": 90, "yearly": 365}.get(plan, 30)
    expires = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()

    with get_db() as db:
        db.execute(
            """UPDATE subscriptions
               SET status='active', current_period_end=?, updated_at=datetime('now')
               WHERE user_email=? AND payment_method='usdt' AND status='pending_usdt'
               ORDER BY created_at DESC LIMIT 1""",
            (expires, email)
        )
        db.commit()
    return {"activated": True, "email": email, "expires": expires}


# ── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print("=" * 56)
    print("  InflectionAI Backend")
    print(f"  http://localhost:{port}")
    print(f"  Docs: http://localhost:{port}/docs")
    print("  Auto-reload: ON (file changes apply instantly)")
    print("=" * 56)
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
