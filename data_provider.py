"""
data_provider.py — Data source layer
======================================
Three providers, all returning a standardised OHLCV DataFrame:
  - YFinanceProvider  : free, no API key, works out of the box
  - MoomooProvider    : real-time / delayed via Moomoo OpenAPI (requires OpenD)
  - DarriusAPIProvider: fetches directly from Darrius backend (no auth needed)

All providers return a DataFrame with:
  index  : DatetimeIndex (timezone-naive)
  columns: open, high, low, close, volume  (all float)

Usage
-----
  from data_provider import YFinanceProvider
  dp = YFinanceProvider()
  df = dp.get_ohlcv("TSLA", timeframe="1d", limit=600)
"""

from abc import ABC, abstractmethod
from typing import Optional
import pandas as pd


# ─────────────────────────────────────────────
# Abstract base
# ─────────────────────────────────────────────

class BaseDataProvider(ABC):
    """
    All providers inherit from this.
    Implement get_ohlcv() and you get scanner / backtest support for free.
    """

    # Map from our unified timeframe strings to provider-specific strings
    TF_MAP: dict = {}

    @abstractmethod
    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str = "1d",
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        limit:     int = 600,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with columns [open, high, low, close, volume]
        and a timezone-naive DatetimeIndex, sorted ascending by time.

        Parameters
        ----------
        symbol    : ticker, e.g. "TSLA", "AAPL", "BTC-USD"
        timeframe : one of 1m 5m 15m 30m 1h 4h 1d 1w 1M
        start     : "YYYY-MM-DD" (optional, overrides limit)
        end       : "YYYY-MM-DD" (optional)
        limit     : max number of bars to return (newest N bars)
        """

    @staticmethod
    def _normalise(df: pd.DataFrame, limit: int) -> pd.DataFrame:
        """Standardise column names, drop NaNs, enforce limit."""
        rename = {}
        for col in df.columns:
            low = col.lower()
            if low in ("open", "high", "low", "close", "volume"):
                rename[col] = low
        df = df.rename(columns=rename)
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        # Strip timezone so all providers behave identically
        if df.index.tzinfo is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        return df.tail(limit)


# ─────────────────────────────────────────────
# Provider 1: yfinance (free, no key needed)
# ─────────────────────────────────────────────

class YFinanceProvider(BaseDataProvider):
    """
    Uses the yfinance library to fetch Yahoo Finance data.
    Free, no API key required.  Slight delay on real-time data.

    Install: pip install yfinance
    """

    TF_MAP = {
        "1m":  "1m",
        "5m":  "5m",
        "15m": "15m",
        "30m": "30m",
        "1h":  "1h",
        "4h":  "1h",    # Yahoo has no 4h; fall back to 1h
        "1d":  "1d",
        "1w":  "1wk",
        "1M":  "1mo",
    }

    # Approximate period strings when no start date is given
    _PERIOD_MAP = {
        "1m":  "7d",
        "5m":  "60d",
        "15m": "60d",
        "30m": "60d",
        "1h":  "730d",
        "4h":  "730d",
        "1d":  "10y",
        "1w":  "20y",
        "1M":  "20y",
    }

    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str = "1d",
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        limit:     int = 600,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Run: pip install yfinance")

        yf_tf = self.TF_MAP.get(timeframe, "1d")
        ticker = yf.Ticker(symbol)

        if start:
            df = ticker.history(start=start, end=end, interval=yf_tf, auto_adjust=True)
        else:
            period = self._PERIOD_MAP.get(timeframe, "10y")
            df = ticker.history(period=period, interval=yf_tf, auto_adjust=True)

        if df.empty:
            raise ValueError(f"No data returned for {symbol} [{timeframe}].")

        # yfinance column names vary; add volume=0 if missing
        if "Volume" not in df.columns and "volume" not in df.columns:
            df["Volume"] = 0

        return self._normalise(df, limit)


# ─────────────────────────────────────────────
# Provider 2: Moomoo OpenAPI
# ─────────────────────────────────────────────

class MoomooProvider(BaseDataProvider):
    """
    Fetches historical K-lines from the Moomoo OpenAPI via the futu-api SDK.

    Prerequisites
    -------------
    1. Install Moomoo desktop app and log in.
    2. Enable OpenD (Settings → OpenD gateway) — default host 127.0.0.1:11111.
    3. pip install futu-api

    Parameters
    ----------
    host    : OpenD host (default "127.0.0.1")
    port    : OpenD port (default 11111)
    market  : default market prefix if symbol has no "." — "US", "HK", "SH", "SZ"
    autype  : adjustment type — "qfq" (forward), "hfq" (backward), "none"

    Symbol format
    -------------
    US stocks : "TSLA" or "US.TSLA"
    HK stocks : "HK.00700"
    A shares  : "SH.600519" or "SZ.000001"
    """

    TF_MAP = {
        "1m":  "K_1M",
        "3m":  "K_3M",
        "5m":  "K_5M",
        "15m": "K_15M",
        "30m": "K_30M",
        "1h":  "K_60M",
        "4h":  "K_4H",
        "1d":  "K_DAY",
        "1w":  "K_WEEK",
        "1M":  "K_MON",
    }

    AUTYPE_MAP = {
        "qfq":  "QFQ",
        "hfq":  "HFQ",
        "none": "NONE",
    }

    def __init__(
        self,
        host:   str = "127.0.0.1",
        port:   int = 11111,
        market: str = "US",
        autype: str = "qfq",
    ):
        self.host   = host
        self.port   = port
        self.market = market.upper()
        self.autype = autype.lower()

    def _code(self, symbol: str) -> str:
        """Ensure symbol has a market prefix."""
        return symbol if "." in symbol else f"{self.market}.{symbol}"

    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str = "1d",
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        limit:     int = 600,
    ) -> pd.DataFrame:
        try:
            import futu
            from futu import OpenQuoteContext, KLType, AuType, KL_FIELD, RET_OK
        except ImportError:
            raise ImportError("Run: pip install futu-api")

        code      = self._code(symbol)
        ktype_str = self.TF_MAP.get(timeframe, "K_DAY")
        ktype     = getattr(KLType, ktype_str)
        autype    = getattr(AuType, self.AUTYPE_MAP.get(self.autype, "QFQ"))

        ctx = OpenQuoteContext(host=self.host, port=self.port)
        try:
            ret, data, _ = ctx.request_history_kline(
                code,
                start=start,
                end=end,
                ktype=ktype,
                autype=autype,
                fields=[KL_FIELD.ALL],
                max_count=limit,
            )
            if ret != RET_OK:
                raise RuntimeError(f"Moomoo API error ({ret}): {data}")
        finally:
            ctx.close()

        df = data.copy()
        df = df.rename(columns={"time_key": "time"})
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        # Moomoo column names are already lowercase
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = 0.0

        return self._normalise(df, limit)

    def get_realtime_quote(self, symbols: list) -> pd.DataFrame:
        """
        Returns a snapshot of current bid/ask and last price for a list of symbols.
        Useful for live signal monitoring.
        """
        try:
            from futu import OpenQuoteContext, RET_OK
        except ImportError:
            raise ImportError("Run: pip install futu-api")

        codes = [self._code(s) for s in symbols]
        ctx = OpenQuoteContext(host=self.host, port=self.port)
        try:
            ret, data = ctx.get_market_snapshot(codes)
            if ret != RET_OK:
                raise RuntimeError(f"Moomoo snapshot error: {data}")
            return data
        finally:
            ctx.close()


# ─────────────────────────────────────────────
# Provider 3: Darrius AI API (no auth)
# ─────────────────────────────────────────────

class DarriusAPIProvider(BaseDataProvider):
    """
    Fetches OHLCV data directly from the Darrius AI backend.
    Currently requires no authentication for basic data.
    Also exposes get_signals() to retrieve their pre-computed signals.

    Note: uses Twelve Data (delayed).  May break if Darrius adds auth.
    """

    BASE_URL = "https://darrius-api.onrender.com/api/market/snapshot"

    TF_MAP = {
        "5m":  "5m",
        "15m": "15m",
        "30m": "30m",
        "1h":  "1h",
        "4h":  "4h",
        "1d":  "1d",
        "1w":  "1w",
        "1M":  "1M",
    }

    def _fetch(self, symbol: str, timeframe: str, limit: int) -> dict:
        try:
            import requests
        except ImportError:
            raise ImportError("Run: pip install requests")

        tf = self.TF_MAP.get(timeframe, "1d")
        resp = requests.get(
            self.BASE_URL,
            params={"symbol": symbol, "tf": tf, "limit": limit, "source": "twelve"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str = "1d",
        start:     Optional[str] = None,
        end:       Optional[str] = None,
        limit:     int = 600,
    ) -> pd.DataFrame:
        data = self._fetch(symbol, timeframe, limit)
        bars = pd.DataFrame(data["bars"])
        bars["time"] = pd.to_datetime(bars["time"], unit="s")
        bars = bars.set_index("time")
        if "volume" not in bars.columns:
            bars["volume"] = 0.0
        return self._normalise(bars, limit)

    def get_signals(self, symbol: str, timeframe: str = "1d", limit: int = 600) -> list:
        """
        Returns the raw signal list from Darrius (pre-computed server-side).
        Each item: {side, time, price, reason, regime, strength}
        """
        data = self._fetch(symbol, timeframe, limit)
        return data.get("signals", [])

    def get_meta(self, symbol: str, timeframe: str = "1d") -> dict:
        """Returns the meta block: ema_period, aux_period, data_mode, etc."""
        data = self._fetch(symbol, timeframe, 1)
        return data.get("meta", {})


# ─────────────────────────────────────────────
# Provider registry
# ─────────────────────────────────────────────

PROVIDER_REGISTRY = {
    "yfinance": YFinanceProvider,
    "moomoo":   MoomooProvider,
    "darrius":  DarriusAPIProvider,
}


def get_provider(name: str, **kwargs) -> BaseDataProvider:
    """Factory: get a provider by name string."""
    cls = PROVIDER_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown provider '{name}'. "
            f"Available: {list(PROVIDER_REGISTRY.keys())}"
        )
    return cls(**kwargs)
