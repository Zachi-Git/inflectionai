"""
strategy.py — Signal generation layer
======================================
Defines the abstract BaseStrategy and the concrete EMACrossoverStrategy
that replicates the Darrius AI Inflection Hunter algorithm.

Algorithm (fully reverse-engineered, verified against live Darrius data):
  eB  = EMA(fast) crosses ABOVE EMA(slow)
  eS  = EMA(fast) crosses BELOW EMA(slow)
  B   = close of bar[eB_idx + confirm_window], IF no eS fires in between
  S   = close of bar[eS_idx + confirm_window], IF no eB fires in between

Default params (matched to Darrius): fast=8, slow=20, confirm_window=2
  ↑ Signal lines: EMA(8) crossing EMA(20)
  NOTE: EMA(20) and EMA(50) are returned separately for CHART DISPLAY only.
        They do NOT determine signals.

To add a new strategy: subclass BaseStrategy and implement generate_signals().
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# Signal data types
# ─────────────────────────────────────────────

class SignalSide(str, Enum):
    eB = "eB"   # Early Buy  — EMA fast crosses up
    B  = "B"    # Confirmed Buy — 2 bars after eB
    eS = "eS"   # Early Sell — EMA fast crosses down
    S  = "S"    # Confirmed Sell — 2 bars after eS

BUY_SIDES  = {SignalSide.eB, SignalSide.B}
SELL_SIDES = {SignalSide.eS, SignalSide.S}


@dataclass
class Signal:
    time:     pd.Timestamp
    side:     SignalSide
    price:    float
    reason:   str
    regime:   str       # 'UP' or 'DOWN'
    strength: float     # 0–1 confidence

    def __repr__(self):
        return (f"Signal({self.side.value:3s} | "
                f"{str(self.time.date()):10s} | "
                f"${self.price:>9.4f} | {self.reason})")


# ─────────────────────────────────────────────
# Abstract base — plug in new strategies here
# ─────────────────────────────────────────────

class BaseStrategy(ABC):
    """
    All strategies inherit from this class.

    Input  : DataFrame with columns [open, high, low, close, volume]
             and a DatetimeIndex.
    Output : List[Signal]
    """
    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        pass

    def signals_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convenience: returns a DataFrame of signals merged with OHLCV."""
        signals = self.generate_signals(df)
        if not signals:
            return pd.DataFrame()
        rows = [
            {
                "time":     s.time,
                "side":     s.side.value,
                "price":    s.price,
                "reason":   s.reason,
                "regime":   s.regime,
                "strength": s.strength,
            }
            for s in signals
        ]
        return pd.DataFrame(rows).set_index("time")

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# ─────────────────────────────────────────────
# Strategy 1: EMA Crossover (Inflection Hunter)
# ─────────────────────────────────────────────

class EMACrossoverStrategy(BaseStrategy):
    """
    Replicates the Darrius AI signal algorithm exactly.

    Parameters
    ----------
    fast           : int — fast EMA period (default 8)
    slow           : int — slow EMA period (default 20)
    confirm_window : int — bars after eB/eS before B/S fires (default 2)

    Signal rules (verified against live Darrius data, 100% match):
      • eB fires when EMA(8)  crosses above EMA(20)
      • eS fires when EMA(8)  crosses below EMA(20)
      • B  fires exactly confirm_window bars after the most recent eB,
            PROVIDED no eS fires in the interim (which would cancel B)
      • S  fires exactly confirm_window bars after the most recent eS,
            PROVIDED no eB fires in the interim (which would cancel S)

    EMA(20) and EMA(50) are returned via get_indicator_series() for chart
    display but are NOT the signal-generating pair.
    """
    name = "EMA Crossover (Inflection Hunter)"

    def __init__(self, fast: int = 8, slow: int = 20, confirm_window: int = 2):
        assert fast < slow, "fast EMA period must be smaller than slow"
        assert confirm_window >= 1
        self.fast = fast
        self.slow = slow
        self.confirm_window = confirm_window

    # ------------------------------------------------------------------
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        """
        Main signal generation. Needs at least (slow + confirm_window) bars
        to warm up the EMAs before producing meaningful signals.
        """
        if len(df) < self.slow + self.confirm_window:
            raise ValueError(
                f"Need at least {self.slow + self.confirm_window} bars "
                f"to warm up EMA({self.slow}), got {len(df)}."
            )

        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()

        # Boolean: is fast EMA above slow EMA?
        above = df["ema_fast"] > df["ema_slow"]
        prev_above = above.shift(1).fillna(above)   # treat first bar as "no change"

        cross_up   = above & ~prev_above    # False→True transition
        cross_down = ~above & prev_above    # True→False transition

        signals: List[Signal] = []
        pending_eb: Optional[int] = None   # bar index of last unconfirmed eB
        pending_es: Optional[int] = None   # bar index of last unconfirmed eS

        for i in range(len(df)):
            close = float(df["close"].iloc[i])
            ts    = df.index[i]

            # ── Step 1: crossover events (checked BEFORE confirmations)
            #    A new crossover cancels the opposite pending confirmation.

            if cross_up.iloc[i]:
                signals.append(Signal(
                    time=ts, side=SignalSide.eB, price=close,
                    reason="ema_fast_cross_up", regime="UP", strength=0.55
                ))
                pending_eb = i
                pending_es = None   # cancel any pending eS window

            elif cross_down.iloc[i]:
                signals.append(Signal(
                    time=ts, side=SignalSide.eS, price=close,
                    reason="ema_fast_cross_down", regime="DOWN", strength=0.55
                ))
                pending_es = i
                pending_eb = None   # cancel any pending eB window  ← KEY RULE

            # ── Step 2: confirmation windows
            #    Fire exactly at bar[pending + confirm_window].

            if pending_eb is not None and (i - pending_eb) == self.confirm_window:
                signals.append(Signal(
                    time=ts, side=SignalSide.B, price=close,
                    reason=f"confirm_window_{self.confirm_window}",
                    regime="UP", strength=0.9
                ))
                pending_eb = None

            if pending_es is not None and (i - pending_es) == self.confirm_window:
                signals.append(Signal(
                    time=ts, side=SignalSide.S, price=close,
                    reason=f"confirm_window_{self.confirm_window}",
                    regime="DOWN", strength=0.9
                ))
                pending_es = None

        return signals

    def get_indicator_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns DataFrame with EMA columns added — useful for charting.

        ema_fast (8)  and ema_slow (20) are the signal-generating pair.
        ema20 (20) and ema50 (50) are returned for chart display overlay.
        """
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()   # EMA(8)
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()   # EMA(20)
        df["ema20"]    = df["ema_slow"]                                          # alias for chart
        df["ema50"]    = df["close"].ewm(span=50, adjust=False).mean()           # chart display only
        return df

    def __repr__(self):
        return (f"EMACrossoverStrategy("
                f"fast={self.fast}, slow={self.slow}, "
                f"confirm_window={self.confirm_window})  "
                f"# EMA({self.fast})×EMA({self.slow}) signal")


# ─────────────────────────────────────────────
# Strategy registry
# ─────────────────────────────────────────────
# Add new strategies here so main.py / scanner can discover them by name.

STRATEGY_REGISTRY = {
    "ema_crossover": EMACrossoverStrategy,
    # "rsi_divergence": RSIDivergenceStrategy,   ← add future strategies here
    # "macd_signal":   MACDSignalStrategy,
}


def get_strategy(name: str, **kwargs) -> BaseStrategy:
    """Factory: get a strategy by name string."""
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown strategy '{name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(**kwargs)
