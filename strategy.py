"""
strategy.py — Signal generation layer
======================================
Defines the abstract BaseStrategy and the concrete EMACrossoverStrategy
that replicates the Darrius AI Inflection Hunter algorithm.

Algorithm (fully reverse-engineered, verified against live Darrius data):
eB = EMA(fast) crosses ABOVE EMA(slow)
eS = EMA(fast) crosses BELOW EMA(slow)
B = close of bar[eB_idx + confirm_window], IF no eS fires in between
S = close of bar[eS_idx + confirm_window], IF no eB fires in between

Default params (matched to Darrius): fast=8, slow=20, confirm_window=2
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd
import numpy as np


class SignalSide(str, Enum):
    eB = "eB"
    B  = "B"
    eS = "eS"
    S  = "S"

BUY_SIDES  = {SignalSide.eB, SignalSide.B}
SELL_SIDES = {SignalSide.eS, SignalSide.S}

@dataclass
class Signal:
    time:     pd.Timestamp
    side:     SignalSide
    price:    float
    reason:   str
    regime:   str
    strength: float

    def __repr__(self):
        return (f"Signal({self.side.value:3s} | "
                f"{str(self.time.date()):10s} | "
                f"${self.price:>9.4f} | {self.reason})")


class BaseStrategy(ABC):
    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        pass

    def signals_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        signals = self.generate_signals(df)
        if not signals:
            return pd.DataFrame()
        rows = [
            {"time": s.time, "side": s.side.value, "price": s.price,
             "reason": s.reason, "regime": s.regime, "strength": s.strength}
            for s in signals
        ]
        return pd.DataFrame(rows).set_index("time")

    def __repr__(self):
        return f"{self.__class__.__name__}()"


class EMACrossoverStrategy(BaseStrategy):
    """
    EMA Crossover signal strategy — loop-based implementation.

    Avoids pandas boolean-shift dtype issues that caused spurious signals
    in the vectorised approach (shift() on bool Series → float64 → ~ broken).
    """
    name = "EMA Crossover (Inflection Hunter)"

    def __init__(self, fast: int = 8, slow: int = 20, confirm_window: int = 2):
        assert fast < slow, "fast EMA period must be smaller than slow"
        assert confirm_window >= 1
        self.fast           = fast
        self.slow           = slow
        self.confirm_window = confirm_window

    def generate_signals(self, df: pd.DataFrame) -> List[Signal]:
        if len(df) < self.slow + self.confirm_window:
            raise ValueError(
                f"Need at least {self.slow + self.confirm_window} bars, got {len(df)}."
            )

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()

        signals:    List[Signal] = []
        pending_eb: Optional[int]  = None
        pending_es: Optional[int]  = None
        prev_above: Optional[bool] = None

        for i in range(len(df)):
            ts        = df.index[i]
            close     = float(df["close"].iloc[i])
            cur_above = bool(ema_fast.iloc[i] > ema_slow.iloc[i])

            if prev_above is not None:
                if cur_above and not prev_above:
                    # EMA fast just crossed ABOVE slow → eB
                    signals.append(Signal(
                        time=ts, side=SignalSide.eB, price=close,
                        reason="ema_fast_cross_up", regime="UP", strength=0.55
                    ))
                    pending_eb = i
                    pending_es = None   # cancel pending S

                elif not cur_above and prev_above:
                    # EMA fast just crossed BELOW slow → eS
                    signals.append(Signal(
                        time=ts, side=SignalSide.eS, price=close,
                        reason="ema_fast_cross_down", regime="DOWN", strength=0.55
                    ))
                    pending_es = i
                    pending_eb = None   # cancel pending B

            # B confirmation: exactly confirm_window bars after eB
            if pending_eb is not None and (i - pending_eb) == self.confirm_window:
                signals.append(Signal(
                    time=ts, side=SignalSide.B, price=close,
                    reason=f"confirm_window_{self.confirm_window}",
                    regime="UP", strength=0.9
                ))
                pending_eb = None

            # S confirmation: exactly confirm_window bars after eS
            if pending_es is not None and (i - pending_es) == self.confirm_window:
                signals.append(Signal(
                    time=ts, side=SignalSide.S, price=close,
                    reason=f"confirm_window_{self.confirm_window}",
                    regime="DOWN", strength=0.9
                ))
                pending_es = None

            prev_above = cur_above

        return signals

    def get_indicator_series(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["ema_fast"] = df["close"].ewm(span=self.fast, adjust=False).mean()
        df["ema_slow"] = df["close"].ewm(span=self.slow, adjust=False).mean()
        df["ema20"]    = df["ema_slow"]
        df["ema50"]    = df["close"].ewm(span=50, adjust=False).mean()
        return df

    def __repr__(self):
        return (f"EMACrossoverStrategy(fast={self.fast}, slow={self.slow}, "
                f"confirm_window={self.confirm_window})")


STRATEGY_REGISTRY = {
    "ema_crossover": EMACrossoverStrategy,
}

def get_strategy(name: str, **kwargs) -> BaseStrategy:
    cls = STRATEGY_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}")
    return cls(**kwargs)
