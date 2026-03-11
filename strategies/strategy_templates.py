"""
strategies/strategy_templates.py
=================================
Pydantic data-models that define the schema for every trading strategy.
All agents produce / consume these models — keeping the contract strict.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enums (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class StrategyType(str, Enum):
    INTRADAY_OPTIONS   = "intraday_options"
    SWING_OPTIONS      = "swing_options"
    SCALPING           = "scalping"
    MOMENTUM           = "momentum"
    MEAN_REVERSION     = "mean_reversion"
    VOLATILITY_BREAKOUT= "volatility_breakout"
    IRON_CONDOR        = "iron_condor"
    STRADDLE           = "straddle"
    STRANGLE           = "strangle"


class Timeframe(str, Enum):
    M1   = "1min"
    M5   = "5min"
    M15  = "15min"
    M30  = "30min"
    H1   = "1hour"
    D1   = "1day"


class OptionType(str, Enum):
    CALL = "CE"
    PUT  = "PE"
    BOTH = "BOTH"


class Direction(str, Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    BOTH  = "BOTH"


class OptionAction(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"


class StrikeSelectionMethod(str, Enum):
    ATM_OFFSET   = "ATM_OFFSET"
    DELTA_TARGET = "DELTA_TARGET"
    OTM_PCT      = "OTM_PCT"


class ExitType(str, Enum):
    TARGET           = "TARGET"
    STOPLOSS         = "STOPLOSS"
    TRAILING_SL      = "TRAILING_SL"
    INDICATOR_SIGNAL = "INDICATOR_SIGNAL"
    TIME_EXIT        = "TIME_EXIT"
    PROFIT_TRAIL     = "PROFIT_TRAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-models
# ─────────────────────────────────────────────────────────────────────────────

class Indicator(BaseModel):
    """Definition of a single technical indicator."""
    name       : str = Field(..., description="e.g. 'EMA', 'RSI', 'VWAP'")
    params     : Dict[str, Any] = Field(default_factory=dict, description="e.g. {'period': 14}")
    signal_role: str = Field("confirmation", description="'entry_trigger' | 'confirmation' | 'exit_trigger'")
    condition  : str = Field("", description="Human-readable rule, e.g. 'price > EMA(20)'")


class EntryCondition(BaseModel):
    """Complete entry logic for a strategy."""
    direction        : Direction = Direction.LONG
    primary_trigger  : str = Field(..., description="The main event that triggers entry")
    confirmation_rules: List[str] = Field(default_factory=list)
    indicators       : List[Indicator] = Field(default_factory=list)
    price_action_rules: List[str] = Field(default_factory=list)
    volume_condition : Optional[str] = None
    time_filter      : Optional[str] = Field(None, description="e.g. '09:15-10:30'")
    market_structure : Optional[str] = None


class ExitCondition(BaseModel):
    """Defines how a strategy exits a trade."""
    target_pct          : float = Field(3.0, gt=0, description="Take profit percentage (must be positive)")
    stoploss_pct        : float = Field(1.5, gt=0, description="Stop loss percentage (must be positive)")
    trailing_sl_pct     : Optional[float] = Field(None, gt=0)
    trailing_sl_trigger : Optional[float] = Field(None, gt=0)
    indicator_exit      : Optional[str]  = None
    time_exit           : Optional[str]  = Field(None, description="Format HH:MM, e.g. '15:15'")
    profit_trail_step   : Optional[float]= Field(None, gt=0)

    @model_validator(mode='after')
    def validate_tp_sl(self) -> 'ExitCondition':
        if self.target_pct <= self.stoploss_pct:
            raise ValueError(f"target_pct ({self.target_pct}) must be greater than stoploss_pct ({self.stoploss_pct})")
        return self


class OptionLeg(BaseModel):
    """Definition of a single leg in a multi-leg options strategy."""
    action                 : OptionAction
    option_type            : OptionType
    strike_selection_method: StrikeSelectionMethod = StrikeSelectionMethod.ATM_OFFSET
    strike_offset          : int = Field(0, description="Used if ATM_OFFSET (e.g. 100, -200)")
    delta_target           : Optional[float] = Field(None, description="Used if DELTA_TARGET (e.g. 0.3)")
    expiry_selection       : str = Field("weekly", description="weekly | monthly | next_expiry")
    quantity_ratio         : int = Field(1, ge=1, description="Quantity ratio relative to other legs")


class MarketRegimeFilter(BaseModel):
    """Filters based on market volatility and trend regimes."""
    min_iv_rank        : float = Field(0.0, ge=0, le=100)
    max_iv_rank        : float = Field(100.0, ge=0, le=100)
    min_adx            : float = Field(0.0, ge=0, le=100)
    regime_type        : str = "any"


class RiskParameters(BaseModel):
    """Per-trade and portfolio-level risk controls."""
    max_risk_per_trade_pct : float = Field(2.0, gt=0, le=10)
    max_open_positions     : int   = Field(3, ge=1, le=10)
    daily_loss_limit_pct   : float = Field(5.0, gt=0, le=15)
    position_sizing        : str   = "fixed"


# ─────────────────────────────────────────────────────────────────────────────
# Master Strategy Model
# ─────────────────────────────────────────────────────────────────────────────

class TradingStrategy(BaseModel):
    """
    Canonical representation of a trading strategy.
    Strictly enforced via Pydantic.
    """
    id            : str           = Field(default_factory=lambda: str(uuid.uuid4()))
    name          : str           = Field(..., min_length=3)
    description   : str           = Field(..., min_length=10)
    strategy_type : StrategyType
    instrument    : str           = Field("NIFTY", description="NIFTY | BANKNIFTY | FINNIFTY")
    timeframe     : Timeframe
    generation    : int           = 0
    parent_id     : Optional[str] = None

    entry         : EntryCondition
    exit          : ExitCondition
    legs          : List[OptionLeg] = Field(default_factory=list)
    regime        : MarketRegimeFilter = Field(default_factory=MarketRegimeFilter)
    risk          : RiskParameters = Field(default_factory=RiskParameters)

    edge_hypothesis    : str = Field(..., min_length=10)
    market_conditions  : str = ""
    known_weaknesses   : List[str] = Field(default_factory=list)
    creator_notes      : str = ""

    @model_validator(mode='after')
    def validate_options_legs(self) -> 'TradingStrategy':
        # Options strategies must have legs
        if self.strategy_type in [StrategyType.IRON_CONDOR, StrategyType.STRADDLE, StrategyType.STRANGLE, StrategyType.INTRADAY_OPTIONS, StrategyType.SWING_OPTIONS]:
            if not self.legs:
                raise ValueError(f"Strategy type {self.strategy_type} requires at least one option leg")
            if self.strategy_type == StrategyType.IRON_CONDOR and len(self.legs) < 4:
                 raise ValueError("Iron Condor requires at least 4 legs")
            if self.strategy_type in [StrategyType.STRADDLE, StrategyType.STRANGLE] and len(self.legs) < 2:
                 raise ValueError(f"{self.strategy_type} requires at least 2 legs")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradingStrategy":
        return cls(**d)

    def summary(self) -> str:
        return (
            f"[{self.strategy_type}] {self.name} | "
            f"{self.instrument} {self.timeframe} | "
            f"Gen {self.generation}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Performance Metrics Model (Unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class PerformanceMetrics(BaseModel):
    strategy_id       : str
    total_trades      : int   = 0
    winning_trades    : int   = 0
    losing_trades     : int   = 0
    win_rate          : float = 0.0
    profit_factor     : float = 0.0
    net_return_pct    : float = 0.0
    max_drawdown_pct  : float = 0.0
    sharpe_ratio      : float = 0.0
    sortino_ratio     : float = 0.0
    calmar_ratio      : float = 0.0
    avg_win_pct       : float = 0.0
    avg_loss_pct      : float = 0.0
    avg_holding_bars  : float = 0.0
    max_consec_losses : int   = 0
    expectancy        : float = 0.0
    passed            : bool  = False
    rejection_reason  : str   = ""

    def brief(self) -> str:
        return (
            f"WR={self.win_rate:.1%} PF={self.profit_factor:.2f} "
            f"Ret={self.net_return_pct:.1f}% DD={self.max_drawdown_pct:.1f}% "
            f"Sharpe={self.sharpe_ratio:.2f} Trades={self.total_trades}"
        )
