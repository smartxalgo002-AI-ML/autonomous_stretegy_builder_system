"""
agents/strategy_validator.py
=============================
Strategy Validator Agent — structural validation before backtesting.
No LLM call — pure rule-based validation for speed and reliability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from memory.mistake_memory import MistakeMemory
from strategies.strategy_templates import TradingStrategy
from utils.logger import AgentLogger


VALID_STRATEGY_TYPES = {
    "intraday_options", "swing_options", "scalping",
    "momentum", "mean_reversion", "volatility_breakout",
    "iron_condor", "straddle", "strangle",
}
VALID_TIMEFRAMES = {"1min", "5min", "15min", "30min", "1hour", "1day"}
VALID_INSTRUMENTS = {"NIFTY", "BANKNIFTY", "FINNIFTY"}
VALID_OPTION_TYPES = {"CE", "PE", "BOTH"}


class StrategyValidatorAgent:
    """
    Runs a battery of structural checks on a generated strategy.
    Returns (is_valid, list_of_issues).
    """

    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config = config
        self.memory = memory
        self._log   = AgentLogger("StrategyValidatorAgent", config)

    def validate(self, strategy: TradingStrategy) -> Tuple[bool, List[str]]:
        issues = []

        # ── Basic field checks ───────────────────────────────────────────────
        if not strategy.name or len(strategy.name.strip()) < 3:
            issues.append("Strategy name too short or empty")

        if strategy.strategy_type not in VALID_STRATEGY_TYPES:
            issues.append(f"Invalid strategy_type: {strategy.strategy_type}")

        if strategy.timeframe not in VALID_TIMEFRAMES:
            issues.append(f"Invalid timeframe: {strategy.timeframe}")

        if strategy.instrument not in VALID_INSTRUMENTS:
            issues.append(f"Unsupported instrument: {strategy.instrument}")

        # ── Entry logic ──────────────────────────────────────────────────────
        if not strategy.entry.primary_trigger:
            issues.append("Entry primary_trigger is missing")

        if len(strategy.entry.indicators) == 0:
            issues.append("At least one indicator is required in entry conditions")

        max_ind = self.config.get("strategy", {}).get("max_indicator_count", 5)
        if len(strategy.entry.indicators) > max_ind:
            issues.append(f"Too many indicators ({len(strategy.entry.indicators)} > {max_ind})")

        for ind in strategy.entry.indicators:
            if not ind.name:
                issues.append("Indicator missing 'name' field")

        # ── Exit logic ───────────────────────────────────────────────────────
        if strategy.exit.target_pct <= 0:
            issues.append("Exit target_pct must be positive")

        if strategy.exit.stoploss_pct <= 0:
            issues.append("Exit stoploss_pct must be positive")

        if strategy.exit.target_pct <= strategy.exit.stoploss_pct:
            issues.append(
                f"target_pct ({strategy.exit.target_pct}) must be > stoploss_pct ({strategy.exit.stoploss_pct})"
            )

        # ── Options Legs ──────────────────────────────────────────────────────
        if not strategy.legs:
            issues.append("At least one option leg is required")
        
        for i, leg in enumerate(strategy.legs):
            if leg.option_type not in VALID_OPTION_TYPES:
                issues.append(f"Leg {i} has invalid option_type: {leg.option_type}")
            
            if leg.quantity_ratio <= 0:
                issues.append(f"Leg {i} quantity_ratio must be positive")


        # ── Risk params ──────────────────────────────────────────────────────
        if strategy.risk.max_risk_per_trade_pct <= 0 or strategy.risk.max_risk_per_trade_pct > 10:
            issues.append(
                f"max_risk_per_trade_pct {strategy.risk.max_risk_per_trade_pct} out of [0, 10] range"
            )

        if strategy.risk.daily_loss_limit_pct > 15:
            issues.append(f"daily_loss_limit_pct {strategy.risk.daily_loss_limit_pct} too high (>15%)")

        # ── Edge hypothesis ──────────────────────────────────────────────────
        if not strategy.edge_hypothesis or len(strategy.edge_hypothesis) < 10:
            issues.append("edge_hypothesis is missing or too brief")

        is_valid = len(issues) == 0

        if is_valid:
            self._log.info(
                f"Strategy VALID: {strategy.name}", phase="VALIDATION", strategy_id=strategy.id
            )
        else:
            self._log.warning(
                f"Strategy INVALID ({len(issues)} issues): {strategy.name}",
                phase="VALIDATION", strategy_id=strategy.id,
            )
            for issue in issues:
                self._log.debug(f"  ✗ {issue}", phase="VALIDATION")

            # Record validation failure in memory
            self.memory.record(
                agent_name  = "StrategyValidatorAgent",
                description = f"Validation failed for '{strategy.name}': {'; '.join(issues[:3])}",
                remedy      = (
                    "Ensure strategy has valid timeframe, instrument, positive target > stoploss, "
                    "at least one indicator, and non-empty edge hypothesis."
                ),
                phase       = "VALIDATION",
                context     = {"strategy_name": strategy.name, "issues": issues},
            )

        return is_valid, issues
