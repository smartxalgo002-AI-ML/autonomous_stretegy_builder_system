"""
agents/risk_manager_agent.py
=============================
Risk Management Agent — final gate before strategy is stored.
Validates strategy risk parameters are within safe operational limits.
"""
from __future__ import annotations
from typing import Dict, Tuple
from backtesting.performance_metrics import RiskFilter
from strategies.strategy_templates import TradingStrategy, PerformanceMetrics
from memory.mistake_memory import MistakeMemory
from utils.logger import AgentLogger


class RiskManagerAgent:
    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config = config
        self.memory = memory
        self._log   = AgentLogger("RiskManagerAgent", config)

    def assess(
        self,
        strategy    : TradingStrategy,
        metrics_dict: Dict,
    ) -> Tuple[bool, str]:
        """
        Final risk gate.
        Returns (risk_ok, reason).
        """
        # Prepare metrics data, ensuring strategy_id is not double-passed
        metrics_data = {
            k: v for k, v in metrics_dict.items()
            if k in PerformanceMetrics.model_fields
        }
        metrics_data.pop("strategy_id", None)

        metrics = PerformanceMetrics(strategy_id=strategy.id, **metrics_data)
        risk_ok, reason = RiskFilter.validate(strategy.to_dict(), metrics)

        if risk_ok:
            self._log.info(
                f"Risk check PASSED: {strategy.name}",
                phase="RISK", strategy_id=strategy.id,
            )
        else:
            self._log.warning(
                f"Risk check FAILED: {strategy.name} — {reason}",
                phase="RISK", strategy_id=strategy.id,
            )
            self.memory.record(
                agent_name  = "RiskManagerAgent",
                description = f"Risk filter failed for '{strategy.name}': {reason}",
                remedy      = (
                    "Ensure target > stoploss, max_risk_per_trade_pct ≤ 5%, "
                    "daily_loss_limit_pct ≤ 10%."
                ),
                phase       = "RISK",
                context     = {"strategy_id": strategy.id},
            )

        return risk_ok, reason
