"""
agents/learning_agent.py
=========================
Self-Learning Agent — analyses failures and updates the mistake memory.
Also generates insights that guide future strategy invention.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from memory.mistake_memory import MistakeMemory
from strategies.strategy_templates import TradingStrategy
from utils.logger import AgentLogger


class LearningAgent:
    """
    Processes pipeline outcomes and distils reusable lessons into memory.
    Called at the end of every evolution cycle.
    """

    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config   = config
        self.memory   = memory
        self._log     = AgentLogger("LearningAgent", config)
        self._cycle   = 0

    # ── Failure recording ────────────────────────────────────────────────────

    def record_invention_failure(self, reason: str, context: Dict = {}):
        self.memory.record(
            "StrategyInventorAgent",
            f"Invention failure: {reason}",
            remedy="Review system prompt and output format. Ensure JSON schema compliance.",
            phase="INVENTION", context=context,
        )
        self._log.warning(f"Invention failure recorded: {reason}", phase="LEARNING")

    def record_validation_failure(self, strategy_name: str, issues: List[str]):
        self.memory.record(
            "StrategyValidatorAgent",
            f"Strategy '{strategy_name}' failed validation: {'; '.join(issues[:3])}",
            remedy=(
                "Ensure generated strategies have: valid timeframe, instrument in [NIFTY/BANKNIFTY/FINNIFTY], "
                "target_pct > stoploss_pct, at least 1 indicator, non-empty edge_hypothesis."
            ),
            phase="VALIDATION",
            context={"strategy_name": strategy_name, "issues": issues},
        )

    def record_backtest_failure(self, strategy_id: str, error: str):
        self.memory.record(
            "BacktestAgent",
            f"Backtest failed for {strategy_id}: {error}",
            remedy="Verify that entry/exit parameters are numeric and within reasonable bounds.",
            phase="BACKTEST",
            context={"strategy_id": strategy_id},
        )

    def record_evaluation_rejection(
        self,
        strategy: TradingStrategy,
        metrics  : Dict,
        reason   : str,
        hint     : str,
    ):
        description = (
            f"'{strategy.name}' rejected — {reason} | "
            f"WR={metrics.get('win_rate',0):.1%} "
            f"PF={metrics.get('profit_factor',0):.2f} "
            f"DD={metrics.get('max_drawdown_pct',0):.1f}%"
        )
        self.memory.record(
            "EvaluatorAgent",
            description,
            remedy=hint,
            phase="EVALUATION",
            context={
                "strategy_id"  : strategy.id,
                "strategy_type": strategy.strategy_type,
                "timeframe"    : strategy.timeframe,
                "metrics"      : metrics,
            },
        )
        self._log.info(
            f"Evaluation rejection recorded for '{strategy.name}'",
            phase="LEARNING",
        )

    def record_risk_failure(self, strategy: TradingStrategy, reason: str):
        self.memory.record(
            "RiskManagerAgent",
            f"Risk filter failed for '{strategy.name}': {reason}",
            remedy="Ensure target > SL and risk parameters are within safe operational limits.",
            phase="RISK",
            context={"strategy_id": strategy.id},
        )

    # ── Insight generation ───────────────────────────────────────────────────

    def generate_cycle_insights(self, cycle_stats: Dict[str, Any]) -> str:
        """
        Build a brief learning summary for the current cycle.
        Logged and optionally fed back into the next invention prompt.
        """
        self._cycle += 1
        accepted  = cycle_stats.get("accepted", 0)
        rejected  = cycle_stats.get("rejected", 0)
        errors    = cycle_stats.get("errors", 0)
        total     = accepted + rejected + errors
        mem_stats = self.memory.summary_stats()

        summary = (
            f"Cycle {self._cycle}: {total} strategies processed | "
            f"Accepted={accepted} Rejected={rejected} Errors={errors} | "
            f"Total mistakes in memory: {mem_stats['total_mistakes']}"
        )
        self._log.info(summary, phase="LEARNING")
        return summary

    def get_guidance_for_inventor(self) -> List[str]:
        """Return recent mistake remedies to inject into invention prompts."""
        return self.memory.get_all_remedies(limit=10)
