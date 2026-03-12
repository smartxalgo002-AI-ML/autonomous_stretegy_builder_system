"""
agents/evaluator_agent.py
=========================
Evaluation Agent — applies acceptance thresholds to backtest metrics.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
from backtesting.performance_metrics import StrategyEvaluator
from strategies.strategy_templates import PerformanceMetrics, TradingStrategy
from database.repository import StrategyRepository, BacktestRepository
from memory.mistake_memory import MistakeMemory
from utils.logger import AgentLogger
from utils.google_sheets import append_to_google_sheet

class EvaluatorAgent:
    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config    = config
        self.memory    = memory
        self._log      = AgentLogger("EvaluatorAgent", config)
        self._evaluator= StrategyEvaluator(config)
        self._repo     = StrategyRepository(config)
        self._bt_repo  = BacktestRepository()

    def evaluate(
        self,
        strategy : TradingStrategy,
        metrics_dict: Dict,
    ) -> Tuple[bool, str, float]:
        """
        Evaluate strategy metrics.
        Returns (passed, reason, composite_score).
        """
        metrics = PerformanceMetrics(strategy_id=strategy.id, **{
            k: v for k, v in metrics_dict.items()
            if k in PerformanceMetrics.model_fields and k != "strategy_id"
        })
        passed, reason, score = self._evaluator.evaluate(metrics)

        # Persistence
        self._bt_repo.mark_passed(strategy.id, passed)

        if passed:
            self._log.info(
                f"ACCEPTED [{score:.1f}]: {strategy.name} — {metrics.brief()}",
                phase="EVALUATION", strategy_id=strategy.id,
            )
        else:
            self._log.warning(
                f"REJECTED: {strategy.name} — {reason}",
                phase="EVALUATION", strategy_id=strategy.id,
            )
            self.memory.record(
                agent_name  = "EvaluatorAgent",
                description = f"Strategy '{strategy.name}' rejected: {reason}",
                remedy      = self._evaluator.improvement_hint(metrics),
                phase       = "EVALUATION",
                context     = {"strategy_id": strategy.id, "metrics": metrics_dict},
            )

        # ── Google Sheets ─────────────────────────────────────────────────────────────
        # This is the only point where strategy object + metrics + decision
        # are all available together — so this is where we write ONE complete row.
        append_to_google_sheet({
            # ─ identity ───────────────────────────────────────────────────────────
            "strategy_id"    : strategy.id,
            "strategy_name"  : strategy.name,
            "symbol"         : strategy.instrument,
            # ─ full strategy JSON — all schema columns auto-extracted ─────────
            "strategy_schema": strategy.to_dict(),
            # ─ performance metrics ─────────────────────────────────────────
            **metrics_dict,
            # ─ agent decision ─────────────────────────────────────────────
            "decision"       : "BUY" if passed else "HOLD",
            "agent_reason"   : reason,
            "passed"         : passed,
        })
        # ───────────────────────────────────────────────────────────────────────

        return passed, reason, score

    def improvement_hint(self, metrics_dict: Dict, strategy_id: str = "") -> str:
        metrics = PerformanceMetrics(strategy_id=strategy_id, **{
            k: v for k, v in metrics_dict.items()
            if k in PerformanceMetrics.model_fields and k != "strategy_id"
        })
        return self._evaluator.improvement_hint(metrics)
