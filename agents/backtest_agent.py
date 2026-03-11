"""
agents/backtest_agent.py
========================
Backtest Agent — orchestrates the BatchBacktester and Walk-Forward Validation.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
from backtesting.backtester import BatchBacktester, _backtest_single_strategy
from backtesting.walk_forward import WalkForwardValidator
from strategies.strategy_templates import TradingStrategy, PerformanceMetrics
from database.repository import BacktestRepository
from memory.mistake_memory import MistakeMemory
from utils.logger import AgentLogger


class BacktestAgent:
    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config    = config
        self.memory    = memory
        self._log      = AgentLogger("BacktestAgent", config)
        self._btester  = BatchBacktester(config)
        self._bt_repo  = BacktestRepository()
        self._wf_validator = WalkForwardValidator(config)

    async def run_backtest(self, strategy: TradingStrategy) -> Optional[Dict]:
        """Run backtest + Walk-Forward validation for a single strategy."""
        self._log.info(
            f"Backtesting: {strategy.name}", phase="BACKTEST", strategy_id=strategy.id
        )
        try:
            # 1. Primary Backtest
            result = _backtest_single_strategy(strategy.to_dict(), self.config)
            if result.get("error"):
                self._log.error(f"Backtest error: {result['error']}", phase="BACKTEST", strategy_id=strategy.id)
                return None

            metrics = result["metrics"]
            
            # 2. Walk-Forward Check (Anti-Overfitting)
            wf_passed = self._wf_validator.validate(strategy)
            if not wf_passed:
                reason = f"Walk-Forward validation failed for {strategy.name}"
                self._log.warning(reason, phase="BACKTEST", strategy_id=strategy.id)
                self.memory.record("BacktestAgent", reason, remedy="Optimize parameters for robustness, not just peak performance.")
                # We return None to trigger rejection in the graph, but the log now shows it's a WF issue.
                return None

            # Persist to DB
            self._bt_repo.save_result({
                **metrics,
                "strategy_id" : strategy.id,
                "start_date"  : result.get("start_date"),
                "end_date"    : result.get("end_date"),
                "trade_log"   : result.get("trade_log", []),
                "equity_curve": result.get("equity_curve", []),
                "passed"      : False,  # will be set by evaluator
            })

            self._log.info(
                f"Backtest SUCCESS: WR={metrics['win_rate']:.1%} "
                f"PF={metrics['profit_factor']:.2f} Ret={metrics['net_return_pct']:.1f}%",
                phase="BACKTEST", strategy_id=strategy.id,
            )
            return metrics
        except Exception as exc:
            self._log.exception(f"BacktestAgent exception: {exc}", phase="BACKTEST")
            return None

    async def run_batch(self, strategies: List[TradingStrategy]) -> List[Optional[Dict]]:
        """Parallel-process a batch of strategies."""
        results = self._btester.run(strategies)
        output  = []
        for res in results:
            if res and not res.get("error"):
                output.append(res.get("metrics"))
            else:
                output.append(None)
        return output

