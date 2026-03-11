"""
backtesting/performance_metrics.py
===================================
Rigorous strategy evaluation for professional quant research.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List
import numpy as np
from strategies.strategy_templates import PerformanceMetrics

class StrategyEvaluator:
    """
    Elite quantitative evaluation engine.
    """
    def __init__(self, config: Dict):
        perf = config.get("performance", {})
        # Stricter thresholds for realistic options trading
        self.min_win_rate       = perf.get("min_win_rate", 0.40) # Options selling can have lower WR if RR is high
        self.min_profit_factor  = perf.get("min_profit_factor", 1.4)
        self.min_return_pct     = perf.get("min_net_return_pct", 10.0) # Assume 1 month lookback
        self.max_drawdown_pct   = perf.get("max_drawdown_pct", 15.0)
        self.min_trade_sample   = perf.get("min_trade_sample", 20)
        self.min_sharpe         = perf.get("min_sharpe_ratio", 1.2)
        self.min_expectancy     = perf.get("min_expectancy", 0.5) # Points or percentage terms
        self.max_consec_losses  = perf.get("max_consecutive_losses", 6)

    def evaluate(self, metrics: PerformanceMetrics) -> Tuple[bool, str, float]:
        failures = []
        
        # ── Hard gates ──────────────────────────────────────────────────────
        if metrics.total_trades < self.min_trade_sample:
            failures.append(f"Statistically Insignificant: {metrics.total_trades} trades")
        
        if metrics.profit_factor < self.min_profit_factor:
            failures.append(f"PF {metrics.profit_factor} < {self.min_profit_factor}")
            
        if metrics.max_drawdown_pct > self.max_drawdown_pct:
            failures.append(f"DD {metrics.max_drawdown_pct}% > {self.max_drawdown_pct}%")
            
        if metrics.expectancy < self.min_expectancy:
            failures.append(f"Expectancy {metrics.expectancy} < {self.min_expectancy}")
            
        if metrics.win_rate < self.min_win_rate:
            failures.append(f"Win Rate {metrics.win_rate:.1%} < {self.min_win_rate:.1%}")

        passed = len(failures) == 0
        reason = "PASSED: Strategy meets survival criteria." if passed else "REJECTED: " + " | ".join(failures)
        score = self._composite_score(metrics)
        
        return passed, reason, score

    def _composite_score(self, m: PerformanceMetrics) -> float:
        """Weighted score [0-100] prioritizing consistency over raw return."""
        pf_score  = min(m.profit_factor / 3.0, 1.0) * 35
        sh_score  = min(m.sharpe_ratio / 2.5, 1.0) * 25
        ret_score = min(m.net_return_pct / 30.0, 1.0) * 20
        wr_score  = min(m.win_rate / 0.60, 1.0) * 10
        dd_pen    = max(0, 1.0 - m.max_drawdown_pct / self.max_drawdown_pct) * 10
        return round(pf_score + sh_score + ret_score + wr_score + dd_pen, 2)

    def improvement_hint(self, metrics: PerformanceMetrics) -> str:
        hints = []
        if metrics.win_rate < self.min_win_rate:
            hints.append("High false entry rate — strengthen regime filters or confirmation indicators.")
        if metrics.profit_factor < 1.0:
            hints.append("Inherent negative edge — consider structural spread changes (Credit vs Debit).")
        if metrics.max_drawdown_pct > self.max_drawdown_pct:
            hints.append("Excessive risk — tighten stop-loss or use hedged multi-leg structures.")
        if metrics.expectancy < 0:
            hints.append("Negative edge detected — re-evaluate the core entry/exit synchronization.")
            
        return " ".join(hints) if hints else "Marginal performance — optimize strike selection and timing."

class RiskFilter:
    @staticmethod
    def validate(strategy_dict: Dict, metrics: PerformanceMetrics) -> Tuple[bool, str]:
        # Implementation of multi-leg risk validation could go here
        return True, "Valid"

