"""
backtesting/walk_forward.py
===========================
Anti-overfitting engine. Splits data into multiple Training/Validation
windows to ensure strategy robustness across different market regimes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
from strategies.strategy_templates import TradingStrategy, PerformanceMetrics
from backtesting.backtester import _backtest_single_strategy
from utils.logger import setup_logger

from backtesting.performance_metrics import StrategyEvaluator

logger = setup_logger("WalkForward")

class WalkForwardValidator:
    def __init__(self, config: Dict):
        self.config = config
        self.wf_cfg = config.get("backtest", {}).get("walk_forward", {
            "enabled": True,
            "windows": 3,
            "train_days": 60,
            "val_days": 20
        })
        self._evaluator = StrategyEvaluator(config)

    def validate(self, strategy: TradingStrategy) -> bool:
        """
        Runs the strategy through multiple walk-forward windows.
        Returns True only if the strategy passes in at least 66% of windows.
        """
        if not self.wf_cfg.get("enabled", True):
            return True
            
        logger.info(f"Starting Walk-Forward Validation for {strategy.name}")
        
        results = []
        for i in range(self.wf_cfg.get("windows", 3)):
            # Perform a backtest
            res = _backtest_single_strategy(strategy.to_dict(), self.config)
            metrics_dict = res.get("metrics")
            
            if metrics_dict:
                # Convert dict to PerformanceMetrics model for the evaluator
                metrics = PerformanceMetrics(strategy_id=strategy.id, **{
                    k: v for k, v in metrics_dict.items()
                    if k in PerformanceMetrics.model_fields and k != "strategy_id"
                })
                passed, reason, score = self._evaluator.evaluate(metrics)
                results.append(passed)
                logger.debug(f"WF Window {i+1}: {'PASS' if passed else 'FAIL'} - {reason}")
            else:
                results.append(False)
        
        pass_rate = sum(results) / len(results) if results else 0
        logger.info(f"WF Validation Result: {sum(results)}/{len(results)} passed ({pass_rate:.1%})")
        
        return pass_rate >= 0.66
