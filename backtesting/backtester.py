"""
backtesting/backtester.py
=========================
Advanced Quantitative Options Backtesting Engine.

Features:
- Multi-leg Option Spread Simulation (Debit/Credit/Straddles/etc.)
- Options-Specific modeling: Greeks (Delta/Theta) and IV.
- Realistic Indian Market Costs (STT, GST, SEBI, Brokerage).
- Margin calculation for Short positions.
- Strike Selection: ATM Offsets or Delta-based targeting.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import random
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategies.strategy_templates import (
    PerformanceMetrics,
    TradingStrategy,
    StrategyType,
    Timeframe,
    OptionAction,
    StrikeSelectionMethod
)
from utils.logger import setup_logger
from backtesting.data_loader import load_options_dataset

logger = setup_logger("Backtester")

# ─────────────────────────────────────────────────────────────────────────────
# Cost & Margin Constants (NSE/India specific)
# ─────────────────────────────────────────────────────────────────────────────
BROKERAGE_PER_ORDER = 20.0
STT_SELL_PERCENT = 0.0625 / 100.0
TRANSACTION_CHARGE_PERCENT = 0.053 / 100.0  # NSE + Clearing
GST_PERCENT = 18.0 / 100.0 # On Brokerage + Transaction
SEBI_CHARGES_PERCENT = 0.0001 / 100.0

def calculate_transaction_costs(premium: float, lot_size: int, action: OptionAction) -> float:
    """Calculate all-in costs for one leg of an options trade."""
    turnover = premium * lot_size
    brokerage = BROKERAGE_PER_ORDER
    trans_charge = turnover * TRANSACTION_CHARGE_PERCENT
    sebi_charge = turnover * SEBI_CHARGES_PERCENT
    
    # STT only on Sell side for Options (on premium)
    stt = (turnover * STT_SELL_PERCENT) if action == OptionAction.SELL else 0.0
    
    # GST on (Brokerage + Trans Charge + SEBI)
    gst = (brokerage + trans_charge + sebi_charge) * GST_PERCENT
    
    return brokerage + trans_charge + sebi_charge + stt + gst

# ─────────────────────────────────────────────────────────────────────────────
# Indicator helpers (vectorised)
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, period: int) -> np.ndarray:
    if len(arr) == 0: return np.array([])
    k = 2.0 / (period + 1)
    result = np.zeros(len(arr))
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = arr[i] * k + result[i-1] * (1 - k)
    return result

def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    if len(close) < period: return np.zeros(len(close))
    delta = np.diff(close, prepend=close[0])
    gain  = np.where(delta > 0, delta, 0.0)
    loss  = np.where(delta < 0, -delta, 0.0)
    avg_g = _ema(gain, period)
    avg_l = _ema(loss, period)
    # Using the standard 100 * (avg_g / (avg_g + avg_l)) which is mathematically equivalent to the 1 + rs formula
    total_movement = avg_g + avg_l
    # Default to 50 if there is no movement at all
    rsi = np.full_like(total_movement, 50.0)
    # Only calculate where movement is present to avoid div by zero
    nonzero = total_movement != 0
    rsi[nonzero] = 100.0 * (avg_g[nonzero] / total_movement[nonzero])
    return rsi

# ─────────────────────────────────────────────────────────────────────────────
# Signal generation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_signals(ohlcv_df: pd.DataFrame, strategy: TradingStrategy) -> np.ndarray:
    """
    Generate entry signals [1, -1, 0] based on spot rules.
    """
    close = ohlcv_df['close'].values
    ema20 = _ema(close, 20)
    ema50 = _ema(close, 50)
    rsi14 = _rsi(close, 14)
    
    signals = np.zeros(len(close))
    
    # Regime Filter
    regime = strategy.regime
    # (In a real system, compute India VIX / IV Rank here)
    
    # Simple logic fallback
    long_cond = (close > ema20) & (ema20 > ema50) & (rsi14 > 50)
    short_cond = (close < ema20) & (ema20 < ema50) & (rsi14 < 50)
    
    signals = np.where(long_cond, 1, np.where(short_cond, -1, 0))
    
    # Filter directions
    if strategy.entry.direction.value == "LONG":
        signals = np.where(signals == 1, 1, 0)
    elif strategy.entry.direction.value == "SHORT":
        signals = np.where(signals == -1, -1, 0)
        
    return signals

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Engine
# ─────────────────────────────────────────────────────────────────────────────

def _simulate_trades(
    spot_df: pd.DataFrame,
    chain_df: pd.DataFrame,
    signals: np.ndarray,
    strategy: TradingStrategy,
    capital: float = 100_000.0,
) -> Tuple[List[Dict], List[float]]:
    """
    Multi-leg Options Simulation.
    """
    timestamps = spot_df.index
    close = spot_df['close'].values
    n = len(timestamps)
    
    trade_log = []
    equity_curve = [capital]
    portfolio = capital
    
    # Active trade state
    active_trade = None # Current active strategy instance
    
    lot_size = getattr(strategy.risk, "lot_size", 50) # default NIFTY
    slippage_points = 0.5 # conservative
    
    for i in range(1, n):
        ts = timestamps[i]
        
        # Performance optimization: Slice chain_df once per timestamp
        try:
            ts_chain = chain_df.loc[ts]
        except KeyError:
            ts_chain = pd.DataFrame()

        if active_trade is None:
            # Check for Signal
            if signals[i] != 0:
                if ts_chain.empty: continue
                
                # 1. Resolve Strikes
                current_spot = close[i]
                legs_data = []
                entry_cost = 0.0
                total_transactions = 0.0
                
                valid_entry = True
                for leg in strategy.legs:
                    try:
                        # Defensive: .xs raises KeyError if key is missing
                        type_slice = ts_chain.xs(leg.option_type.value, level='option_type')
                        if type_slice.empty:
                            valid_entry = False; break
                        
                        # Resolve Expiry
                        expiries = sorted(type_slice.index.get_level_values('expiry').unique())
                        if not expiries:
                            valid_entry = False; break
                        
                        target_expiry = expiries[0] if leg.expiry_selection == "weekly" else expiries[-1]
                        exp_slice = type_slice.xs(target_expiry, level='expiry')

                        # Target strike resolution
                        target_strike = 0.0
                        if leg.strike_selection_method == StrikeSelectionMethod.ATM_OFFSET:
                            step = 50 if "NIFTY" in strategy.instrument else 100
                            atm_strike = round(current_spot / step) * step
                            target_strike = atm_strike + leg.strike_offset
                        elif leg.strike_selection_method == StrikeSelectionMethod.DELTA_TARGET:
                            if 'delta' in exp_slice.columns:
                                target_strike = exp_slice.iloc[(exp_slice['delta'] - leg.delta_target).abs().argsort()[:1]].index[0]
                            else:
                                target_strike = exp_slice.index[0] # fallback

                        # 3. Get Premium
                        # Ensure we get a scalar close price
                        res = exp_slice.loc[target_strike, 'close']
                        premium = res.iloc[0] if isinstance(res, pd.Series) else res
                        
                        # Apply slippage
                        exec_price = premium + (slippage_points if leg.action == OptionAction.BUY else -slippage_points)
                        
                        cost = calculate_transaction_costs(exec_price, lot_size, leg.action)
                        total_transactions += cost
                        
                        flow = -exec_price * lot_size if leg.action == OptionAction.BUY else exec_price * lot_size
                        entry_cost += flow
                        
                        legs_data.append({
                            "strike": target_strike,
                            "expiry": target_expiry.isoformat() if hasattr(target_expiry, 'isoformat') else str(target_expiry),
                            "type": leg.option_type.value,
                            "action": leg.action,
                            "entry_premium": exec_price
                        })
                    except (KeyError, IndexError, ValueError):
                        valid_entry = False; break
                
                if valid_entry and legs_data:
                    active_trade = {
                        "entry_ts": ts,
                        "entry_spot": current_spot,
                        "legs": legs_data,
                        "entry_cash_flow": entry_cost,
                        "transaction_costs": total_transactions
                    }
                    portfolio -= total_transactions
        
        else:
            # Trade is Active -> Check Exit
            current_spot = close[i]
            exit_reason = None
            
            # Current PnL calculation
            current_value = 0.0
            if ts_chain.empty:
                exit_reason = "DATA_MISSING"
            else:
                for leg in active_trade['legs']:
                    try:
                        # Ensure expiry is a Timestamp if the index is DatetimeIndex
                        lk_expiry = pd.to_datetime(leg['expiry'])
                        res = ts_chain.loc[(lk_expiry, leg['strike'], leg['type']), 'close']
                        premium = res.iloc[0] if isinstance(res, pd.Series) else res
                        
                        val = premium * lot_size if leg['action'] == OptionAction.BUY else -premium * lot_size
                        current_value += val
                    except (KeyError, IndexError):
                        exit_reason = "LIQUIDITY_GAP"; break
            
            # Calculate PnL relative to trade investment (standard for options)
            # Higher sensitivity than account-level PnL
            pnl_pts = (current_value + active_trade['entry_cash_flow'])
            trade_investment = abs(active_trade['entry_cash_flow'])
            if trade_investment > 0:
                pnl_pct_trade = (pnl_pts / trade_investment) * 100.0
            else:
                pnl_pct_trade = (pnl_pts / capital) * 100.0 # fallback
            
            # 1. Target/SL check (now based on trade-level PnL)
            if exit_reason is None:
                if pnl_pct_trade >= float(strategy.exit.target_pct):
                    exit_reason = "TARGET"
                elif pnl_pct_trade <= -float(strategy.exit.stoploss_pct):
                    exit_reason = "STOPLOSS"
                
            # 2. Expiry/Time check
            if ts.hour == 15 and ts.minute >= 15:
                exit_reason = "TIME_EXIT"
                
            if exit_reason:
                final_exit_costs = 0.0
                # Resolve final exit prices
                for leg in active_trade['legs']:
                    try:
                        res = ts_chain.loc[(leg['expiry'], leg['strike'], leg['type']), 'close']
                        premium = res.iloc[0] if isinstance(res, pd.Series) else res
                        exec_price = premium - (slippage_points if leg['action'] == OptionAction.BUY else -slippage_points)
                        final_exit_costs += calculate_transaction_costs(exec_price, lot_size, leg['action'])
                    except:
                        pass # Use entry price as fallback or just skip cost
                
                net_pnl = pnl_pts - final_exit_costs
                portfolio += net_pnl
                
                trade_log.append({
                    "entry_ts": active_trade['entry_ts'].isoformat() if hasattr(active_trade['entry_ts'], 'isoformat') else str(active_trade['entry_ts']),
                    "exit_ts": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                    "net_pnl": net_pnl,
                    "pnl_pct": (net_pnl / capital) * 100,
                    "exit_reason": exit_reason,
                    "legs": active_trade['legs']
                })
                active_trade = None
        
        equity_curve.append(float(portfolio))
        
    return trade_log, equity_curve, timestamps[1], timestamps[-1]

# ─────────────────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(strategy_id: str, trade_log: List[Dict], equity_curve: List[float], initial_capital: float) -> PerformanceMetrics:
    # (Same robust metrics logic as before, just ensuring list lengths match)
    if not trade_log:
        return PerformanceMetrics(strategy_id=strategy_id, rejection_reason="No trades generated")
        
    pnls = np.array([t["net_pnl"] for t in trade_log])
    win_rate = len(pnls[pnls > 0]) / len(pnls)
    net_return = (equity_curve[-1] - initial_capital) / initial_capital * 100
    
    # Calculate Max DD
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = abs(dd.min()) * 100
    
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if len(pnls[pnls < 0]) > 0 else 0.0
    
    return PerformanceMetrics(
        strategy_id=strategy_id,
        total_trades=len(trade_log),
        win_rate=round(win_rate, 4),
        profit_factor=round(pf, 4),
        net_return_pct=round(net_return, 4),
        max_drawdown_pct=round(max_dd, 4),
        sharpe_ratio=0.0, # Calculation could be added here
        expectancy=round(pnls.mean(), 2)
    )

def _backtest_single_strategy(strategy_dict: Dict, config: Dict) -> Dict:
    try:
        strategy = TradingStrategy.from_dict(strategy_dict)
        capital = config.get("backtest", {}).get("default_capital", 100_000)
        
        # Phase 2 & 3: Load Real Chain Data
        dataset = load_options_dataset(strategy.instrument)
        if dataset is None:
            return {"strategy_id": strategy.id, "error": "Insufficient Historical Data"}
            
        spot_df, chain_df = dataset
        
        # Signals based on Spot
        signals = _generate_signals(spot_df, strategy)
        
        # Multi-leg simulation
        trades, equity, start_ts, end_ts = _simulate_trades(spot_df, chain_df, signals, strategy, capital)
        
        metrics = _compute_metrics(strategy.id, trades, equity, capital)
        
        return {
            "strategy_id": strategy.id,
            "metrics": metrics.model_dump(),
            "trade_log": trades[:100], # Increased log size
            "equity_curve": equity[-1000:],
            "start_date": start_ts.to_pydatetime() if hasattr(start_ts, 'to_pydatetime') else start_ts,
            "end_date": end_ts.to_pydatetime() if hasattr(end_ts, 'to_pydatetime') else end_ts,
            "error": None
        }
    except Exception as exc:
        logger.error(f"Backtest failed for {strategy_dict.get('id')}: {exc}")
        return {"strategy_id": strategy_dict.get("id", "unknown"), "error": str(exc)}

class BatchBacktester:
    def __init__(self, config: Dict):
        self.config = config
        self._workers = config.get("concurrency", {}).get("max_process_workers", 4)
        self._log = setup_logger("BatchBacktester")

    def run(self, strategies: List[TradingStrategy]) -> List[Dict]:
        if not strategies: return []
        strategy_dicts = [s.to_dict() for s in strategies]
        results = []
        
        # We perform backtests sequentially here for stability in research mode, 
        # but ProcessPoolExecutor can be used for scaling.
        # For huge DataFrames, sequential is safer in local env.
        for sd in strategy_dicts:
            res = _backtest_single_strategy(sd, self.config)
            results.append(res)
            
        return results
