"""
test_sheets.py
==============
Smoke-test for google_sheets.py — run before starting the full system.

    python test_sheets.py

Tests:
  1. Backward-compat call (original 11-field dict from BacktestRepository)
  2. Rich call with strategy_schema + agent decision + context
"""

import time
from utils.google_sheets import append_to_google_sheet

# ── Test 1: Backward-compat (original repository call format) ─────────────────
print("▶ Test 1: backward-compat row (original 11 fields) …")
append_to_google_sheet({
    "timestamp"      : "2026-03-12T10:03:49",
    "strategy_id"    : "COMPAT-001",
    "strategy_name"  : "Legacy Backtest Row",
    "instrument"     : "NIFTY",          # old field — maps to symbol
    "win_rate"       : 0.62,             # maps to accuracy
    "profit_factor"  : 1.85,
    "net_return_pct" : 23.4,             # maps to profit
    "drawdown_pct"   : 8.1,             # maps to drawdown
    "sharpe_ratio"   : 1.42,
    "total_trades"   : 87,               # maps to trades
    "passed"         : True,             # maps to active
})

# Fire-and-forget — give the thread time to complete before test 2
time.sleep(3)

# ── Test 2: Rich call with full strategy schema + agent decision ───────────────
print("▶ Test 2: rich row (full schema + agent decision + context) …")
sample_schema = {
    "_id"          : "STG-2026-BANKNIFTY",
    "name"         : "BankNIFTY 9:20 Trend Confirmation",
    "type"         : "intraday_options",
    "timeframe"    : "5min",
    "orderType"    : "MARKET",
    "productType"  : "MIS",
    "tags"         : ["momentum", "trend-following", "options"],
    "signalSource" : {"underlyingSymbol": "BANKNIFTY"},
    "indicators"   : [
        {"type": "EMA", "period": 9},
        {"type": "RSI", "period": 14},
        {"type": "VWAP"},
    ],
    "execution": {
        "entry": {
            "rules": [
                {"description": "Price crosses above EMA9 after 9:20"},
                {"description": "RSI > 55 for momentum confirmation"},
                {"description": "Price above VWAP"},
            ]
        },
        "exit": {
            "rules": [
                {"description": "Target 1.5% or Stoploss 0.75%"},
                {"description": "Time exit at 15:00"},
            ]
        },
    },
    "risk": {
        "target"  : {"value": 1.5},
        "stoploss": {"value": 0.75},
        "trailing": {"value": 0.3},
    },
}

append_to_google_sheet({
    "timestamp"        : "2026-03-12T11:41:38",
    "strategy_id"      : "STG-2026-BANKNIFTY",
    "strategy_name"    : "BankNIFTY 9:20 Trend Confirmation",
    "symbol"           : "BANKNIFTY",
    # ── Performance ────────────────────────────────────────────────────────
    "win_rate"         : 0.71,
    "net_return_pct"   : 34.2,
    "max_drawdown_pct" : 5.3,
    "total_trades"     : 142,
    "passed"           : True,
    # ── Agent decision ─────────────────────────────────────────────────────
    "decision"         : "BUY",
    "agent_reason"     : "Strong uptrend confirmed by EMA cross + RSI > 60 + PCR < 0.8. Macro backdrop positive after RBI hold.",
    # ── Full strategy schema ───────────────────────────────────────────────
    "strategy_schema"  : sample_schema,   # dict — will be auto-serialised
    # ── Context ────────────────────────────────────────────────────────────
    "news_context"     : "RBI holds rates; FII inflow ₹2,400 Cr; Nifty reclaims 22,500.",
    "macro_context"    : "US CPI lower than expected; DXY weak; global risk-on sentiment.",
})

time.sleep(3)
print("✅ Done — check your Google Sheet for 2 new rows (+ auto-created headers if first run).")
