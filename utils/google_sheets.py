"""
utils/google_sheets.py
======================
Extended utility — appends rich strategy evaluation rows to Google Sheets.

Columns written (in order)
---------------------------
timestamp | strategy_id | strategy_name | symbol | decision | accuracy |
profit | drawdown | error_rate | trades | samples | active |
strategy_schema | indicators | entry_logic | exit_logic |
target | stoploss | trailing | timeframe | order_type | product_type |
strategy_type | tags | news_context | macro_context |
agent_reason | evaluation_result

Backward compatibility
-----------------------
The original 11-field callers from BacktestRepository continue to work as-is.
Any column not supplied in data_dict simply receives an empty string —
no existing call site needs to change.

Non-blocking
------------
Every write is dispatched on a daemon thread so the trading loop is never
stalled by a slow network round-trip.

Authentication
--------------
Service account key: keys.json (project root).
Only the Sheets API scope is required — Google Drive API is NOT needed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Configurable ──────────────────────────────────────────────────────────────
# Spreadsheet ID from URL: .../spreadsheets/d/<SPREADSHEET_ID>/edit
SPREADSHEET_ID: str = "1Qoq900Fxfj461nWQgzfPjLf16vQZF-7NmYVRqmalB1o"

# Path to service account key (relative to project root / cwd)
SERVICE_ACCOUNT_KEY_FILE: str = "keys.json"

# ── Full column schema (order defines sheet column positions) ─────────────────
SHEET_COLUMNS: List[str] = [
    # ── Core identity & timing ────────────────────────────────────────────────
    "timestamp",
    "strategy_id",
    "strategy_name",
    "symbol",
    # ── Agent decision ────────────────────────────────────────────────────────
    "decision",
    # ── Performance metrics ───────────────────────────────────────────────────
    "accuracy",        # maps from win_rate (backward compat alias)
    "profit",          # maps from net_return_pct / profit_factor
    "drawdown",        # maps from drawdown_pct / max_drawdown_pct
    "error_rate",      # 1 - win_rate; derived if not supplied
    "trades",          # total_trades
    "samples",         # alias for trades when not separately supplied
    "active",          # passed / accepted flag
    # ── Full strategy JSON ────────────────────────────────────────────────────
    "strategy_schema",
    # ── Flattened schema fields ───────────────────────────────────────────────
    "indicators",
    "entry_logic",
    "exit_logic",
    "target",
    "stoploss",
    "trailing",
    "timeframe",
    "order_type",
    "product_type",
    "strategy_type",
    "tags",
    # ── Context ───────────────────────────────────────────────────────────────
    "news_context",
    "macro_context",
    # ── Agent reasoning ───────────────────────────────────────────────────────
    "agent_reason",
    "evaluation_result",
]
# ─────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("google_sheets")

# ── Module-level sheet client cache (one auth per process) ────────────────────
_sheet_cache: Dict[str, Any] = {}
_cache_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_key_path() -> Optional[str]:
    """Return the absolute path to keys.json, or None if not found."""
    key_path = SERVICE_ACCOUNT_KEY_FILE
    if not os.path.isabs(key_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        key_path = os.path.join(project_root, SERVICE_ACCOUNT_KEY_FILE)
    return key_path if os.path.isfile(key_path) else None


def _get_sheet():
    """
    Return an authenticated gspread Worksheet, reusing a cached client.
    Thread-safe via _cache_lock.
    """
    with _cache_lock:
        if "sheet" in _sheet_cache:
            return _sheet_cache["sheet"]

        import gspread
        from google.oauth2 import service_account

        key_path = _resolve_key_path()
        if key_path is None:
            raise FileNotFoundError(f"Service account key '{SERVICE_ACCOUNT_KEY_FILE}' not found.")

        scopes = ["https://www.googleapis.com/auth/spreadsheets"]
        creds = service_account.Credentials.from_service_account_file(key_path, scopes=scopes)
        client = gspread.authorize(creds)
        sheet = client.open_by_key(SPREADSHEET_ID).sheet1
        _sheet_cache["sheet"] = sheet
        return sheet


def _ensure_headers(sheet) -> None:
    """
    If row 1 is empty or does not start with 'timestamp', write the full
    header row.  This is called once per write cycle and is cheap because
    gspread caches the cell values.
    """
    try:
        first_row = sheet.row_values(1)
        if first_row and first_row[0].strip().lower() == "timestamp":
            return  # headers already present
        sheet.insert_row(SHEET_COLUMNS, index=1, value_input_option="USER_ENTERED")
        logger.info("[GoogleSheets] Header row created with %d columns.", len(SHEET_COLUMNS))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[GoogleSheets] Could not verify/create headers: %s", exc)


def _safe_str(value: Any) -> str:
    """Convert any value to a sheet-safe string."""
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# Schema field extractor
# ─────────────────────────────────────────────────────────────────────────────

def _extract_schema_fields(schema: Any) -> Dict[str, str]:
    """
    Extract and flatten important fields from a raw strategy schema dict.

    The schema may come from different sources (AI agent output, backtester,
    legacy dict).  All lookups are defensive — missing paths return "".

    Expected structure (best-case):
    {
      "_id": ..., "name": ..., "type": ..., "timeframe": ...,
      "orderType": ..., "productType": ..., "tags": [...],
      "signalSource": {"underlyingSymbol": ...},
      "indicators": [{"type": ...}, ...],
      "execution": {
        "entry": {"rules": [...]},
        "exit":  {"rules": [...]}
      },
      "risk": {
        "target":   {"value": ...},
        "stoploss": {"value": ...},
        "trailing": {"value": ...}
      }
    }
    """
    if not isinstance(schema, dict):
        return {}

    def _get(*path, default=""):
        node = schema
        for key in path:
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node == default:
                return default
        return node if node is not None else default

    # ── indicators ─────────────────────────────────────────────────────────
    raw_indicators = schema.get("indicators", [])
    if isinstance(raw_indicators, list):
        indicators = ", ".join(
            str(ind.get("type", ind) if isinstance(ind, dict) else ind)
            for ind in raw_indicators
        )
    else:
        indicators = _safe_str(raw_indicators)

    # ── entry / exit rules ─────────────────────────────────────────────────
    entry_rules = _get("execution", "entry", "rules", default=[])
    exit_rules  = _get("execution", "exit",  "rules", default=[])

    def _rules_summary(rules) -> str:
        if isinstance(rules, list):
            return " | ".join(
                str(r.get("description", r) if isinstance(r, dict) else r)
                for r in rules
            )
        return _safe_str(rules)

    # ── tags ───────────────────────────────────────────────────────────────
    raw_tags = schema.get("tags", [])
    tags = ", ".join(str(t) for t in raw_tags) if isinstance(raw_tags, list) else _safe_str(raw_tags)

    return {
        "strategy_id"   : _safe_str(_get("_id") or _get("id")),
        "strategy_name" : _safe_str(_get("name")),
        "symbol"        : _safe_str(_get("signalSource", "underlyingSymbol")),
        "timeframe"     : _safe_str(_get("timeframe")),
        "order_type"    : _safe_str(_get("orderType")),
        "product_type"  : _safe_str(_get("productType")),
        "strategy_type" : _safe_str(_get("type")),
        "tags"          : tags,
        "indicators"    : indicators,
        "entry_logic"   : _rules_summary(entry_rules),
        "exit_logic"    : _rules_summary(exit_rules),
        "target"        : _safe_str(_get("risk", "target",   "value")),
        "stoploss"      : _safe_str(_get("risk", "stoploss", "value")),
        "trailing"      : _safe_str(_get("risk", "trailing", "value")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation result derivation
# ─────────────────────────────────────────────────────────────────────────────

def _derive_evaluation_result(decision: str, passed: Any) -> str:
    """
    Map decision + outcome to Win / Fail per business rules:

      BUY  + Target hit  (passed=True)  → Win
      BUY  + StopLoss hit(passed=False) → Fail
      HOLD + StopLoss hit(passed=False) → Win
      HOLD + Target hit  (passed=True)  → Fail
    """
    d = str(decision).strip().upper()
    p = bool(passed) if not isinstance(passed, bool) else passed
    if d == "BUY":
        return "Win" if p else "Fail"
    if d == "HOLD":
        return "Win" if not p else "Fail"
    return ""  # unknown decision — leave blank


# ─────────────────────────────────────────────────────────────────────────────
# Core write logic (runs inside a daemon thread)
# ─────────────────────────────────────────────────────────────────────────────

def _write_row(data_dict: Dict[str, Any]) -> None:
    """
    Resolve every column, ensure headers exist, and append the row.
    Runs in a daemon thread — exceptions are logged, never re-raised.
    """
    try:
        try:
            import gspread  # noqa: F401
            from google.oauth2 import service_account  # noqa: F401
        except ImportError as exc:
            logger.error(
                "[GoogleSheets] Missing dependency (%s). "
                "Install with: pip install gspread google-auth", exc
            )
            return

        # ── Get authenticated sheet ────────────────────────────────────────
        try:
            sheet = _get_sheet()
        except Exception as exc:
            logger.error("[GoogleSheets] Could not connect to sheet: %s", exc, exc_info=True)
            return

        # ── Auto-create headers if absent ──────────────────────────────────
        _ensure_headers(sheet)

        # ── Extract schema fields if strategy_schema supplied ──────────────
        raw_schema = data_dict.get("strategy_schema")
        schema_fields: Dict[str, str] = {}
        schema_json: str = ""

        if raw_schema is not None:
            if isinstance(raw_schema, str):
                try:
                    parsed = json.loads(raw_schema)
                    schema_fields = _extract_schema_fields(parsed)
                    schema_json = raw_schema
                except json.JSONDecodeError:
                    schema_json = raw_schema
            elif isinstance(raw_schema, dict):
                schema_fields = _extract_schema_fields(raw_schema)
                schema_json = json.dumps(raw_schema, ensure_ascii=False)

        # ── Resolve backward-compat aliases ───────────────────────────────
        # The original repository call uses: win_rate, profit_factor,
        # net_return_pct, max_drawdown_pct, passed, instrument.
        # Map them onto the new column names transparently.
        win_rate    = data_dict.get("win_rate",        data_dict.get("accuracy", ""))
        profit      = data_dict.get("net_return_pct",  data_dict.get("profit_factor", data_dict.get("profit", "")))
        drawdown    = data_dict.get("max_drawdown_pct",data_dict.get("drawdown_pct",  data_dict.get("drawdown", "")))
        trades      = data_dict.get("total_trades",    data_dict.get("trades", ""))
        passed      = data_dict.get("passed",          data_dict.get("active", ""))
        symbol      = data_dict.get("symbol",          data_dict.get("instrument", schema_fields.get("symbol", "")))

        # error_rate derived from win_rate when not explicitly given
        error_rate  = data_dict.get("error_rate", "")
        if error_rate == "" and win_rate != "":
            try:
                error_rate = round(1.0 - float(win_rate), 4)
            except (ValueError, TypeError):
                error_rate = ""

        decision    = str(data_dict.get("decision", "")).upper()
        agent_reason= data_dict.get("agent_reason", "")
        eval_result = data_dict.get("evaluation_result",
                                    _derive_evaluation_result(decision, passed) if decision else "")

        # ── Build final resolved dict ──────────────────────────────────────
        resolved: Dict[str, Any] = {
            # Identity
            "timestamp"        : data_dict.get("timestamp", datetime.utcnow().isoformat(timespec="seconds")),
            "strategy_id"      : data_dict.get("strategy_id",   schema_fields.get("strategy_id",   "")),
            "strategy_name"    : data_dict.get("strategy_name", schema_fields.get("strategy_name", "")),
            "symbol"           : symbol,
            # Agent decision
            "decision"         : decision,
            # Metrics
            "accuracy"         : win_rate,
            "profit"           : profit,
            "drawdown"         : drawdown,
            "error_rate"       : error_rate,
            "trades"           : trades,
            "samples"          : data_dict.get("samples", trades),
            "active"           : passed,
            # Schema
            "strategy_schema"  : schema_json,
            # Flattened schema (schema_fields win only if not directly in data_dict)
            "indicators"       : data_dict.get("indicators",    schema_fields.get("indicators",    "")),
            "entry_logic"      : data_dict.get("entry_logic",   schema_fields.get("entry_logic",   "")),
            "exit_logic"       : data_dict.get("exit_logic",    schema_fields.get("exit_logic",    "")),
            "target"           : data_dict.get("target",        schema_fields.get("target",        "")),
            "stoploss"         : data_dict.get("stoploss",      schema_fields.get("stoploss",      "")),
            "trailing"         : data_dict.get("trailing",      schema_fields.get("trailing",      "")),
            "timeframe"        : data_dict.get("timeframe",     schema_fields.get("timeframe",     "")),
            "order_type"       : data_dict.get("order_type",    schema_fields.get("order_type",    "")),
            "product_type"     : data_dict.get("product_type",  schema_fields.get("product_type",  "")),
            "strategy_type"    : data_dict.get("strategy_type", schema_fields.get("strategy_type", "")),
            "tags"             : data_dict.get("tags",          schema_fields.get("tags",          "")),
            # Context
            "news_context"     : data_dict.get("news_context",  ""),
            "macro_context"    : data_dict.get("macro_context", ""),
            # Agent reasoning
            "agent_reason"     : agent_reason,
            "evaluation_result": eval_result,
        }

        # ── Serialise to ordered list matching SHEET_COLUMNS ───────────────
        row = [_safe_str(resolved.get(col, "")) for col in SHEET_COLUMNS]

        sheet.append_row(row, value_input_option="USER_ENTERED")

        logger.info(
            "[GoogleSheets] Row appended — strategy_id=%s  decision=%s  result=%s",
            resolved.get("strategy_id", "?"),
            resolved.get("decision",    "?"),
            resolved.get("evaluation_result", "?"),
        )

    except Exception as exc:  # noqa: BLE001
        # Never crash the trading system due to a Sheets failure
        logger.error("[GoogleSheets] Failed to append row: %s", exc, exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (signature unchanged — all callers remain compatible)
# ─────────────────────────────────────────────────────────────────────────────

def append_to_google_sheet(data_dict: Dict[str, Any]) -> None:
    """
    Fire-and-forget: dispatch *data_dict* as a new sheet row on a daemon thread.

    Parameters
    ----------
    data_dict : dict
        Accepts ANY combination of old or new keys.  Unknown keys are ignored;
        missing keys default to an empty string.

    New keys (optional — all have defaults):
        strategy_schema  – raw strategy dict or JSON string; fields extracted automatically
        decision         – "BUY" or "HOLD"
        agent_reason     – free-form explanation from AI agent
        evaluation_result– "Win" or "Fail"; derived automatically if omitted
        news_context     – Tavily / news summary
        macro_context    – macro / geopolitical signal text
        samples          – number of samples (defaults to trades)
        indicators       – pre-formatted string (or extracted from schema)
        entry_logic      – pre-formatted string (or extracted from schema)
        exit_logic       – pre-formatted string (or extracted from schema)
        target/stoploss/trailing – numeric values (or extracted from schema)
        symbol           – underlying symbol (falls back to instrument)

    Existing keys (full backward compat):
        timestamp, strategy_id, strategy_name, instrument,
        win_rate, profit_factor, net_return_pct, max_drawdown_pct,
        drawdown_pct, sharpe_ratio, total_trades, trades, passed
    """
    t = threading.Thread(target=_write_row, args=(data_dict,), daemon=True)
    t.start()
