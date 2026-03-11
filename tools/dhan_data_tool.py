"""
tools/dhan_data_tool.py
=======================
LangChain-compatible tool wrapper for the Dhan API.
Fetches historical OHLCV and options chain data.

When API keys are not configured, the tool falls back to
synthetic data so the system can run in development mode.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type

import requests
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from utils.logger import setup_logger

logger = setup_logger("DhanDataTool")


# ─────────────────────────────────────────────────────────────────────────────
# Tool Input Schema
# ─────────────────────────────────────────────────────────────────────────────

class DhanHistoricalInput(BaseModel):
    symbol       : str = Field(..., description="Symbol e.g. NIFTY, BANKNIFTY")
    exchange_seg : str = Field("IDX_I", description="Exchange segment")
    instrument   : str = Field("INDEX", description="Instrument type")
    expiry_code  : int = Field(0, description="0 for index/equity")
    timeframe    : str = Field("5", description="Candle interval in minutes")
    from_date    : str = Field(..., description="From date YYYY-MM-DD")
    to_date      : str = Field(..., description="To date YYYY-MM-DD")


class DhanOptionsChainInput(BaseModel):
    symbol       : str = Field(..., description="Underlying symbol e.g. NIFTY")
    expiry_date  : str = Field(..., description="Expiry date YYYY-MM-DD")


# ─────────────────────────────────────────────────────────────────────────────
# Dhan API client
# ─────────────────────────────────────────────────────────────────────────────

DHAN_BASE_URL = "https://api.dhan.co"
DHAN_HIST_URL = f"{DHAN_BASE_URL}/charts/historical"
DHAN_OC_URL   = f"{DHAN_BASE_URL}/optionchain"

# Map timeframe string to Dhan resolution
TIMEFRAME_MAP = {
    "1":    "1",
    "5":    "5",
    "15":   "15",
    "25":   "25",
    "60":   "60",
    "1day": "D",
    "D":    "D",
}


class DhanAPIClient:
    """Low-level REST client for Dhan API with retry + fallback."""

    def __init__(self, client_id: str, access_token: str, max_retries: int = 3):
        self.client_id    = client_id
        self.access_token = access_token
        self.max_retries  = max_retries
        self._session     = requests.Session()
        self._session.headers.update({
            "access-token": access_token,
            "client-id"   : client_id,
            "Content-Type": "application/json",
        })

    def _post(self, url: str, payload: Dict) -> Optional[Dict]:
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.post(url, json=payload, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.warning(f"Dhan API attempt {attempt}/{self.max_retries} failed: {exc}")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
        return None

    def get_historical(
        self,
        security_id  : str,
        exchange_seg : str,
        instrument   : str,
        expiry_code  : int,
        timeframe    : str,
        from_date    : str,
        to_date      : str,
    ) -> Optional[Dict]:
        payload = {
            "securityId"  : security_id,
            "exchangeSegment": exchange_seg,
            "instrument"  : instrument,
            "expiryCode"  : expiry_code,
            "oi"          : "false",
            "fromDate"    : from_date,
            "toDate"      : to_date,
        }
        # Dhan separates intraday and daily endpoints
        if timeframe.upper() == "D":
            url = f"{DHAN_BASE_URL}/charts/historical"
            payload["resolution"] = "D"
        else:
            url = f"{DHAN_BASE_URL}/charts/intraday"
            payload["interval"] = timeframe
        return self._post(url, payload)


# ─────────────────────────────────────────────────────────────────────────────
# LangChain Tool
# ─────────────────────────────────────────────────────────────────────────────

class DhanHistoricalDataTool(BaseTool):
    """
    LangChain tool: fetch historical OHLCV from Dhan API.
    Falls back to synthetic data if API credentials are missing.
    """
    name       : str = "dhan_historical_data"
    description: str = (
        "Fetch historical OHLCV candle data for Indian NSE instruments "
        "(NIFTY, BANKNIFTY, etc.) from the Dhan API. "
        "Input: symbol, timeframe, from_date, to_date."
    )
    args_schema: Type[BaseModel] = DhanHistoricalInput

    # Injected at construction time
    client_id   : str = ""
    access_token: str = ""

    class Config:
        arbitrary_types_allowed = True

    def _run(
        self,
        symbol       : str,
        exchange_seg : str = "IDX_I",
        instrument   : str = "INDEX",
        expiry_code  : int = 0,
        timeframe    : str = "5",
        from_date    : str = "",
        to_date      : str = "",
        **kwargs
    ) -> str:
        if not self.client_id or not self.access_token:
            logger.warning("Dhan credentials not set — returning synthetic data")
            return self._synthetic_response(symbol, timeframe, from_date, to_date)

        try:
            client = DhanAPIClient(self.client_id, self.access_token)
            data   = client.get_historical(
                security_id  = self._symbol_to_id(symbol),
                exchange_seg = exchange_seg,
                instrument   = instrument,
                expiry_code  = expiry_code,
                timeframe    = TIMEFRAME_MAP.get(timeframe, timeframe),
                from_date    = from_date,
                to_date      = to_date,
            )
            if data:
                return json.dumps({"source": "dhan_api", "symbol": symbol, "data": data})
            else:
                logger.warning("Dhan API returned empty data — falling back to synthetic")
                return self._synthetic_response(symbol, timeframe, from_date, to_date)
        except Exception as exc:
            logger.error(f"DhanDataTool error: {exc}")
            return self._synthetic_response(symbol, timeframe, from_date, to_date)

    async def _arun(self, *args, **kwargs) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._run(*args, **kwargs))

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _symbol_to_id(symbol: str) -> str:
        """Map common index symbols to Dhan security IDs."""
        mapping = {
            "NIFTY"    : "13",
            "BANKNIFTY": "25",
            "FINNIFTY" : "27",
            "SENSEX"   : "1",
        }
        return mapping.get(symbol.upper(), symbol)

    @staticmethod
    def _synthetic_response(symbol: str, timeframe: str, from_date: str, to_date: str) -> str:
        """Return synthetic OHLCV data for testing/dev."""
        import numpy as np, random
        base = {"NIFTY": 22000, "BANKNIFTY": 48000, "FINNIFTY": 21000}.get(symbol.upper(), 22000)
        n    = 200
        np.random.seed(42)
        closes = base + np.cumsum(np.random.normal(0, base * 0.005, n))
        candles = []
        for i, c in enumerate(closes):
            o = c * (1 + np.random.uniform(-0.002, 0.002))
            h = max(o, c) * (1 + np.random.uniform(0, 0.003))
            l = min(o, c) * (1 - np.random.uniform(0, 0.003))
            v = int(np.random.lognormal(10, 0.5))
            candles.append({"open": round(o,2), "high": round(h,2), "low": round(l,2), "close": round(c,2), "volume": v})
        return json.dumps({
            "source" : "synthetic",
            "symbol" : symbol,
            "from"   : from_date,
            "to"     : to_date,
            "bars"   : candles,
            "note"   : "Synthetic data — configure Dhan API keys for live data",
        })


def build_dhan_tool(config: Dict) -> DhanHistoricalDataTool:
    """Factory function; reads credentials from config."""
    dhan_cfg = config.get("api_keys", {}).get("dhan", {})
    return DhanHistoricalDataTool(
        client_id    = dhan_cfg.get("client_id", ""),
        access_token = dhan_cfg.get("access_token", ""),
    )
