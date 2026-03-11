"""
tools/tavily_search_tool.py
===========================
LangChain-compatible Tavily internet-search tool.
Used by the StrategyInventorAgent to fetch market context and research.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional, Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from utils.logger import setup_logger

logger = setup_logger("TavilySearchTool")


class TavilySearchInput(BaseModel):
    query      : str = Field(..., description="Search query")
    max_results: int = Field(3, description="Maximum number of results to return")


class TavilySearchTool(BaseTool):
    """
    Search the internet for market news, strategy ideas, or research papers.
    Uses the Tavily API when configured, otherwise returns a stub.
    """
    name       : str = "tavily_search"
    description: str = (
        "Search the internet for current market news, volatility events, "
        "earnings calendars, options strategy research, or any market intelligence. "
        "Useful for getting real-world context before inventing strategies."
    )
    args_schema: Type[BaseModel] = TavilySearchInput

    api_key: str = ""

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str, max_results: int = 3, **kwargs) -> str:
        if not self.api_key or self.api_key == "YOUR_TAVILY_API_KEY":
            logger.warning("Tavily API key not configured — returning stub response")
            return self._stub_response(query)

        try:
            from tavily import TavilyClient  # type: ignore
            client  = TavilyClient(api_key=self.api_key)
            results = client.search(
                query          = query,
                max_results    = max_results,
                search_depth   = "advanced",
                include_answer = True,
            )
            formatted = {
                "query"  : query,
                "answer" : results.get("answer", ""),
                "results": [
                    {
                        "title"  : r.get("title", ""),
                        "url"    : r.get("url",   ""),
                        "snippet": r.get("content", "")[:500],
                    }
                    for r in results.get("results", [])[:max_results]
                ],
            }
            return json.dumps(formatted)
        except ImportError:
            logger.warning("tavily-python not installed — pip install tavily-python")
            return self._stub_response(query)
        except Exception as exc:
            logger.error(f"Tavily search error: {exc}")
            return self._stub_response(query)

    async def _arun(self, query: str, max_results: int = 3, **kwargs) -> str:
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self._run(query, max_results))

    @staticmethod
    def _stub_response(query: str) -> str:
        return json.dumps({
            "query"  : query,
            "answer" : "Tavily API not configured. Configure api_keys.tavily in config.yaml for live search.",
            "results": [
                {
                    "title"  : "Indian VIX Overview",
                    "url"    : "https://nseindia.com",
                    "snippet": "India VIX is currently in the 13-15 range indicating low volatility. "
                               "FII activity is mixed. Nifty near key resistance at 22,500.",
                },
                {
                    "title"  : "Options Strategy Research",
                    "url"    : "https://example.com",
                    "snippet": "Intraday options strategies perform best in high-volatility environments. "
                               "Consider delta-hedged approaches in trending markets.",
                },
            ],
            "source": "stub",
        })


def build_tavily_tool(config: Dict) -> TavilySearchTool:
    """Factory that reads Tavily API key from config."""
    api_key = config.get("api_keys", {}).get("tavily", "")
    return TavilySearchTool(api_key=api_key)
