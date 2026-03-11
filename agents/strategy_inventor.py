"""
agents/strategy_inventor.py
============================
Strategy Inventor Agent.

Responsibilities:
  - Use the LLM to generate completely new trading strategies
  - Consult mistake memory to avoid past failures
  - Optionally search the web for market context (Tavily)
  - Return a validated TradingStrategy object
"""

from __future__ import annotations

import json
import random
import uuid
import asyncio
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from memory.mistake_memory import MistakeMemory
from strategies.strategy_generator import (
    SYSTEM_PROMPT_INVENTOR,
    build_invention_prompt,
    strategy_parser,
    parse_strategy_response
)
from strategies.strategy_templates import StrategyType, TradingStrategy
from utils.logger import AgentLogger

STRATEGY_TYPES = [e.value for e in StrategyType]

class StrategyInventorAgent:
    """
    LLM-powered agent that invents new options trading strategies using LCEL and strict parsing.
    """

    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config  = config
        self.memory  = memory
        self._log    = AgentLogger("StrategyInventorAgent", config)
        llm_cfg      = config.get("llm", {})
        
        # Configure ChatOllama
        ollama_kwargs = {
            "model": llm_cfg.get("coder_model") or llm_cfg.get("model") or "deepseek-v3.1:671b-cloud",
            "temperature": llm_cfg.get("inventor_temperature", 0.7),
            "base_url": llm_cfg.get("base_url") or llm_cfg.get("ollama_base_url") or "http://localhost:11434",
        }
        
        api_key = config.get("api_keys", {}).get("ollama", "")
        if api_key:
            ollama_kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
            
        self._llm = ChatOllama(**ollama_kwargs)
        self._existing_names: List[str] = []

        # Define LCEL Chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_INVENTOR),
            ("human", "{user_prompt}"),
        ])
        # Note: We don't pipe to parser directly here if we want manual retry on raw
        self.chain = self.prompt | self._llm

    def _get_market_context(self, search_tool=None) -> str:
        if search_tool is None:
            return "NIFTY ~22,000, India VIX ~14. Mixed global sentiment."
        try:
            query = "NSE India options market outlook today"
            return search_tool._run(query)
        except:
            return "Standard market conditions."

    async def invent(
        self,
        existing_strategy_names: List[str],
        search_tool=None,
        strategy_type_hint: Optional[str] = None,
        retries: int = 2
    ) -> Optional[TradingStrategy]:
        """
        Generate a new strategy. Uses LCEL chain and robust parsing with retries.
        """
        self._log.info("Starting strategy invention…", phase="INVENTION")

        past_mistakes = self.memory.retrieve_relevant(
             "invalid schema target_pct stoploss_pct options legs", top_k=5
        )
        market_ctx = self._get_market_context(search_tool)

        if not strategy_type_hint and random.random() < 0.4:
            strategy_type_hint = random.choice(STRATEGY_TYPES)

        user_prompt = build_invention_prompt(
            past_mistakes=past_mistakes,
            existing_strategy_names=existing_strategy_names,
            market_context=market_ctx,
            strategy_type_hint=strategy_type_hint,
        )

        for attempt in range(retries + 1):
            try:
                self._log.debug(f"Invention attempt {attempt+1}/{retries+1}…", phase="INVENTION")
                
                # Execute LCEL chain
                # We inject format_instructions into the system prompt via variable
                response = await self.chain.ainvoke({
                    "user_prompt": user_prompt,
                    "format_instructions": strategy_parser.get_format_instructions()
                })
                
                raw_text = response.content
                strategy = parse_strategy_response(raw_text)

                if strategy:
                    # Final sanity fixes
                    strategy.id = str(uuid.uuid4())
                    self._log.info(f"Strategy invented: {strategy.name}", phase="INVENTION", strategy_id=strategy.id)
                    return strategy
                
                self._log.warning(f"Attempt {attempt+1} failed to parse JSON.", phase="INVENTION")
                self._log.debug(f"RAW OUTPUT (last 500 chars):\n{raw_text[-500:]}", phase="INVENTION")
                
                if attempt < retries:
                    # Adjust prompt for correction
                    user_prompt += "\n\nCRITICAL: Your previous output was not valid JSON. Ensure you return a single RFC-8259 compliant JSON object starting with '{' and ending with '}'. Use the EXACT schema provided. No markdown fillers outside the block."
                    await asyncio.sleep(2)

            except Exception as exc:
                self._log.error(f"Attempt {attempt+1} error: {exc}", phase="INVENTION")
                if attempt == retries:
                    self.memory.record(
                        agent_name="StrategyInventorAgent",
                        description=f"Invention failure: {exc}",
                        remedy="Tighten schema and reduce temperature.",
                        phase="INVENTION"
                    )
        
        return None

