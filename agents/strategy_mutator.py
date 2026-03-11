"""
agents/strategy_mutator.py
==========================
Strategy Mutation Agent — evolves existing strategies using LLM-driven
parameter and logic modifications.
"""

from __future__ import annotations

import uuid
import asyncio
from typing import Any, Dict, List, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from memory.mistake_memory import MistakeMemory
from strategies.strategy_generator import (
    SYSTEM_PROMPT_MUTATOR,
    build_mutation_prompt,
    strategy_parser,
    parse_strategy_response
)
from strategies.strategy_templates import TradingStrategy
from utils.logger import AgentLogger

class StrategyMutatorAgent:
    """
    Takes a parent strategy + its backtest metrics and produces an
    improved child strategy via LCEL-guided mutation.
    """

    def __init__(self, config: Dict, memory: MistakeMemory):
        self.config = config
        self.memory = memory
        self._log   = AgentLogger("StrategyMutatorAgent", config)
        llm_cfg     = config.get("llm", {})

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

        # Define LCEL Chain
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT_MUTATOR),
            ("human", "{user_prompt}"),
        ])
        self.chain = self.prompt | self._llm

    async def mutate(
        self,
        parent: TradingStrategy,
        backtest_metrics: Dict,
        improvement_hint: str = "",
        retries: int = 1
    ) -> Optional[TradingStrategy]:
        """
        Mutate *parent* strategy. Uses LCEL and robust parsing with retries.
        """
        self._log.info(
            f"Mutating: {parent.name} → Gen {parent.generation + 1}",
            phase="MUTATION", strategy_id=parent.id,
        )

        past_mistakes = self.memory.retrieve_relevant(
            "strategy mutation invalid JSON schema", top_k=5
        )

        user_prompt = build_mutation_prompt(
            parent_strategy  = parent.to_dict(),
            backtest_metrics = backtest_metrics,
            past_mistakes    = past_mistakes,
            improvement_hint = improvement_hint,
        )

        for attempt in range(retries + 1):
            try:
                self._log.debug(f"Mutation attempt {attempt+1}/{retries+1}…", phase="MUTATION")
                
                response = await self.chain.ainvoke({
                    "user_prompt": user_prompt,
                    "format_instructions": strategy_parser.get_format_instructions()
                })
                
                child = parse_strategy_response(response.content)

                if child:
                    # Lineage
                    child.id        = str(uuid.uuid4())
                    child.parent_id = parent.id
                    child.generation= parent.generation + 1
                    
                    self._log.info(f"Mutation complete: {child.name}", phase="MUTATION", strategy_id=child.id)
                    return child

                if attempt < retries:
                    user_prompt += "\n\nCRITICAL: Return ONLY valid JSON."
                    await asyncio.sleep(1)

            except Exception as exc:
                self._log.error(f"Mutation attempt {attempt+1} failed: {exc}", phase="MUTATION")

        return None

