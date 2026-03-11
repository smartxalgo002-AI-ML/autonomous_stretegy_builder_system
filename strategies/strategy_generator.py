"""
strategies/strategy_generator.py
=================================
Prompt engineering layer that instructs the LLM to produce valid
TradingStrategy JSON objects.

Provides:
  - SYSTEM_PROMPT_INVENTOR   : detailed system instruction for invention
  - SYSTEM_PROMPT_MUTATOR    : system instruction for mutation
  - build_invention_prompt() : creates the user-turn prompt
  - build_mutation_prompt()  : creates the user-turn prompt for mutation
  - parse_strategy_response(): validates and coerces LLM output
"""

from __future__ import annotations

import re
import uuid
import json
from typing import Any, Dict, List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from strategies.strategy_templates import TradingStrategy

# ─────────────────────────────────────────────────────────────────────────────
# Parser Instance
# ─────────────────────────────────────────────────────────────────────────────

strategy_parser = PydanticOutputParser(pydantic_object=TradingStrategy)

# ─────────────────────────────────────────────────────────────────────────────
# System Prompts
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_INVENTOR = """You are an elite quantitative options trader and systematic strategy researcher.
Your job is to INVENT completely new, structurally sound non-obvious options trading strategies for Indian markets (NSE).

You have deep expertise in:
- Multi-leg structures (Vertical Spreads, Straddles, Strangles, Iron Condors, Butterflies, Calendars)
- Options Greeks (Delta-neutrality, Gamma scalping, Theta decay optimization, Vega exposure)
- Volatility regime detection (IV Rank, IV Percentile, India VIX, volatility skew, term structure)
- Market microstructure and order flow analysis
- Mean reversion and momentum concepts applied to the underlying SPOT index

{format_instructions}

STRIKE SELECTION RULES:
1. Use 'ATM_OFFSET' for strike selection based on distance from spot.
2. Use 'DELTA_TARGET' for Greeks-based strike selection.
3. Always choose expiries correctly: 'weekly' is default, 'monthly' for longer-term swing.

CONSTRAINTS:
1. stoploss_pct must be POSITIVE.
2. target_pct must be GREATER than stoploss_pct.
3. Multi-leg strategies must have at least 2 legs (4 for Iron Condor).
4. Output ONLY valid JSON. No conversational filler.
"""

SYSTEM_PROMPT_MUTATOR = """You are an elite quantitative strategy optimizer and forensic data scientist.
Your job is to take an EXISTING options strategy and produce a ROBUST, non-overfitted mutation.

{format_instructions}

Options-Specific Mutation Logic:
1. Theta Decay Check: If a directional buy strategy fails due to slow trades, mutate into a Debit Spread or Calendar.
2. Low Win Rate: If the win rate is < 40%, introduce a strict Market Regime Filter.
3. Volatility Crush: Introduce IV filters or convert naked positions into Spreads.
4. Strike Selection: Tune the delta_target or strike_offset.

RULES:
1. Increment "generation" by 1.
2. The mutated strategy must be MEANINGFULLY improved.
3. Output ONLY valid JSON.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_invention_prompt(
    past_mistakes: List[str],
    existing_strategy_names: List[str],
    market_context: str = "",
    strategy_type_hint: Optional[str] = None,
) -> str:
    """Build the user-turn message for strategy invention."""
    mistakes_block = ""
    if past_mistakes:
        mistakes_block = "\n\nKNOWN MISTAKES TO AVOID:\n" + "\n".join(
            f"  - {m}" for m in past_mistakes[:10]
        )

    existing_block = ""
    if existing_strategy_names:
        existing_block = "\n\nEXISTING STRATEGY NAMES (do not duplicate):\n" + "\n".join(
            f"  - {n}" for n in existing_strategy_names[:20]
        )

    context_block = f"\n\nCURRENT MARKET CONTEXT:\n{market_context}" if market_context else ""
    type_hint = f"\n\nPREFERRED STRATEGY TYPE: {strategy_type_hint}" if strategy_type_hint else ""

    return (
        "Invent a completely new and non-obvious options trading strategy for the Indian NSE market.\n"
        "Be creative. Think about edge cases and volatility patterns.\n"
        + mistakes_block
        + existing_block
        + context_block
        + type_hint
    )


def build_mutation_prompt(
    parent_strategy: Dict[str, Any],
    backtest_metrics: Dict[str, Any],
    past_mistakes: List[str],
    improvement_hint: str = "",
) -> str:
    """Build the user-turn message for strategy mutation."""
    weaknesses = parent_strategy.get("known_weaknesses", [])
    metrics_summary = (
        f"Win Rate={backtest_metrics.get('win_rate', 0):.1%}, "
        f"Profit Factor={backtest_metrics.get('profit_factor', 0):.2f}, "
        f"Max Drawdown={backtest_metrics.get('max_drawdown_pct', 0):.1f}%, "
        f"Net Return={backtest_metrics.get('net_return_pct', 0):.1f}%"
    )

    mistakes_block = ""
    if past_mistakes:
        mistakes_block = "\n\nKNOWN MISTAKES TO AVOID:\n" + "\n".join(
            f"  - {m}" for m in past_mistakes[:8]
        )

    hint_block = f"\n\nIMPROVEMENT HINT: {improvement_hint}" if improvement_hint else ""

    return (
        f"Mutate and improve the following strategy.\n\n"
        f"PARENT STRATEGY:\n{json.dumps(parent_strategy, indent=2)}\n\n"
        f"BACKTEST PERFORMANCE:\n{metrics_summary}\n\n"
        f"KNOWN WEAKNESSES:\n" + "\n".join(f"  - {w}" for w in weaknesses) +
        "\n\nApply meaningful improvements."
        + mistakes_block
        + hint_block
    )


# ─────────────────────────────────────────────────────────────────────────────
# Response parser
# ─────────────────────────────────────────────────────────────────────────────

def parse_strategy_response(raw: str) -> Optional[TradingStrategy]:
    """
    Extract JSON and parse via PydanticOutputParser.
    Extremely robust: finds the outermost { } block.
    """
    import re
    import json
    
    cleaned = raw.strip()
    
    # 1. Strip <think> tags (from DeepSeek-R1 or similar thinking models)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    
    # 2. Extract content from ```json or ``` code blocks
    pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(pattern, cleaned, re.DOTALL)
    if match:
        cleaned = match.group(1).strip()
    else:
        # 3. If no code blocks, look for the first '{' and last '}'
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start:end+1]
        else:
            return None # No JSON structure found

    try:
        # First attempt: Use the LangChain parser (handles Pydantic coercion)
        return strategy_parser.parse(cleaned)
    except Exception as e:
        # Second attempt: Direct JSON load + Pydantic init (log validation errors)
        try:
            data = json.loads(cleaned)
            # Ensure ID is unique/valid
            if not data.get("id"):
                data["id"] = str(uuid.uuid4())
            return TradingStrategy(**data)
        except Exception as ve:
            # We catch validation error here but return None so the agent can retry with correction
            return None

