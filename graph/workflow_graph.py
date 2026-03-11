"""
graph/workflow_graph.py
=======================
LangGraph-powered agent workflow for the autonomous trading system.

Graph topology (per-strategy):

  [start]
     │
     ▼
  INVENT_STRATEGY
     │
     ▼
  VALIDATE_STRATEGY  ──(invalid)──► LEARNING_UPDATE ──► [end]
     │(valid)
     ▼
  RUN_BACKTEST  ──(error)──► LEARNING_UPDATE ──► [end]
     │(success)
     ▼
  EVALUATE_METRICS  ──(rejected, mutations_left)──► MUTATE_STRATEGY ──┐
     │(accepted)                    │(mutation failed / max gen)       │
     ▼                              ▼                                  │
  RISK_FILTER       LEARNING_UPDATE ──► [end]                         │
     │(pass)                                                           │
     ▼                                                                 │
  STORE_STRATEGY ──► LEARNING_UPDATE ──► [end]                        │
                                                                       │
  ◄──────────────────────────────────────────────────────────────────┘
  (loop: mutated strategy → VALIDATE → BACKTEST → EVALUATE → ...)
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langgraph.graph import END, StateGraph

from agents.backtest_agent import BacktestAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.learning_agent import LearningAgent
from agents.risk_manager_agent import RiskManagerAgent
from agents.strategy_inventor import StrategyInventorAgent
from agents.strategy_mutator import StrategyMutatorAgent
from agents.strategy_validator import StrategyValidatorAgent
from database.repository import StrategyRepository, EvolutionRepository
from memory.mistake_memory import MistakeMemory
from strategies.strategy_templates import TradingStrategy
from utils.logger import AgentLogger


# ─────────────────────────────────────────────────────────────────────────────
# Graph State
# ─────────────────────────────────────────────────────────────────────────────

class TradingGraphState(TypedDict, total=False):
    # ── Inputs ─────────────────────────────────────────────────────────────
    existing_strategy_names : List[str]
    strategy_type_hint      : Optional[str]

    # ── Strategy being processed ────────────────────────────────────────────
    current_strategy        : Optional[Dict]   # TradingStrategy.to_dict()
    validation_issues       : List[str]
    backtest_metrics        : Optional[Dict]
    eval_passed             : bool
    eval_reason             : str
    eval_score              : float
    risk_passed             : bool
    risk_reason             : str

    # ── Mutation control ────────────────────────────────────────────────────
    mutation_attempts       : int
    max_mutation_attempts   : int
    improvement_hint        : str

    # ── Cycle tracking ──────────────────────────────────────────────────────
    cycle_stats             : Dict[str, int]
    final_status            : str   # ACCEPTED | REJECTED | ERROR


# ─────────────────────────────────────────────────────────────────────────────
# Node implementations
# ─────────────────────────────────────────────────────────────────────────────

MAX_MUTATION_ATTEMPTS = 2  # mutate rejected strategy up to 2 times


class TradingWorkflowGraph:
    """
    Wraps the LangGraph StateGraph and exposes a single `.run()` coroutine
    that processes one full strategy lifecycle.
    """

    def __init__(
        self,
        config  : Dict,
        memory  : MistakeMemory,
        search_tool=None,
    ):
        self.config   = config
        self.memory   = memory
        self._log     = AgentLogger("WorkflowGraph", config)
        self._search  = search_tool

        # ── Agent instances ─────────────────────────────────────────────────
        self.inventor  = StrategyInventorAgent(config, memory)
        self.mutator   = StrategyMutatorAgent(config, memory)
        self.validator = StrategyValidatorAgent(config, memory)
        self.backtester= BacktestAgent(config, memory)
        self.evaluator = EvaluatorAgent(config, memory)
        self.risk_mgr  = RiskManagerAgent(config, memory)
        self.learner   = LearningAgent(config, memory)

        # ── Repositories ────────────────────────────────────────────────────
        self._strat_repo = StrategyRepository(config)
        self._evo_repo   = EvolutionRepository()

        # ── Build graph ─────────────────────────────────────────────────────
        self._graph = self._build_graph()

    # ─── Node functions ──────────────────────────────────────────────────────

    async def _node_invent(self, state: TradingGraphState) -> TradingGraphState:
        """Invent a brand-new strategy."""
        strategy = await self.inventor.invent(
            existing_strategy_names = state.get("existing_strategy_names", []),
            search_tool             = self._search,
            strategy_type_hint      = state.get("strategy_type_hint"),
        )
        if strategy is None:
            self.learner.record_invention_failure("LLM returned no valid strategy")
            return {**state, "current_strategy": None, "final_status": "ERROR",
                    "cycle_stats": self._inc(state, "errors")}

        self._strat_repo.save_strategy(strategy.to_dict())
        self._evo_repo.record_evolution({
            "strategy_id": strategy.id, "operation": "INVENTED",
            "generation": 0, "inventor_notes": strategy.creator_notes,
        })
        return {
            **state,
            "current_strategy"  : strategy.to_dict(),
            "mutation_attempts" : 0,
            "max_mutation_attempts": MAX_MUTATION_ATTEMPTS,
        }

    async def _node_validate(self, state: TradingGraphState) -> TradingGraphState:
        """Validate strategy structure."""
        if not state.get("current_strategy"):
            return {**state, "final_status": "ERROR", "cycle_stats": self._inc(state, "errors")}

        strategy  = TradingStrategy.from_dict(state["current_strategy"])
        is_valid, issues = self.validator.validate(strategy)

        if not is_valid:
            self.learner.record_validation_failure(strategy.name, issues)
            self._strat_repo.reject_strategy(strategy.id, "; ".join(issues))
            return {
                **state,
                "validation_issues": issues,
                "final_status"     : "REJECTED",
                "cycle_stats"      : self._inc(state, "rejected"),
            }
        return {**state, "validation_issues": []}

    async def _node_backtest(self, state: TradingGraphState) -> TradingGraphState:
        """Run the backtest engine."""
        strategy = TradingStrategy.from_dict(state["current_strategy"])
        metrics  = await self.backtester.run_backtest(strategy)

        if metrics is None:
            # Check if it was a WF failure or something else via logs
            reject_reason = "Backtest failed or Walk-Forward validation failed"
            self.learner.record_backtest_failure(strategy.id, reject_reason)
            self._strat_repo.reject_strategy(strategy.id, reject_reason)
            return {
                **state,
                "backtest_metrics": None,
                "final_status"    : "REJECTED", # Changed from ERROR to REJECTED if it passed validation
                "cycle_stats"     : self._inc(state, "rejected"),
            }
        self._strat_repo.update_strategy_metrics(strategy.id, metrics)
        return {**state, "backtest_metrics": metrics}

    async def _node_evaluate(self, state: TradingGraphState) -> TradingGraphState:
        """Evaluate backtest metrics against acceptance thresholds."""
        strategy = TradingStrategy.from_dict(state["current_strategy"])
        metrics  = state["backtest_metrics"]
        passed, reason, score = self.evaluator.evaluate(strategy, metrics)

        hint = ""
        if not passed:
            hint = self.evaluator.improvement_hint(metrics, strategy.id)
            self.learner.record_evaluation_rejection(strategy, metrics, reason, hint)

        return {
            **state,
            "eval_passed"    : passed,
            "eval_reason"    : reason,
            "eval_score"     : score,
            "improvement_hint": hint,
        }

    async def _node_risk_filter(self, state: TradingGraphState) -> TradingGraphState:
        """Apply final risk gate."""
        strategy = TradingStrategy.from_dict(state["current_strategy"])
        metrics  = state["backtest_metrics"]
        passed, reason = self.risk_mgr.assess(strategy, metrics)

        if not passed:
            self.learner.record_risk_failure(strategy, reason)

        return {**state, "risk_passed": passed, "risk_reason": reason}

    async def _node_store(self, state: TradingGraphState) -> TradingGraphState:
        """Persist accepted strategy."""
        strategy = TradingStrategy.from_dict(state["current_strategy"])
        self._strat_repo.accept_strategy(strategy.id)
        self._log.info(
            f"✅ Strategy STORED: {strategy.name} (score={state.get('eval_score',0):.1f})",
            phase="STORE", strategy_id=strategy.id,
        )
        return {
            **state,
            "final_status": "ACCEPTED",
            "cycle_stats" : self._inc(state, "accepted"),
        }

    async def _node_mutate(self, state: TradingGraphState) -> TradingGraphState:
        """Mutate the current strategy to try to improve it."""
        attempts = state.get("mutation_attempts", 0)
        max_att  = state.get("max_mutation_attempts", MAX_MUTATION_ATTEMPTS)

        if attempts >= max_att:
            strategy = TradingStrategy.from_dict(state["current_strategy"])
            self._strat_repo.reject_strategy(strategy.id, f"Max mutations reached ({max_att})")
            return {
                **state,
                "final_status": "REJECTED",
                "cycle_stats" : self._inc(state, "rejected"),
            }

        parent  = TradingStrategy.from_dict(state["current_strategy"])
        metrics = state.get("backtest_metrics") or {}
        hint    = state.get("improvement_hint", "")

        child = await self.mutator.mutate(parent, metrics, hint)
        if child is None:
            self._strat_repo.reject_strategy(parent.id, "Mutation failed")
            return {
                **state,
                "final_status": "REJECTED",
                "cycle_stats" : self._inc(state, "rejected"),
            }

        self._strat_repo.save_strategy(child.to_dict())
        self._evo_repo.record_evolution({
            "strategy_id": child.id, "parent_id": parent.id,
            "operation": "MUTATED", "generation": child.generation,
            "inventor_notes": hint,
        })
        return {
            **state,
            "current_strategy"  : child.to_dict(),
            "mutation_attempts" : attempts + 1,
            "backtest_metrics"  : None,
        }

    async def _node_learning(self, state: TradingGraphState) -> TradingGraphState:
        """End-of-cycle learning update."""
        stats   = state.get("cycle_stats", {})
        summary = self.learner.generate_cycle_insights(stats)
        return {**state, "learning_summary": summary}

    # ─── Routing functions ────────────────────────────────────────────────────

    def _route_after_validate(self, state: TradingGraphState) -> str:
        if state.get("validation_issues") or state.get("final_status") in ("ERROR", "REJECTED"):
            return "learning"
        return "backtest"

    def _route_after_backtest(self, state: TradingGraphState) -> str:
        if state.get("backtest_metrics") is None:
            return "learning"
        return "evaluate"

    def _route_after_evaluate(self, state: TradingGraphState) -> str:
        if state.get("eval_passed"):
            return "risk_filter"
        # Can we still mutate?
        attempts = state.get("mutation_attempts", 0)
        max_att  = state.get("max_mutation_attempts", MAX_MUTATION_ATTEMPTS)
        if attempts < max_att:
            return "mutate"
        return "learning"

    def _route_after_risk(self, state: TradingGraphState) -> str:
        return "store" if state.get("risk_passed") else "learning"

    def _route_after_mutate(self, state: TradingGraphState) -> str:
        if state.get("final_status") in ("REJECTED", "ERROR"):
            return "learning"
        return "validate"  # mutated child goes through full pipeline again

    # ─── Graph construction ───────────────────────────────────────────────────

    def _build_graph(self) -> Any:
        g = StateGraph(TradingGraphState)

        g.add_node("invent",      self._node_invent)
        g.add_node("validate",    self._node_validate)
        g.add_node("backtest",    self._node_backtest)
        g.add_node("evaluate",    self._node_evaluate)
        g.add_node("risk_filter", self._node_risk_filter)
        g.add_node("store",       self._node_store)
        g.add_node("mutate",      self._node_mutate)
        g.add_node("learning",    self._node_learning)

        g.set_entry_point("invent")
        g.add_edge("invent", "validate")
        g.add_conditional_edges("validate",    self._route_after_validate,
                                {"backtest": "backtest", "learning": "learning"})
        g.add_conditional_edges("backtest",    self._route_after_backtest,
                                {"evaluate": "evaluate", "learning": "learning"})
        g.add_conditional_edges("evaluate",    self._route_after_evaluate,
                                {"risk_filter": "risk_filter", "mutate": "mutate", "learning": "learning"})
        g.add_conditional_edges("risk_filter", self._route_after_risk,
                                {"store": "store", "learning": "learning"})
        g.add_conditional_edges("mutate",      self._route_after_mutate,
                                {"validate": "validate", "learning": "learning"})
        g.add_edge("store",    "learning")
        g.add_edge("learning", END)

        return g.compile()

    # ─── Public API ───────────────────────────────────────────────────────────

    async def run(
        self,
        existing_names    : List[str]     = [],
        strategy_type_hint: Optional[str] = None,
    ) -> Dict:
        """
        Execute one full strategy lifecycle through the graph.
        Returns the final state dict.
        """
        initial_state: TradingGraphState = {
            "existing_strategy_names": existing_names,
            "strategy_type_hint"     : strategy_type_hint,
            "current_strategy"       : None,
            "validation_issues"      : [],
            "backtest_metrics"       : None,
            "eval_passed"            : False,
            "eval_reason"            : "",
            "eval_score"             : 0.0,
            "risk_passed"            : False,
            "risk_reason"            : "",
            "mutation_attempts"      : 0,
            "max_mutation_attempts"  : MAX_MUTATION_ATTEMPTS,
            "improvement_hint"       : "",
            "cycle_stats"            : {"accepted": 0, "rejected": 0, "errors": 0},
            "final_status"           : "",
        }
        final_state = await self._graph.ainvoke(initial_state)
        return final_state

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _inc(state: TradingGraphState, key: str) -> Dict[str, int]:
        stats = dict(state.get("cycle_stats", {}))
        stats[key] = stats.get(key, 0) + 1
        return stats
