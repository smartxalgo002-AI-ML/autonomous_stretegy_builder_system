"""
database/repository.py
======================
Data-access layer (Repository pattern) for all database operations.
All agents interact with the database through this module — never raw SQL.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Generator, List, Optional, Any

from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import Session, sessionmaker

from .models import (
    AgentMistake,
    BacktestResult,
    Base,
    Strategy,
    StrategyEvolution,
)
from utils.logger import AgentLogger


# ─────────────────────────────────────────────────────────────────────────────
# Database initialisation
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """
    Manages the SQLAlchemy engine and session factory.
    Call ``init(config)`` once at startup before using the repository.
    """

    _engine = None
    _SessionFactory = None

    @classmethod
    def init(cls, config: Dict) -> None:
        db_cfg = config.get("database", {})
        url     = db_cfg.get("url", "sqlite:///trading_system.db")
        echo    = db_cfg.get("echo", False)

        cls._engine = create_engine(
            url,
            echo=echo,
            connect_args={"check_same_thread": False} if "sqlite" in url else {},
            pool_pre_ping=True,
        )
        cls._SessionFactory = sessionmaker(bind=cls._engine, expire_on_commit=False)
        Base.metadata.create_all(cls._engine)

    @classmethod
    @contextmanager
    def session(cls) -> Generator[Session, None, None]:
        if cls._SessionFactory is None:
            raise RuntimeError("Database not initialised. Call Database.init(config) first.")
        session: Session = cls._SessionFactory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# ─────────────────────────────────────────────────────────────────────────────
# Repository helpers
# ─────────────────────────────────────────────────────────────────────────────

class StrategyRepository:
    """CRUD + query helpers for the Strategy table."""

    def __init__(self, config: Dict):
        self._log = AgentLogger("StrategyRepository", config)

    # ── Create ──────────────────────────────────────────────────────────────

    def save_strategy(self, strategy_dict: Dict) -> Strategy:
        """Persist a new strategy. Returns the saved ORM object."""
        with Database.session() as session:
            strategy = Strategy(
                id=strategy_dict.get("id", str(uuid.uuid4())),
                name=strategy_dict["name"],
                description=strategy_dict.get("description", ""),
                strategy_type=strategy_dict.get("strategy_type", "intraday_options"),
                instrument=strategy_dict.get("instrument", "NIFTY"),
                timeframe=strategy_dict.get("timeframe", "5min"),
                status="PENDING",
                definition=strategy_dict,
                generation=strategy_dict.get("generation", 0),
                parent_id=strategy_dict.get("parent_id"),
            )
            session.add(strategy)
            self._log.info(f"Saved strategy {strategy.id} – {strategy.name}")
            
            # Log the full schema for research and debugging visibility
            schema_json = json.dumps(strategy_dict, indent=2)
            self._log.info(
                f"\n[STRATEGY GENERATED]\n"
                f"Strategy ID: {strategy.id}\n"
                f"Strategy Name: {strategy.name}\n\n"
                f"Strategy Schema:\n{schema_json}"
            )
            return strategy

    # ── Read ────────────────────────────────────────────────────────────────

    def get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        with Database.session() as session:
            return session.get(Strategy, strategy_id)

    def get_accepted_strategies(self, limit: int = 50) -> List[Strategy]:
        with Database.session() as session:
            return (
                session.query(Strategy)
                .filter(Strategy.accepted == True)
                .order_by(desc(Strategy.profit_factor))
                .limit(limit)
                .all()
            )

    def get_top_strategies(self, n: int = 10) -> List[Strategy]:
        """Return top N strategies sorted by profit factor."""
        with Database.session() as session:
            return (
                session.query(Strategy)
                .filter(Strategy.accepted == True)
                .order_by(desc(Strategy.profit_factor))
                .limit(n)
                .all()
            )

    def get_pending_strategies(self) -> List[Strategy]:
        with Database.session() as session:
            return (
                session.query(Strategy)
                .filter(Strategy.status == "PENDING")
                .all()
            )

    def count_strategies(self, status: Optional[str] = None) -> int:
        with Database.session() as session:
            q = session.query(func.count(Strategy.id))
            if status:
                q = q.filter(Strategy.status == status)
            return q.scalar()

    # ── Update ──────────────────────────────────────────────────────────────

    def update_strategy_metrics(self, strategy_id: str, metrics: Dict) -> None:
        with Database.session() as session:
            strategy = session.get(Strategy, strategy_id)
            if strategy:
                strategy.win_rate       = metrics.get("win_rate")
                strategy.profit_factor  = metrics.get("profit_factor")
                strategy.net_return_pct = metrics.get("net_return_pct")
                strategy.max_drawdown   = metrics.get("max_drawdown_pct")
                strategy.sharpe_ratio   = metrics.get("sharpe_ratio")
                strategy.total_trades   = metrics.get("total_trades")
                strategy.updated_at     = datetime.utcnow()

    def accept_strategy(self, strategy_id: str) -> None:
        with Database.session() as session:
            strategy = session.get(Strategy, strategy_id)
            if strategy:
                strategy.status   = "ACCEPTED"
                strategy.accepted = True
                strategy.updated_at = datetime.utcnow()
                self._log.info(f"Strategy {strategy_id} ACCEPTED ✓")

    def reject_strategy(self, strategy_id: str, reason: str = "") -> None:
        with Database.session() as session:
            strategy = session.get(Strategy, strategy_id)
            if strategy:
                strategy.status           = "REJECTED"
                strategy.accepted         = False
                strategy.rejection_reason = reason
                strategy.updated_at       = datetime.utcnow()
                self._log.info(f"Strategy {strategy_id} REJECTED – {reason}")

    # ── Delete ──────────────────────────────────────────────────────────────

    def delete_strategy(self, strategy_id: str) -> None:
        with Database.session() as session:
            strategy = session.get(Strategy, strategy_id)
            if strategy:
                session.delete(strategy)


class BacktestRepository:
    """CRUD helpers for BacktestResult."""

    def save_result(self, result: Dict) -> BacktestResult:
        with Database.session() as session:
            bt = BacktestResult(
                strategy_id      = result["strategy_id"],
                start_date       = result.get("start_date"),
                end_date         = result.get("end_date"),
                total_trades     = result.get("total_trades", 0),
                winning_trades   = result.get("winning_trades", 0),
                losing_trades    = result.get("losing_trades", 0),
                win_rate         = result.get("win_rate"),
                profit_factor    = result.get("profit_factor"),
                net_return_pct   = result.get("net_return_pct"),
                max_drawdown_pct = result.get("max_drawdown_pct"),
                sharpe_ratio     = result.get("sharpe_ratio"),
                sortino_ratio    = result.get("sortino_ratio"),
                calmar_ratio     = result.get("calmar_ratio"),
                avg_win_pct      = result.get("avg_win_pct"),
                avg_loss_pct     = result.get("avg_loss_pct"),
                avg_holding_bars = result.get("avg_holding_bars"),
                max_consec_losses= result.get("max_consec_losses"),
                expectancy       = result.get("expectancy"),
                trade_log        = result.get("trade_log", []),
                equity_curve     = result.get("equity_curve", []),
                passed           = result.get("passed", False),
            )
            session.add(bt)
            return bt
    def mark_passed(self, strategy_id: str, passed: bool) -> None:
        with Database.session() as session:
            bt = (
                session.query(BacktestResult)
                .filter(BacktestResult.strategy_id == strategy_id)
                .order_by(desc(BacktestResult.created_at))
                .first()
            )
            if bt:
                bt.passed = passed

    def get_results_for_strategy(self, strategy_id: str) -> List[BacktestResult]:
        with Database.session() as session:
            return (
                session.query(BacktestResult)
                .filter(BacktestResult.strategy_id == strategy_id)
                .order_by(desc(BacktestResult.created_at))
                .all()
            )


class MistakeRepository:
    """CRUD helpers for AgentMistake (self-learning memory)."""

    def record_mistake(self, mistake: Dict) -> AgentMistake:
        with Database.session() as session:
            m = AgentMistake(
                agent_name  = mistake.get("agent_name", "unknown"),
                phase       = mistake.get("phase"),
                error_type  = mistake.get("error_type"),
                description = mistake["description"],
                context     = mistake.get("context"),
                remedy      = mistake.get("remedy"),
                strategy_id = mistake.get("strategy_id"),
                embedding   = mistake.get("embedding"),
            )
            session.add(m)
            return m

    def get_recent_mistakes(self, limit: int = 50) -> List[AgentMistake]:
        with Database.session() as session:
            return (
                session.query(AgentMistake)
                .order_by(desc(AgentMistake.created_at))
                .limit(limit)
                .all()
            )

    def get_mistakes_by_agent(self, agent_name: str) -> List[AgentMistake]:
        with Database.session() as session:
            return (
                session.query(AgentMistake)
                .filter(AgentMistake.agent_name == agent_name)
                .order_by(desc(AgentMistake.created_at))
                .all()
            )


class EvolutionRepository:
    """Audit trail for strategy lineage."""

    def record_evolution(self, record: Dict) -> StrategyEvolution:
        with Database.session() as session:
            evo = StrategyEvolution(
                strategy_id    = record["strategy_id"],
                parent_id      = record.get("parent_id"),
                generation     = record.get("generation", 0),
                operation      = record.get("operation", "INVENTED"),
                changes        = record.get("changes"),
                inventor_notes = record.get("inventor_notes"),
            )
            session.add(evo)
            return evo
