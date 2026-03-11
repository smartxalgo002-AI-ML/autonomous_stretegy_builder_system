"""
database/models.py
==================
SQLAlchemy ORM models for the autonomous trading system.

Tables
------
  strategies         – accepted (and rejected) trading strategies
  backtest_results   – per-strategy backtest performance metrics
  agent_mistakes     – mistakes recorded by the self-learning agent
  strategy_evolution – lineage / mutation history between strategies
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, JSON, String, Text, ForeignKey, Index,
)
from sqlalchemy.orm import DeclarativeBase, relationship, backref


class Base(DeclarativeBase):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Core Strategy table
# ─────────────────────────────────────────────────────────────────────────────

class Strategy(Base):
    """
    Persists a generated trading strategy along with its current lifecycle
    state (PENDING → BACKTESTED → ACCEPTED / REJECTED).
    """
    __tablename__ = "strategies"

    id             = Column(String(36), primary_key=True)           # UUID
    name           = Column(String(256), nullable=False)
    description    = Column(Text, nullable=True)
    strategy_type  = Column(String(64), nullable=False)             # intraday_options, swing…
    instrument     = Column(String(32), nullable=False)             # NIFTY, BANKNIFTY…
    timeframe      = Column(String(16), nullable=False)             # 5min, 1day…
    status         = Column(String(32), default="PENDING")          # PENDING/ACCEPTED/REJECTED
    accepted       = Column(Boolean, default=False)

    # Full strategy spec stored as JSON for schema-free flexibility
    definition     = Column(JSON, nullable=False)

    # Scores / quick-access fields (denormalised from BacktestResult)
    win_rate       = Column(Float, nullable=True)
    profit_factor  = Column(Float, nullable=True)
    net_return_pct = Column(Float, nullable=True)
    max_drawdown   = Column(Float, nullable=True)
    sharpe_ratio   = Column(Float, nullable=True)
    total_trades   = Column(Integer, nullable=True)

    # Provenance
    parent_id      = Column(String(36), ForeignKey("strategies.id"), nullable=True)
    generation     = Column(Integer, default=0)
    rejection_reason = Column(Text, nullable=True)

    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    backtest_results = relationship("BacktestResult", back_populates="strategy", cascade="all, delete-orphan")
    children       = relationship("Strategy", backref=backref("parent", remote_side="Strategy.id"), foreign_keys=[parent_id])

    __table_args__ = (
        Index("ix_strategies_status",   "status"),
        Index("ix_strategies_accepted", "accepted"),
        Index("ix_strategies_type",     "strategy_type"),
    )

    def __repr__(self) -> str:
        return f"<Strategy id={self.id} name={self.name!r} status={self.status}>"


# ─────────────────────────────────────────────────────────────────────────────
# Backtest Results
# ─────────────────────────────────────────────────────────────────────────────

class BacktestResult(Base):
    """
    Stores the full performance metrics produced by the backtester for a
    specific strategy over a specific date range.
    """
    __tablename__ = "backtest_results"

    id               = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id      = Column(String(36), ForeignKey("strategies.id"), nullable=False)

    # Date range of the backtest
    start_date       = Column(DateTime, nullable=True)
    end_date         = Column(DateTime, nullable=True)

    # Core metrics
    total_trades     = Column(Integer,  default=0)
    winning_trades   = Column(Integer,  default=0)
    losing_trades    = Column(Integer,  default=0)
    win_rate         = Column(Float,    nullable=True)
    profit_factor    = Column(Float,    nullable=True)
    net_return_pct   = Column(Float,    nullable=True)
    max_drawdown_pct = Column(Float,    nullable=True)
    sharpe_ratio     = Column(Float,    nullable=True)
    sortino_ratio    = Column(Float,    nullable=True)
    calmar_ratio     = Column(Float,    nullable=True)
    avg_win_pct      = Column(Float,    nullable=True)
    avg_loss_pct     = Column(Float,    nullable=True)
    avg_holding_bars = Column(Float,    nullable=True)
    max_consec_losses= Column(Integer,  nullable=True)
    expectancy       = Column(Float,    nullable=True)

    # Full trade log stored as JSON array
    trade_log        = Column(JSON,     nullable=True)
    equity_curve     = Column(JSON,     nullable=True)    # list of portfolio values

    passed           = Column(Boolean,  default=False)
    created_at       = Column(DateTime, default=datetime.utcnow)

    # Relationship
    strategy         = relationship("Strategy", back_populates="backtest_results")

    __table_args__ = (
        Index("ix_bt_strategy_id", "strategy_id"),
        Index("ix_bt_passed",      "passed"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Mistakes (Self-Learning)
# ─────────────────────────────────────────────────────────────────────────────

class AgentMistake(Base):
    """
    Records every failure or sub-optimal decision made by any agent.
    The LearningAgent consults this table (and a vector index on top) before
    generating or mutating strategies to avoid repeating known errors.
    """
    __tablename__ = "agent_mistakes"

    id           = Column(Integer, primary_key=True, autoincrement=True)
    agent_name   = Column(String(128), nullable=False)
    phase        = Column(String(64),  nullable=True)   # INVENTION / VALIDATION / BACKTEST…
    error_type   = Column(String(128), nullable=True)   # ValueError / TimeoutError…
    description  = Column(Text,        nullable=False)  # human-readable summary
    context      = Column(JSON,        nullable=True)   # raw context (strategy_id, params…)
    remedy       = Column(Text,        nullable=True)   # what should be done differently
    strategy_id  = Column(String(36),  nullable=True)
    embedding    = Column(JSON,        nullable=True)   # vector embedding for similarity search
    created_at   = Column(DateTime,    default=datetime.utcnow)

    __table_args__ = (
        Index("ix_mistakes_agent", "agent_name"),
        Index("ix_mistakes_type",  "error_type"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Evolution Log
# ─────────────────────────────────────────────────────────────────────────────

class StrategyEvolution(Base):
    """
    Audit trail that records how each strategy was created or mutated.
    Enables reconstruction of the full evolutionary lineage.
    """
    __tablename__ = "strategy_evolution"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id    = Column(String(36), nullable=False)
    parent_id      = Column(String(36), nullable=True)
    generation     = Column(Integer, default=0)
    operation      = Column(String(64), nullable=False)   # INVENTED / MUTATED / CROSSED
    changes        = Column(JSON,  nullable=True)         # diff / description of changes
    inventor_notes = Column(Text,  nullable=True)         # LLM rationale
    created_at     = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_evo_strategy_id", "strategy_id"),
        Index("ix_evo_parent_id",   "parent_id"),
    )
