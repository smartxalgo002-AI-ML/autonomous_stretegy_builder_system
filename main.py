from __future__ import annotations
import argparse
import asyncio
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import yaml

from database.models import Base
from database.repository import Database, StrategyRepository, MistakeRepository
from graph.workflow_graph import TradingWorkflowGraph
from memory.mistake_memory import MistakeMemory
from tools.dhan_data_tool import build_dhan_tool
from tools.tavily_search_tool import build_tavily_tool
from utils.logger import setup_logger, AgentLogger

logger = setup_logger("Main")

def load_config(path: str = "config/config.yaml") -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        logger.error(f"Config file not found: {path}")
        sys.exit(1)
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

def initialise_system(config: Dict):
    log = AgentLogger("SystemInit", config)
    log_dir = config.get("logging", {}).get("log_dir", "./logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Database.init(config)
    log.info("Database initialised")
    return log

async def evolution_loop(config: Dict, max_cycles: int = -1, type_hint: Optional[str] = None):
    log = AgentLogger("EvolutionLoop", config)
    memory = MistakeMemory(config)
    search_tool = build_tavily_tool(config)
    graph = TradingWorkflowGraph(config, memory, search_tool=search_tool)
    strat_repo = StrategyRepository(config)
    cycle = 0
    _shutdown = False

    while not _shutdown:
        if max_cycles > 0 and cycle >= max_cycles: break
        cycle += 1
        log.info(f"CYCLE {cycle} START")
        existing_names = [s.name for s in strat_repo.get_top_strategies(n=50)]
        try:
            state = await graph.run(existing_names=existing_names, strategy_type_hint=type_hint)
            log.info(f"Cycle {cycle} complete. Status: {state.get('final_status')}")
        except Exception as e:
            log.exception(f"Error: {e}")
        if max_cycles == 1: break
        await asyncio.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--cycles", type=int, default=-1)
    parser.add_argument("--type", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    log = initialise_system(config)
    try:
        asyncio.run(evolution_loop(config, args.cycles, args.type))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
