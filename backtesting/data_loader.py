"""
backtesting/data_loader.py
==========================
Elite data loader for historical options research.
Loads full option chains and spot data from local CSVs.
"""

import os
import glob
import pandas as pd
import numpy as np
import random
from typing import Optional, Dict, Tuple

from utils.logger import setup_logger

logger = setup_logger("DataLoader")

BASE_DATA_DIR = "E:/utkarsh/Historical-data/data/index_data"

def get_data_directories(instrument: str) -> list[str]:
    """Map the instrument string to the local directories."""
    instrument = instrument.upper()
    if instrument == "NIFTY":
        return [os.path.join(BASE_DATA_DIR, "NIFTY_WEEKLY_DATA", "*"), os.path.join(BASE_DATA_DIR, "NIFTY MONTHLY DATA")]
    elif instrument == "BANKNIFTY":
        return [os.path.join(BASE_DATA_DIR, "BANKNIFTY DATA")]
    elif instrument == "FINNIFTY":
        return [os.path.join(BASE_DATA_DIR, "FINNIFTY DATA")]
    return []

def load_options_dataset(instrument: str, required_days: int = 30) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Loads both SPOT data and the full OPTION CHAIN data.
    Returns:
        (spot_df, chain_df)
        spot_df: MultiIndex [timestamp] -> [open, high, low, close, volume]

        chain_df: MultiIndex [timestamp, expiry, strike, type] -> [bid, ask, ltp, iv, delta, theta, etc.]
    """
    dirs = get_data_directories(instrument)
    all_files = []
    for d in dirs:
        # Recursive search for CSVs in case they are nested in year folders
        pattern = os.path.join(d, "**", "*.csv")
        all_files.extend(glob.glob(pattern, recursive=True))
    
    if not all_files:
        logger.error(f"No data files found for {instrument}")
        return None
        
    # For research, let's pick a larger segment of files
    # Each file roughly represents 1 month or 1 week.
    start_idx = max(0, len(all_files) - 5) 
    files_to_load = all_files[start_idx:]
    
    logger.info(f"Loading {len(files_to_load)} files for {instrument}...")
    
    full_df_list = []
    for f in files_to_load:
        try:
            df = pd.read_csv(f)
            full_df_list.append(df)
        except Exception as e:
            logger.error(f"Failed to load {f}: {e}")
            
    if not full_df_list: return None
    
    raw_df = pd.concat(full_df_list)
    raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'])
    
    # 1. Extract Spot Data
    spot_df = raw_df.groupby('timestamp').agg({
        'spot': 'first',
        'volume': 'sum' # total market volume proxy
    }).rename(columns={'spot': 'close'})
    
    # Generate OHLC for spot (simplified since we have 1min snapshots)
    spot_df['open'] = spot_df['close']
    spot_df['high'] = spot_df['close']
    spot_df['low'] = spot_df['close']
    spot_df = spot_df[['open', 'high', 'low', 'close', 'volume']]
    
    # 2. Extract Chain Data
    # Expected columns: timestamp, expiry_date, strike_price, option_type, bid, ask, close, iv, delta, gamma, theta, vega, volume, oi
    possible_expiry_cols = ['expiry_date', 'expiry']
    expiry_col = next((c for c in possible_expiry_cols if c in raw_df.columns), None)
    
    needed_cols = ['timestamp', 'strike_price', 'option_type', 'close', 'volume', 'oi', 'iv']
    if expiry_col: needed_cols.append(expiry_col)
    if 'strike_label' in raw_df.columns: needed_cols.append('strike_label')
    
    chain_df = raw_df[needed_cols].copy()
    if expiry_col:
        chain_df.rename(columns={expiry_col: 'expiry'}, inplace=True)
        chain_df['expiry'] = pd.to_datetime(chain_df['expiry'])
    else:
        # Fallback if no expiry col: assume weekly
        chain_df['expiry'] = chain_df['timestamp'] # dummy
    
    # Map 'CE'/'PE' to unified CE/PE
    type_map = {'CALL': 'CE', 'PUT': 'PE', 'CE': 'CE', 'PE': 'PE'}
    chain_df['option_type'] = chain_df['option_type'].astype(str).str.upper().map(type_map)
    
    # --- DATA CLEANING ---
    # Drop rows missing critical MultiIndex components or price data
    # Corrupted rows often have strike_price=NaN or close=NaN
    before_count = len(chain_df)
    chain_df.dropna(subset=['timestamp', 'expiry', 'strike_price', 'option_type', 'close'], inplace=True)
    
    # Ensure strike_price is numeric and positive
    chain_df['strike_price'] = pd.to_numeric(chain_df['strike_price'], errors='coerce')
    chain_df.dropna(subset=['strike_price'], inplace=True)
    chain_df = chain_df[chain_df['strike_price'] > 0]
    
    after_count = len(chain_df)
    if after_count < before_count:
        logger.warning(f"Cleaned {before_count - after_count} corrupted rows from option chain data.")

    # Set MultiIndex and SORT it for performance
    # Use ['timestamp', 'expiry', 'strike_price', 'option_type']
    chain_df.set_index(['timestamp', 'expiry', 'strike_price', 'option_type'], inplace=True)
    chain_df.sort_index(inplace=True)
    
    logger.info(f"Dataset loaded: {len(spot_df)} spot bars, {len(chain_df)} chain rows.")
    return spot_df, chain_df

def load_historical_ohlcv(instrument: str, timeframe: str, required_bars: int) -> Optional[np.ndarray]:
    """
    Legacy compatibility wrapper that returns just the spot OHLCV array.
    """
    dataset = load_options_dataset(instrument)
    if not dataset: return None
    spot_df, _ = dataset
    
    # Resampling code
    rule_map = {"1min": "1min", "5min": "5min", "15min": "15min", "1hour": "1h", "1day": "1D"}
    rule = rule_map.get(timeframe, "5min")
    
    resampled = pd.DataFrame()
    resampled['open'] = spot_df['open'].resample(rule).first()
    resampled['high'] = spot_df['close'].resample(rule).max()
    resampled['low'] = spot_df['close'].resample(rule).min()
    resampled['close'] = spot_df['close'].resample(rule).last()
    resampled['volume'] = spot_df['volume'].resample(rule).sum()
    resampled.dropna(inplace=True)
    
    if len(resampled) > required_bars:
        resampled = resampled.iloc[-required_bars:]
        
    timestamps_sec = resampled.index.astype('int64') // 10**9
    return np.column_stack([
        timestamps_sec,
        resampled['open'].values,
        resampled['high'].values,
        resampled['low'].values,
        resampled['close'].values,
        resampled['volume'].values
    ])

