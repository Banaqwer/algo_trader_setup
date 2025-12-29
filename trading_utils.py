"""
Utility functions for trading operations.
Contains common calculations used across multiple modules.
"""

import pandas as pd


def calculate_spread_adjustment(entry_price: float, direction: int, 
                                spread_pips: float, instrument: str) -> tuple[float, float]:
    """Calculate spread-adjusted price and spread cost.
    
    Args:
        entry_price: Base entry price
        direction: 1 for long, -1 for short
        spread_pips: Spread in pips
        instrument: Instrument name (e.g., 'EUR_USD', 'USD_JPY')
    
    Returns:
        tuple: (adjusted_price, spread_cost)
    """
    # Calculate pip value based on instrument
    if instrument.endswith("JPY"):
        spread_price = spread_pips * 0.0001
    else:
        spread_price = spread_pips * 0.00001
    
    # Adjust price based on direction
    if direction > 0:  # Long position
        adjusted_price = entry_price + spread_price / 2
    else:  # Short position
        adjusted_price = entry_price - spread_price / 2
    
    spread_cost = abs(adjusted_price - entry_price)
    
    return adjusted_price, spread_cost


def calculate_slippage_adjustment(price: float, direction: int, 
                                   slippage_bps: float) -> tuple[float, float]:
    """Calculate slippage-adjusted price and slippage cost.
    
    Args:
        price: Current price
        direction: 1 for long, -1 for short
        slippage_bps: Slippage in basis points
    
    Returns:
        tuple: (adjusted_price, slippage_cost)
    """
    slippage_price = price * slippage_bps / 10000
    
    if direction > 0:  # Long position
        adjusted_price = price + slippage_price
    else:  # Short position
        adjusted_price = price - slippage_price
    
    slippage_cost = slippage_price
    
    return adjusted_price, slippage_cost


def detect_session(timestamp: pd.Timestamp) -> str:
    """Determine current trading session based on UTC time.
    
    Args:
        timestamp: Timestamp in UTC
    
    Returns:
        str: Session name ('Asia', 'London', 'NewYork', or 'Overlap')
    """
    hour = timestamp.hour
    if 22 <= hour or hour < 8:
        return "Asia"
    elif 8 <= hour < 17:
        return "London"
    elif 17 <= hour < 22:
        return "NewYork"
    else:
        return "Overlap"
