# custom_indicators.py
# Implementation of common technical indicators without external TA libraries
import numpy as np
import pandas as pd
from functools import lru_cache


def simple_moving_average(data, period=14, column='close'):
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        period (int): Period for SMA calculation
        column (str): Column name to use for calculation
        
    Returns:
        np.array: Array containing SMA values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if column not in data.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    # Calculate SMA using pandas rolling window
    sma = data[column].rolling(window=period).mean().values
    
    return sma

def average_true_range(data, period=14):
    """
    Calculate Average True Range (ATR) using vectorized operations
    
    Args:
        data (pd.DataFrame): DataFrame containing OHLC data
        period (int): Period for ATR calculation
        
    Returns:
        np.array: Array containing ATR values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    required_columns = ['high', 'low', 'close']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain {required_columns} columns")
    
    # Get numpy arrays for calculation (faster than accessing DataFrame columns repeatedly)
    high = data['high'].values
    low = data['low'].values
    close = data['close'].values
    
    # Create shifted version of close
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]  # Set the first value to avoid NaN
    
    # Calculate the three differences all at once using numpy arrays
    tr1 = high - low                  # Current high - current low
    tr2 = np.abs(high - prev_close)   # Current high - previous close
    tr3 = np.abs(low - prev_close)    # Current low - previous close
    
    # Get true range as the max of the three differences
    # This vectorized approach is much faster than looping
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Allocate output array
    atr = np.full_like(tr, np.nan)
    
    # First ATR value is the average of the first period TR values
    atr[period-1] = np.mean(tr[:period])
    
    # Vectorized calculation of remaining ATR values
    # Wilder's smoothing formula: ATR = ((Prior ATR * (period-1)) + Current TR) / period
    for i in range(period, len(tr)):
        atr[i] = ((atr[i-1] * (period-1)) + tr[i]) / period
    
    return atr

def volume_sma(data, period=20):
    """
    Calculate Volume Simple Moving Average
    
    Args:
        data (pd.DataFrame): DataFrame containing volume data
        period (int): Period for SMA calculation
        
    Returns:
        np.array: Array containing Volume SMA values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if 'volume' not in data.columns:
        raise ValueError("DataFrame must contain a 'volume' column")
    
    # Calculate Volume SMA using pandas rolling window
    volume_sma = data['volume'].rolling(window=period).mean().values
    
    return volume_sma

def relative_volume(data, period=20):
    """
    Calculate Relative Volume (current volume / average volume)
    
    Args:
        data (pd.DataFrame): DataFrame containing volume data
        period (int): Period for average volume calculation
        
    Returns:
        np.array: Array containing Relative Volume values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if 'volume' not in data.columns:
        raise ValueError("DataFrame must contain a 'volume' column")
    
    # Calculate average volume
    avg_volume = volume_sma(data, period)
    
    # Calculate relative volume
    rel_volume = np.zeros_like(avg_volume)
    volume = data['volume'].values
    
    # Avoid division by zero
    for i in range(len(rel_volume)):
        if np.isnan(avg_volume[i]) or avg_volume[i] == 0:
            rel_volume[i] = np.nan
        else:
            rel_volume[i] = volume[i] / avg_volume[i]
    
    return rel_volume

def exponential_moving_average(data, period=20, column='close'):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        period (int): Period for EMA calculation
        column (str): Column name to use for calculation
        
    Returns:
        np.array: Array containing EMA values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if column not in data.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    # Calculate the multiplier
    multiplier = 2 / (period + 1)
    
    # Calculate EMA using pandas ewm
    ema = data[column].ewm(span=period, adjust=False).mean().values
    
    return ema

def rsi(data, period=14, column='close'):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        period (int): Period for RSI calculation
        column (str): Column name to use for calculation
        
    Returns:
        np.array: Array containing RSI values
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if column not in data.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    # Calculate price changes
    delta = np.zeros(len(data))
    values = data[column].values
    
    for i in range(1, len(values)):
        delta[i] = values[i] - values[i-1]
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0)
    losses = np.where(delta < 0, -delta, 0)
    
    # Initialize arrays for average gains and losses
    avg_gains = np.zeros_like(gains)
    avg_losses = np.zeros_like(losses)
    
    # First values are simple averages
    avg_gains[period] = np.mean(gains[1:period+1])
    avg_losses[period] = np.mean(losses[1:period+1])
    
    # Calculate subsequent values using the RSI formula
    for i in range(period+1, len(delta)):
        avg_gains[i] = ((avg_gains[i-1] * (period-1)) + gains[i]) / period
        avg_losses[i] = ((avg_losses[i-1] * (period-1)) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.zeros_like(avg_gains)
    rsi_values = np.zeros_like(avg_gains)
    
    for i in range(period, len(delta)):
        if avg_losses[i] == 0:
            rsi_values[i] = 100
        else:
            rs[i] = avg_gains[i] / avg_losses[i]
            rsi_values[i] = 100 - (100 / (1 + rs[i]))
    
    # Set initial values to NaN
    rsi_values[:period] = np.nan
    
    return rsi_values

def bollinger_bands(data, period=20, num_std_dev=2, column='close'):
    """
    Calculate Bollinger Bands
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        period (int): Period for SMA calculation
        num_std_dev (float): Number of standard deviations for bands
        column (str): Column name to use for calculation
        
    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if column not in data.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    # Calculate middle band (SMA)
    middle_band = simple_moving_average(data, period, column)
    
    # Calculate standard deviation
    rolling_std = data[column].rolling(window=period).std().values
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * num_std_dev)
    lower_band = middle_band - (rolling_std * num_std_dev)
    
    return middle_band, upper_band, lower_band

def moving_average_convergence_divergence(data, fast_period=12, slow_period=26, signal_period=9, column='close'):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        data (pd.DataFrame): DataFrame containing price data
        fast_period (int): Period for fast EMA
        slow_period (int): Period for slow EMA
        signal_period (int): Period for signal line
        column (str): Column name to use for calculation
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a pandas DataFrame")
    
    if column not in data.columns:
        raise ValueError(f"Column {column} not found in DataFrame")
    
    # Calculate fast and slow EMAs
    fast_ema = exponential_moving_average(data, fast_period, column)
    slow_ema = exponential_moving_average(data, slow_period, column)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Create a temporary DataFrame for signal line calculation
    temp_df = pd.DataFrame({'macd': macd_line})
    
    # Calculate signal line (EMA of MACD line)
    signal_line = temp_df['macd'].ewm(span=signal_period, adjust=False).mean().values
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

@lru_cache(maxsize=128)
def get_timeframe_seconds(timeframe):
    """Cached version of timeframe to seconds conversion"""
    timeframe_seconds = {
        '1m': 60,
        '3m': 3 * 60,
        '5m': 5 * 60,
        '15m': 15 * 60,
        '30m': 30 * 60,
        '1h': 60 * 60,
        '2h': 2 * 60 * 60,
        '4h': 4 * 60 * 60,
        '6h': 6 * 60 * 60,
        '8h': 8 * 60 * 60,
        '12h': 12 * 60 * 60,
        '1d': 24 * 60 * 60,
        '3d': 3 * 24 * 60 * 60,
        '1w': 7 * 24 * 60 * 60,
        '1M': 30 * 24 * 60 * 60,
    }
    
    return timeframe_seconds.get(timeframe, 60 * 60)  # Default to 1h