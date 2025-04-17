# strategies_custom.py - Trading strategies using custom indicators
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from custom_indicators import simple_moving_average, average_true_range, volume_sma, relative_volume, rsi

class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    All strategies should inherit from this base class.
    """
    
    @abstractmethod
    def get_timeframes(self):
        """
        Returns a list of timeframes required by the strategy.
        
        Returns:
            list: List of timeframe strings (e.g., ['1h', '4h'])
        """
        pass
    
    @abstractmethod
    def get_scan_interval(self):
        """
        Returns the scanning interval in seconds.
        
        Returns:
            int: Seconds to wait between scans
        """
        pass
    
    @abstractmethod
    def analyze(self, symbol, data):
        """
        Analyze the symbol data and return a signal if found.
        
        Args:
            symbol (str): Symbol being analyzed
            data (dict): Dictionary of DataFrames for each timeframe
            
        Returns:
            dict or None: Signal dict if a signal is found, None otherwise
        """
        pass

class SMACrossStrategy(BaseStrategy):
    """
    A simple moving average crossover strategy.
    Generates signals when the fast MA crosses above/below the slow MA.
    """
    
    def __init__(self, fast_ma=20, slow_ma=50, volume_ma=20, min_volume_factor=1.5):
        """
        Initialize the strategy parameters.
        
        Args:
            fast_ma (int): Fast moving average period
            slow_ma (int): Slow moving average period
            volume_ma (int): Volume moving average period
            min_volume_factor (float): Minimum volume factor compared to average
        """
        self.name = "SMA Crossover"
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
        self.volume_ma = volume_ma
        self.min_volume_factor = min_volume_factor
    
    def get_timeframes(self):
        """Required timeframes for this strategy."""
        return ['1h', '4h']
    
    def get_scan_interval(self):
        """Scanning interval in seconds."""
        return 300  # 5 minutes
    
    def analyze(self, symbol, data):
        """
        Analyze the data and return a signal if found.
        
        Args:
            symbol (str): Symbol being analyzed
            data (dict): Dictionary of DataFrames for each timeframe
            
        Returns:
            dict or None: Signal dict if a signal is found, None otherwise
        """
        # Get the 1h data for primary analysis
        df = data['1h'].copy()
        
        # Calculate indicators using custom functions
        df['fast_ma'] = simple_moving_average(df, self.fast_ma, 'close')
        df['slow_ma'] = simple_moving_average(df, self.slow_ma, 'close')
        df['volume_ma'] = volume_sma(df, self.volume_ma)
        
        # Calculate crossover
        df['crossover'] = np.zeros(len(df))
        
        # Detect crossovers
        for i in range(1, len(df)):
            if (np.isnan(df['fast_ma'].iloc[i-1]) or np.isnan(df['slow_ma'].iloc[i-1]) or 
                np.isnan(df['fast_ma'].iloc[i]) or np.isnan(df['slow_ma'].iloc[i])):
                continue
                
            if (df['fast_ma'].iloc[i-1] < df['slow_ma'].iloc[i-1] and 
                df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i]):
                df['crossover'].iloc[i] = 1  # Bullish crossover
            elif (df['fast_ma'].iloc[i-1] > df['slow_ma'].iloc[i-1] and 
                  df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i]):
                df['crossover'].iloc[i] = -1  # Bearish crossover
        
        # Get the latest candle
        latest_index = -1
        latest = df.iloc[latest_index]
        
        # Check for signals
        if latest['crossover'] != 0:
            # Check volume condition
            volume_condition = latest['volume'] > latest['volume_ma'] * self.min_volume_factor
            
            # Confirm with 4h trend
            df_4h = data['4h'].copy()
            df_4h['trend_ma'] = simple_moving_average(df_4h, 20, 'close')
            trend_condition = False
            
            if latest['crossover'] == 1:  # Bullish signal
                trend_condition = df_4h.iloc[-1]['close'] > df_4h.iloc[-1]['trend_ma']
                signal_type = 'BUY'
            else:  # Bearish signal
                trend_condition = df_4h.iloc[-1]['close'] < df_4h.iloc[-1]['trend_ma']
                signal_type = 'SELL'
            
            # Return signal if all conditions are met
            if volume_condition and trend_condition:
                return {
                    'symbol': symbol,
                    'type': signal_type,
                    'price': latest['close'],
                    'time': latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                    'timeframe': '1h',
                    'strategy': self.name,
                    'fast_ma': self.fast_ma,
                    'slow_ma': self.slow_ma
                }
        
        return None

class ATRBreakoutStrategy(BaseStrategy):
    """
    ATR-based breakout strategy.
    Generates signals when price breaks out of a range defined by ATR.
    """
    
    def __init__(self, atr_period=14, atr_multiplier=2.0, lookback=5):
        """
        Initialize the strategy parameters.
        
        Args:
            atr_period (int): Period for ATR calculation
            atr_multiplier (float): Multiplier for ATR to define breakout range
            lookback (int): Number of candles to look back for range
        """
        self.name = "ATR Breakout"
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.lookback = lookback
    
    def get_timeframes(self):
        """Required timeframes for this strategy."""
        return ['1h', '4h']
    
    def get_scan_interval(self):
        """Scanning interval in seconds."""
        return 300  # 5 minutes
    
    def analyze(self, symbol, data):
        """
        Analyze the data and return a signal if found.
        
        Args:
            symbol (str): Symbol being analyzed
            data (dict): Dictionary of DataFrames for each timeframe
            
        Returns:
            dict or None: Signal dict if a signal is found, None otherwise
        """
        # Get the 1h data for primary analysis
        df = data['1h'].copy()
        
        # Calculate ATR
        df['atr'] = average_true_range(df, self.atr_period)
        
        # Get the latest completed candle
        latest_index = -1
        latest = df.iloc[latest_index]
        
        # Need enough data for lookback
        if len(df) < self.lookback + 1:
            return None
        
        # Calculate range high and low
        lookback_data = df.iloc[-(self.lookback+1):-1]
        range_high = lookback_data['high'].max()
        range_low = lookback_data['low'].min()
        
        # Get ATR value
        atr_value = latest['atr']
        if np.isnan(atr_value):
            return None
        
        # Define breakout levels
        breakout_range = atr_value * self.atr_multiplier
        upper_breakout = range_high + breakout_range
        lower_breakout = range_low - breakout_range
        
        # Check for breakouts
        breakout_signal = None
        if latest['close'] > upper_breakout:
            # Bullish breakout
            breakout_signal = {
                'symbol': symbol,
                'type': 'BUY',
                'price': latest['close'],
                'time': latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'timeframe': '1h',
                'strategy': self.name,
                'atr': atr_value,
                'breakout_level': upper_breakout
            }
        elif latest['close'] < lower_breakout:
            # Bearish breakout
            breakout_signal = {
                'symbol': symbol,
                'type': 'SELL',
                'price': latest['close'],
                'time': latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'timeframe': '1h',
                'strategy': self.name,
                'atr': atr_value,
                'breakout_level': lower_breakout
            }
        
        # If we have a breakout, confirm with 4h trend
        if breakout_signal:
            df_4h = data['4h'].copy()
            # Calculate 20-period SMA for trend
            df_4h['trend_ma'] = simple_moving_average(df_4h, 20, 'close')
            
            # Check if breakout is aligned with larger timeframe trend
            if breakout_signal['type'] == 'BUY':
                if df_4h.iloc[-1]['close'] > df_4h.iloc[-1]['trend_ma']:
                    return breakout_signal
            else:  # SELL signal
                if df_4h.iloc[-1]['close'] < df_4h.iloc[-1]['trend_ma']:
                    return breakout_signal
        
        return None

class VolumeBreakoutStrategy(BaseStrategy):
    """
    Volume breakout strategy.
    Generates signals when volume spikes and price breaks out.
    """
    
    def __init__(self, volume_period=20, volume_threshold=2.0, price_lookback=5):
        """
        Initialize the strategy parameters.
        
        Args:
            volume_period (int): Period for volume SMA calculation
            volume_threshold (float): Volume threshold as multiplier of average volume
            price_lookback (int): Lookback period for price range
        """
        self.name = "Volume Breakout"
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.price_lookback = price_lookback
    
    def get_timeframes(self):
        """Required timeframes for this strategy."""
        return ['1h', '4h', '1d']
    
    def get_scan_interval(self):
        """Scanning interval in seconds."""
        return 300  # 5 minutes
    
    def analyze(self, symbol, data):
        """
        Analyze the data and return a signal if found.
        
        Args:
            symbol (str): Symbol being analyzed
            data (dict): Dictionary of DataFrames for each timeframe
            
        Returns:
            dict or None: Signal dict if a signal is found, None otherwise
        """
        # Get the hourly data for primary analysis
        df = data['1h'].copy()
        
        # Calculate volume indicators
        df['volume_sma'] = volume_sma(df, self.volume_period)
        df['rel_volume'] = relative_volume(df, self.volume_period)
        
        # Get the latest candle
        latest = df.iloc[-1]
        
        # Check if volume threshold is exceeded
        if latest['rel_volume'] < self.volume_threshold:
            return None
        
        # Need enough data for lookback
        if len(df) < self.price_lookback + 1:
            return None
        
        # Calculate price range from lookback period
        lookback_data = df.iloc[-(self.price_lookback+1):-1]
        range_high = lookback_data['high'].max()
        range_low = lookback_data['low'].min()
        
        # Check for price breakout
        breakout_signal = None
        if latest['close'] > range_high:
            # Bullish breakout
            breakout_signal = {
                'symbol': symbol,
                'type': 'BUY',
                'price': latest['close'],
                'time': latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'timeframe': '1h',
                'strategy': self.name,
                'rel_volume': latest['rel_volume'],
                'breakout_level': range_high
            }
        elif latest['close'] < range_low:
            # Bearish breakout
            breakout_signal = {
                'symbol': symbol,
                'type': 'SELL',
                'price': latest['close'],
                'time': latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                'timeframe': '1h',
                'strategy': self.name,
                'rel_volume': latest['rel_volume'],
                'breakout_level': range_low
            }
        
        # If we have a breakout, confirm with daily trend
        if breakout_signal:
            df_daily = data['1d'].copy()
            # Calculate 20-period SMA for trend
            df_daily['trend_ma'] = simple_moving_average(df_daily, 20, 'close')
            
            # Check if breakout is aligned with larger timeframe trend
            if breakout_signal['type'] == 'BUY':
                if df_daily.iloc[-1]['close'] > df_daily.iloc[-1]['trend_ma']:
                    return breakout_signal
            else:  # SELL signal
                if df_daily.iloc[-1]['close'] < df_daily.iloc[-1]['trend_ma']:
                    return breakout_signal
        
        return None

class RSIStrategy(BaseStrategy):
    """
    RSI-based strategy that looks for oversold/overbought conditions
    with trend confirmation using custom indicators.
    """
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        """
        Initialize the strategy parameters.
        
        Args:
            rsi_period (int): RSI period
            oversold (int): Oversold threshold
            overbought (int): Overbought threshold
        """
        self.name = "RSI Strategy"
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def get_timeframes(self):
        """Required timeframes for this strategy."""
        return ['1h', '4h', '1d']
    
    def get_scan_interval(self):
        """Scanning interval in seconds."""
        return 300  # 5 minutes
    
    def analyze(self, symbol, data):
        """
        Analyze the data and return a signal if found.
        
        Args:
            symbol (str): Symbol being analyzed
            data (dict): Dictionary of DataFrames for each timeframe
            
        Returns:
            dict or None: Signal dict if a signal is found, None otherwise
        """
        # Calculate RSI for all timeframes
        signals = {}
        
        for tf in self.get_timeframes():
            df = data[tf].copy()
            
            # Calculate RSI using custom function
            df['rsi'] = rsi(df, self.rsi_period, 'close')
            
            # Calculate simple moving average for trend
            df['sma50'] = simple_moving_average(df, 50, 'close')
            
            # Need at least 2 candles for crossover detection
            if len(df) < 2:
                continue
            
            # Get the latest and previous candles
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # Check if RSI values are valid
            if np.isnan(prev['rsi']) or np.isnan(latest['rsi']):
                continue
            
            # RSI crossing up through oversold
            if prev['rsi'] < self.oversold and latest['rsi'] > self.oversold:
                signals[tf] = {
                    'type': 'BUY',
                    'condition': 'Oversold',
                    'rsi': latest['rsi'],
                    'trend': latest['close'] > latest['sma50']
                }
            
            # RSI crossing down through overbought
            elif prev['rsi'] > self.overbought and latest['rsi'] < self.overbought:
                signals[tf] = {
                    'type': 'SELL',
                    'condition': 'Overbought',
                    'rsi': latest['rsi'],
                    'trend': latest['close'] < latest['sma50']
                }
        
        # Check for confluence across timeframes
        if '1h' in signals and '4h' in signals:
            # Both timeframes show the same signal type
            if signals['1h']['type'] == signals['4h']['type']:
                signal_type = signals['1h']['type']
                
                # Daily trend confirmation
                daily_df = data['1d'].copy()
                daily_df['sma50'] = simple_moving_average(daily_df, 50, 'close')
                daily_trend = data['1d'].iloc[-1]['close'] > daily_df['sma50'].iloc[-1]
                
                # Ensure trend alignment
                if (signal_type == 'BUY' and daily_trend) or (signal_type == 'SELL' and not daily_trend):
                    return {
                        'symbol': symbol,
                        'type': signal_type,
                        'price': data['1h'].iloc[-1]['close'],
                        'time': data['1h'].iloc[-1]['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'timeframe': '1h/4h',
                        'strategy': self.name,
                        'rsi_1h': signals['1h']['rsi'],
                        'rsi_4h': signals['4h']['rsi']
                    }
        
        return None

# Factory function to get strategy instance
def get_strategy(strategy_name="SMACrossStrategy"):
    """
    Factory function to get a strategy instance by name.
    
    Args:
        strategy_name (str): Name of the strategy class
        
    Returns:
        BaseStrategy: An instance of the requested strategy
    """
    strategies = {
        "SMACrossStrategy": SMACrossStrategy(),
        "ATRBreakoutStrategy": ATRBreakoutStrategy(),
        "VolumeBreakoutStrategy": VolumeBreakoutStrategy(),
        "RSIStrategy": RSIStrategy()
    }
    
    return strategies.get(strategy_name, SMACrossStrategy())