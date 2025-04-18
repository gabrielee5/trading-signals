# cache_manager.py
import time
import logging
import pandas as pd
from datetime import datetime

logger = logging.getLogger("trading_scanner")

class MarketDataCache:
    """
    Class to manage caching of market data to minimize API requests.
    Provides methods to store, retrieve, and manage cache freshness.
    """
    
    def __init__(self, max_cache_size_mb=200, max_age_hours=24):
        """
        Initialize the market data cache.
        
        Args:
            max_cache_size_mb (int): Maximum cache size in MB
            max_age_hours (int): Maximum age of cache entries in hours
        """
        self.cache = {}
        self.last_access = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'updates': 0
        }
        self.max_cache_size_mb = max_cache_size_mb
        self.max_age_hours = max_age_hours
    
    def get(self, key):
        """
        Get data from cache if available.
        
        Args:
            key (str): Cache key
            
        Returns:
            object or None: Cached data if available, None otherwise
        """
        if key in self.cache:
            # Update last access time
            self.last_access[key] = time.time()
            self.cache_stats['hits'] += 1
            return self.cache[key]
        
        self.cache_stats['misses'] += 1
        return None
    
    def set(self, key, data):
        """
        Store data in cache.
        
        Args:
            key (str): Cache key
            data (object): Data to cache
        """
        if key in self.cache:
            self.cache_stats['updates'] += 1
        
        self.cache[key] = data
        self.last_access[key] = time.time()
        
        # Check if we need to free up space
        self._manage_cache_size()
    
    def is_fresh(self, key, max_age_seconds):
        """
        Check if a cache entry is fresh.
        
        Args:
            key (str): Cache key
            max_age_seconds (int): Maximum age in seconds
            
        Returns:
            bool: True if cache entry exists and is fresh, False otherwise
        """
        if key not in self.last_access:
            return False
        
        current_time = time.time()
        age_seconds = current_time - self.last_access[key]
        
        return age_seconds <= max_age_seconds
    
    def get_timeframe_seconds(self, timeframe):
        """
        Convert a timeframe string to seconds.
        
        Args:
            timeframe (str): Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            int: Timeframe in seconds
        """
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
    
    def update_ohlcv_data(self, key, timeframe, new_data):
        """
        Update OHLCV data in cache by appending new candles and removing duplicates.
        
        Args:
            key (str): Cache key
            timeframe (str): Timeframe string
            new_data (pandas.DataFrame): New OHLCV data
            
        Returns:
            pandas.DataFrame: Updated OHLCV data
        """
        if key not in self.cache:
            self.set(key, new_data)
            return new_data
        
        # Get existing data
        existing_data = self.cache[key]
        
        # Combine data and remove duplicates
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data.drop_duplicates(subset=['timestamp'], keep='last')
        combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)
        
        # Update cache
        self.set(key, combined_data)
        
        return combined_data
    
    def should_update(self, key, timeframe):
        """
        Check if data for a key and timeframe should be updated.
        
        Args:
            key (str): Cache key
            timeframe (str): Timeframe string
            
        Returns:
            bool: True if data should be updated, False otherwise
        """
        # If not in cache, definitely update
        if key not in self.cache:
            return True
        
        # Get existing data
        data = self.cache[key]
        
        # If empty data, update
        if data.empty:
            return True
        
        # Get last timestamp
        last_timestamp = data['timestamp'].iloc[-1]
        
        # Convert to timestamp
        if isinstance(last_timestamp, pd.Timestamp):
            last_timestamp = last_timestamp.timestamp()
        else:
            # If it's already a datetime
            last_timestamp = datetime.fromtimestamp(last_timestamp / 1000.0).timestamp()
        
        # Calculate time since last candle
        time_since_last = time.time() - last_timestamp
        
        # Get timeframe in seconds
        timeframe_seconds = self.get_timeframe_seconds(timeframe)
        
        # Should update if more than one timeframe period has passed
        return time_since_last > timeframe_seconds
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            dict: Cache statistics
        """
        stats = self.cache_stats.copy()
        stats['size_mb'] = self._estimate_cache_size_mb()
        stats['entries'] = len(self.cache)
        
        # Calculate hit ratio
        total_requests = stats['hits'] + stats['misses']
        stats['hit_ratio'] = stats['hits'] / total_requests if total_requests > 0 else 0
        
        return stats
    
    def _manage_cache_size(self):
        """
        Manage cache size by removing least recently used items if needed.
        """
        # Estimate current cache size
        current_size_mb = self._estimate_cache_size_mb()
        
        if current_size_mb > self.max_cache_size_mb:
            # Sort keys by last access time (oldest first)
            keys_by_access = sorted(self.last_access.items(), key=lambda x: x[1])
            
            removed_count = 0
            # Remove oldest items until we're under the limit
            for key, _ in keys_by_access:
                if key in self.cache:
                    del self.cache[key]
                del self.last_access[key]
                removed_count += 1
                
                # Re-check size
                current_size_mb = self._estimate_cache_size_mb()
                if current_size_mb <= self.max_cache_size_mb:
                    break
            
            logger.info(f"Cache cleanup: removed {removed_count} entries, new size: {current_size_mb:.2f}MB")
    
    def _estimate_cache_size_mb(self):
        """
        Estimate the memory usage of the cache in MB.
        
        Returns:
            float: Estimated memory usage in MB
        """
        total_size = 0
        for df in self.cache.values():
            if isinstance(df, pd.DataFrame):
                # Use pandas memory_usage for DataFrames
                total_size += df.memory_usage(deep=True).sum()
            else:
                # Fallback for other objects
                import sys
                total_size += sys.getsizeof(df)
        
        # Convert bytes to MB
        return total_size / (1024 * 1024)
    
    def clean(self):
        """
        Remove old entries from the cache.
        
        Returns:
            int: Number of entries removed
        """
        current_time = time.time()
        max_age_seconds = self.max_age_hours * 3600
        keys_to_remove = []
        
        for key, last_access in list(self.last_access.items()):
            age_seconds = current_time - last_access
            if age_seconds > max_age_seconds:
                keys_to_remove.append(key)
        
        # Remove old cache entries
        for key in keys_to_remove:
            if key in self.cache:
                del self.cache[key]
            del self.last_access[key]
        
        if keys_to_remove:
            logger.info(f"Cache maintenance: removed {len(keys_to_remove)} old entries")
        
        return len(keys_to_remove)