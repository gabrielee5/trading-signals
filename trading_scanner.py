#!/usr/bin/env python3
# trading_scanner.py - Main file for the crypto trading signal scanner

import json
import time
import logging
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import telegram
from strategies_custom import get_strategy
from cache_manager import MarketDataCache

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("trading_scanner")

class TradingScanner:
    """
    A class that continuously scans the market and generates trading signals
    based on a specified strategy.
    """
    
    def __init__(self, config_path="config.json"):
        """
        Initialize the trading scanner with the provided configuration.
        
        Args:
            config_path (str): Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.exchange_config = self.config["exchange"]
        self.pairlist_config = self.config["pairlists"][0]
        self.telegram_config = self.config["telegram"]
        
        # Get cache configuration if available
        cache_config = self.config.get("cache", {})
        max_cache_size_mb = cache_config.get("max_size_mb", 300)
        max_age_hours = cache_config.get("max_age_hours", 48)
        
        # Initialize exchange
        exchange_name = self.exchange_config["name"]
        self.exchange = self._init_exchange(exchange_name)
        
        # Set up pairs to monitor
        self.pairs = []
        
        # Initialize telegram bot if enabled
        self.telegram_bot = None
        if self.telegram_config["enabled"]:
            self._init_telegram()
        
        # Load strategy
        strategy_name = self.config.get("strategy", "SMACrossStrategy")
        self.strategy = get_strategy(strategy_name)
        logger.info(f"Using strategy: {self.strategy.__class__.__name__}")
        
        # Initialize data cache manager
        self.market_cache = MarketDataCache(max_cache_size_mb=max_cache_size_mb, max_age_hours=max_age_hours)
        logger.info(f"Initialized market data cache (max size: {max_cache_size_mb}MB, max age: {max_age_hours}h)")
        
        # Running flag for continuous operation
        self.running = False
        
        # Keep track of last cache statistics logged
        self.last_cache_stats_log = time.time()
    
    def _load_config(self, config_path):
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}")
            raise
    
    def _init_exchange(self, exchange_name):
        """Initialize exchange connection."""
        try:
            exchange_class = getattr(ccxt, exchange_name)
            exchange = exchange_class({
                'apiKey': self.exchange_config.get('key'),
                'secret': self.exchange_config.get('secret'),
                'enableRateLimit': True,
                'options': {'defaultType': self.exchange_config.get('asset_type', 'spot')}
            })
            logger.info(f"Connected to exchange: {exchange_name}")
            return exchange
        except Exception as e:
            logger.error(f"Error initializing exchange: {str(e)}")
            raise
    
    def _init_telegram(self):
        """Initialize Telegram bot for notifications."""
        try:
            self.telegram_bot = telegram.Bot(token=self.telegram_config["token"])
            logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {str(e)}")
            self.telegram_config["enabled"] = False
    
    def send_telegram_message(self, message):
        """Send a message via Telegram if enabled."""
        if self.telegram_config["enabled"] and self.telegram_bot:
            try:
                self.telegram_bot.send_message(
                    chat_id=self.telegram_config["chat_id"],
                    text=message,
                    parse_mode='Markdown'
                )
            except Exception as e:
                logger.error(f"Error sending Telegram message: {str(e)}")
    
    def update_pair_list(self):
        """Update the list of cryptocurrency pairs to monitor."""
        try:
            # If pair_whitelist is provided and not empty, use it
            whitelist = self.exchange_config.get("pair_whitelist", [])
            blacklist = self.exchange_config.get("pair_blacklist", [])
            
            if whitelist:
                self.pairs = whitelist
            else:
                # Otherwise use the dynamic pairlist configuration
                markets = self.exchange.fetch_markets()
                
                # Filter by asset type
                asset_type = self.exchange_config.get("asset_type", "spot")
                filtered_markets = [m for m in markets if m['type'] == asset_type]
                
                # Get volume information if using VolumePairList
                if self.pairlist_config["method"] == "VolumePairList":
                    # Get 24h ticker data for all pairs
                    tickers = self.exchange.fetch_tickers()
                    
                    # Sort by volume
                    sort_key = self.pairlist_config.get("sort_key", "quoteVolume")
                    sorted_tickers = sorted(
                        tickers.items(), 
                        key=lambda x: x[1].get(sort_key, 0), 
                        reverse=True
                    )
                    
                    # Take top N pairs
                    number_assets = self.pairlist_config.get("number_assets", 20)
                    top_pairs = [pair for pair, _ in sorted_tickers[:number_assets]]
                    
                    # Apply minimum value filter if specified
                    min_value = self.pairlist_config.get("min_value", 0)
                    if min_value > 0:
                        top_pairs = [
                            pair for pair in top_pairs 
                            if tickers[pair].get(sort_key, 0) >= min_value
                        ]
                    
                    self.pairs = top_pairs
                else:
                    # Default to all pairs
                    self.pairs = [market['symbol'] for market in filtered_markets]
            
            # Apply blacklist
            if blacklist:
                self.pairs = [p for p in self.pairs if p not in blacklist]
                
            logger.info(f"Updated pair list: {len(self.pairs)} pairs")
            
        except Exception as e:
            logger.error(f"Error updating pair list: {str(e)}")
            # In case of error, keep the previous pair list
    
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        Uses the MarketDataCache to minimize API calls.
        
        Args:
            symbol (str): Symbol to fetch data for
            timeframe (str): Timeframe (e.g., '1m', '5m', '1h', '1d')
            limit (int): Number of candles to fetch
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        # Create cache key
        cache_key = f"{symbol}_{timeframe}"
        
        # Check if we have fresh data in cache
        timeframe_seconds = self.market_cache.get_timeframe_seconds(timeframe)
        # For OHLCV data, we consider it fresh if it's less than half a candle old
        fresh_threshold = timeframe_seconds * 0.5
        
        cached_data = self.market_cache.get(cache_key)
        
        if cached_data is not None and not self.market_cache.should_update(cache_key, timeframe):
            # Cache hit and data is fresh
            return cached_data
        
        # If we have cached data but it needs updating
        if cached_data is not None:
            try:
                # Get the last timestamp
                last_timestamp = cached_data['timestamp'].iloc[-1]
                if isinstance(last_timestamp, pd.Timestamp):
                    # Convert pandas timestamp to milliseconds
                    since = int(last_timestamp.timestamp() * 1000)
                else:
                    # Assume it's already milliseconds
                    since = int(last_timestamp)
                
                # Fetch only new data
                new_ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)
                
                if new_ohlcv and len(new_ohlcv) > 0:
                    # Convert to DataFrame
                    new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                    
                    # Update cache with new data
                    updated_df = self.market_cache.update_ohlcv_data(cache_key, timeframe, new_df)
                    
                    logger.debug(f"Updated {symbol} {timeframe} with {len(new_df)} new candles")
                    return updated_df
                
                # If no new data, return cached data
                return cached_data
                
            except Exception as e:
                logger.warning(f"Error updating data for {symbol} {timeframe}: {str(e)}")
                # Return cached data if update fails
                return cached_data
        
        # If no cache or update needed, fetch all data
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Store in cache
            self.market_cache.set(cache_key, df)
            
            logger.debug(f"Fetched {symbol} {timeframe} data ({len(df)} candles)")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return None
    
    def analyze_pair(self, symbol):
        """
        Analyze a single pair and return trading signals.
        
        Args:
            symbol (str): Symbol to analyze
            
        Returns:
            dict or None: Trading signal if found, None otherwise
        """
        try:
            # Fetch data for different timeframes
            timeframes = self.strategy.get_timeframes()
            data = {}
            
            for tf in timeframes:
                data[tf] = self.fetch_ohlcv(symbol, timeframe=tf)
                
                # Skip if data is missing
                if data[tf] is None or len(data[tf]) < 10:  # Minimum data requirement
                    logger.warning(f"Insufficient data for {symbol} on {tf} timeframe")
                    return None
            
            # Analyze using the strategy
            return self.strategy.analyze(symbol, data)
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None
    
    def scan_market(self):
        """
        Scan all pairs in the market for trading signals.
        """
        signals = []
        
        # Run cache maintenance
        self.market_cache.clean()
        
        # Occasionally log cache statistics
        current_time = time.time()
        if current_time - self.last_cache_stats_log > 3600:  # Once per hour
            cache_stats = self.market_cache.get_cache_stats()
            logger.info(
                f"Cache stats: {cache_stats['entries']} entries, "
                f"{cache_stats['size_mb']:.2f}MB, "
                f"Hit ratio: {cache_stats['hit_ratio']:.2f}, "
                f"Hits: {cache_stats['hits']}, "
                f"Misses: {cache_stats['misses']}"
            )
            self.last_cache_stats_log = current_time
        
        for symbol in self.pairs:
            try:
                signal = self.analyze_pair(symbol)
                if signal:
                    signals.append(signal)
                    logger.info(f"Signal found: {signal}")
                    
                    # Notify via Telegram if enabled
                    if self.telegram_config["enabled"]:
                        message = f"*Signal Alert*\n"\
                                 f"Symbol: {signal['symbol']}\n"\
                                 f"Type: {signal['type']}\n"\
                                 f"Price: {signal['price']:.8f}\n"\
                                 f"Timeframe: {signal['timeframe']}\n"\
                                 f"Strategy: {signal['strategy']}\n"\
                                 f"Time: {signal['time']}"
                        
                        self.send_telegram_message(message)
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                logger.error(f"Error processing {symbol}: {str(e)}")
        
        return signals
    
    def _scanner_thread(self):
        """Thread function for continuous market scanning."""
        while self.running:
            try:
                # Update pair list every refresh_period
                if hasattr(self, 'last_pairlist_update'):
                    elapsed = time.time() - self.last_pairlist_update
                    if elapsed >= self.pairlist_config.get("refresh_period", 1800):
                        self.update_pair_list()
                        self.last_pairlist_update = time.time()
                else:
                    self.update_pair_list()
                    self.last_pairlist_update = time.time()
                
                # Scan the market
                signals = self.scan_market()
                
                if signals:
                    logger.info(f"Found {len(signals)} signals in this iteration")
                else:
                    logger.info("No signals found in this iteration")
                
                # Wait for a defined interval before next scan
                scan_interval = self.strategy.get_scan_interval()
                time.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Error in scanner thread: {str(e)}")
                time.sleep(60)  # Wait a bit before retrying after an error
    
    def start(self):
        """Start the continuous market scanner."""
        if self.running:
            logger.warning("Scanner is already running")
            return
        
        self.running = True
        self.scanner_thread = threading.Thread(target=self._scanner_thread)
        self.scanner_thread.daemon = True
        self.scanner_thread.start()
        
        logger.info("Trading scanner started")
    
    def stop(self):
        """Stop the continuous market scanner."""
        self.running = False
        if hasattr(self, 'scanner_thread') and self.scanner_thread.is_alive():
            self.scanner_thread.join(timeout=10)
        logger.info("Trading scanner stopped")
        
        # Log final cache statistics
        cache_stats = self.market_cache.get_cache_stats()
        logger.info(
            f"Final cache stats: {cache_stats['entries']} entries, "
            f"{cache_stats['size_mb']:.2f}MB, "
            f"Hit ratio: {cache_stats['hit_ratio']:.2f}"
        )

def main():
    """Main entry point for the trading scanner."""
    try:
        logger.info("Starting crypto trading signal scanner...")
        
        # Create and start the trading scanner
        scanner = TradingScanner()
        scanner.start()
        
        # Display startup message
        print("=" * 60)
        print("Crypto Trading Signal Scanner")
        print("=" * 60)
        print(f"Scanner is running with strategy: {scanner.strategy.__class__.__name__}")
        print(f"Monitoring exchange: {scanner.exchange_config['name']}")
        print(f"Asset type: {scanner.exchange_config['asset_type']}")
        print(f"Telegram notifications: {'Enabled' if scanner.telegram_config['enabled'] else 'Disabled'}")
        print("\nPress Ctrl+C to stop the scanner")
        print("=" * 60)
        
        # Keep the main thread alive
        while True:
            try:
                time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down scanner...")
                scanner.stop()
                print("Scanner stopped. Goodbye!")
                break
            
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)