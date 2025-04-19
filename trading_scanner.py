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
from telegram.request import HTTPXRequest
import asyncio
from strategies_custom import get_strategy
from cache_manager import MarketDataCache
import concurrent.futures
from functools import lru_cache, wraps

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

class RateLimiter:
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls
            self.calls = [call for call in self.calls if now - call < self.period]
            
            # Check if we've reached the limit
            if len(self.calls) >= self.max_calls:
                # Calculate sleep time
                sleep_time = self.period - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    # Update our current time after sleeping
                    now = time.time()
                    # Clean calls list again after sleeping
                    self.calls = [call for call in self.calls if now - call < self.period]
            
            # Add this call
            self.calls.append(now)
            
            return func(*args, **kwargs)
        return wrapper
    
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

    async def _send_telegram_message_async(self, message):
        """Send a message via Telegram asynchronously."""
        if not self.telegram_config["enabled"] or not self.telegram_bot:
            return
            
        try:
            request = HTTPXRequest(connection_pool_size=8)
            bot = telegram.Bot(token=self.telegram_config["token"], request=request)
            await bot.send_message(
                chat_id=self.telegram_config["chat_id"],
                text=message,
                parse_mode='Markdown'
            )
        except Exception as e:
            logger.error(f"Error sending Telegram message: {str(e)}")

    def send_telegram_message(self, message):
        """Send a message via Telegram if enabled."""
        if not self.telegram_config["enabled"] or not self.telegram_bot:
            return
        
        try:
            # Create a new event loop for each message
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async function in the new loop
            loop.run_until_complete(self._send_telegram_message_async(message))
            loop.close()
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
                # Get all available markets
                markets = self.exchange.fetch_markets()
                asset_type = self.exchange_config.get("asset_type", "spot")
                
                # Filter by asset type based on the Bybit API response structure
                if asset_type == "perpetual":
                    # For perpetual contracts, check for type = swap and symbol ending with :USDT
                    filtered_markets = [
                        m for m in markets 
                        if m['type'] == 'swap' and m['symbol'].endswith(':USDT')
                    ]
                elif asset_type == "spot":
                    filtered_markets = [m for m in markets if m['type'] == 'spot']
                elif asset_type == "future":
                    filtered_markets = [m for m in markets if m['type'] == 'future']
                elif asset_type == "option":
                    filtered_markets = [m for m in markets if m['type'] == 'option']
                else:
                    filtered_markets = [m for m in markets if m['type'] == asset_type]
                
                # If entire_universe is true, use all available assets for the specific type
                if self.pairlist_config.get("entire_universe", False):
                    self.pairs = [market['symbol'] for market in filtered_markets]
                    logger.info(f"Using entire universe: {len(self.pairs)} {asset_type} pairs")
                
                # Otherwise use the dynamic pairlist configuration
                elif self.pairlist_config["method"] == "VolumePairList":
                    # Get 24h ticker data for all pairs
                    tickers = self.exchange.fetch_tickers()
                    
                    # Create a set of valid symbols from filtered markets
                    valid_symbols = {market['symbol'] for market in filtered_markets}
                    
                    # Sort by volume
                    sort_key = self.pairlist_config.get("sort_key", "quoteVolume")
                    sorted_tickers = sorted(
                        [(symbol, ticker) for symbol, ticker in tickers.items() if symbol in valid_symbols],
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
    
    @RateLimiter(max_calls=5, period=1.0)  # Max 3 calls per second
    def fetch_ohlcv(self, symbol, timeframe='1h', limit=100):
        """Rate-limited version of fetch_ohlcv"""
        # Original implementation
        cache_key = f"{symbol}_{timeframe}"
        cached_data = self.market_cache.get(cache_key)
        
        if cached_data is not None and not self.market_cache.should_update(cache_key, timeframe):
            return cached_data
        
        if cached_data is not None:
            try:
                # Get the last timestamp
                last_timestamp = cached_data['timestamp'].iloc[-1]
                if isinstance(last_timestamp, pd.Timestamp):
                    since = int(last_timestamp.timestamp() * 1000)
                else:
                    since = int(last_timestamp)
                
                new_ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since)
                
                if new_ohlcv and len(new_ohlcv) > 0:
                    new_df = pd.DataFrame(new_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
                    updated_df = self.market_cache.update_ohlcv_data(cache_key, timeframe, new_df)
                    logger.debug(f"Updated {symbol} {timeframe} with {len(new_df)} new candles")
                    return updated_df
                
                return cached_data
                    
            except Exception as e:
                logger.warning(f"Error updating data for {symbol} {timeframe}: {str(e)}")
                return cached_data
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            self.market_cache.set(cache_key, df)
            logger.debug(f"Fetched {symbol} {timeframe} data ({len(df)} candles)")
            return df
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_ohlcv(self, symbols, timeframe='1h', limit=100):
        """
        Fetch OHLCV data for multiple symbols with rate limiting
        """
        results = {}
        symbols_to_fetch = []
        
        # Check cache first
        for symbol in symbols:
            cache_key = f"{symbol}_{timeframe}"
            cached_data = self.market_cache.get(cache_key)
            
            if cached_data is not None and not self.market_cache.should_update(cache_key, timeframe):
                # Cache hit and data is fresh
                results[symbol] = cached_data
            else:
                symbols_to_fetch.append(symbol)
        
        # Fetch remaining symbols sequentially with rate limiting
        # This is safer than parallel fetching to avoid rate limits
        for symbol in symbols_to_fetch:
            try:
                results[symbol] = self.fetch_ohlcv(symbol, timeframe, limit)
            except Exception as e:
                logger.error(f"Error fetching OHLCV for {symbol}: {str(e)}")
                results[symbol] = None
        
        return results

    def analyze_multiple_pairs(self, symbols):
        """
        Analyze multiple pairs and return trading signals.
        
        Args:
            symbols (list): Symbols to analyze
            
        Returns:
            list: List of trading signals
        """
        signals = []
        timeframes = self.strategy.get_timeframes()
        
        # Fetch data for each timeframe
        timeframe_data = {}
        for tf in timeframes:
            timeframe_data[tf] = self.fetch_multiple_ohlcv(symbols, timeframe=tf)
        
        # Make a copy of symbols to avoid modification during iteration
        symbols_to_analyze = symbols.copy()
        
        # Now analyze each symbol
        for symbol in symbols_to_analyze:
            try:
                data = {}
                has_all_data = True
                
                for tf in timeframes:
                    data[tf] = timeframe_data[tf].get(symbol)
                    if data[tf] is None or len(data[tf]) < 10:
                        has_all_data = False
                        break
                
                if has_all_data:
                    signal = self.strategy.analyze(symbol, data)
                    if signal:
                        signals.append(signal)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
        
        return signals

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
        Scan all pairs in the market for trading signals with rate limiting
        """
        # Run cache maintenance
        self.market_cache.clean()
        
        # Log cache statistics if needed
        current_time = time.time()
        if current_time - self.last_cache_stats_log > 3600:
            cache_stats = self.market_cache.get_cache_stats()
            logger.info(f"Cache stats: {cache_stats['entries']} entries, {cache_stats['size_mb']:.2f}MB, Hit ratio: {cache_stats['hit_ratio']:.2f}")
            self.last_cache_stats_log = current_time
        
        # Use a smaller batch size to avoid rate limits
        batch_size = 5  # Process 5 symbols at a time
        all_signals = []
        
        # Process all pairs in batches
        for i in range(0, len(self.pairs), batch_size):
            batch_pairs = self.pairs[i:i+batch_size]
            logger.debug(f"Processing batch {i//batch_size + 1}/{(len(self.pairs) + batch_size - 1)//batch_size}: {batch_pairs}")
            
            try:
                batch_signals = self.analyze_multiple_pairs(batch_pairs)
                
                # Process signals from this batch
                for signal in batch_signals:
                    # Convert timestamp in signal to user's timezone if present
                    if 'time' in signal and signal['time']:
                        try:
                            # Parse the timestamp and convert to user's timezone (+2)
                            from datetime import timezone, timedelta
                            dt = datetime.strptime(signal['time'], '%Y-%m-%d %H:%M:%S')
                            dt = dt.replace(tzinfo=None)
                            dt_utc = dt.replace(tzinfo=timezone.utc)
                            dt_local = dt_utc.astimezone(timezone(timedelta(hours=2)))
                            signal['time'] = dt_local.strftime('%Y-%m-%d %H:%M:%S')
                        except Exception as e:
                            logger.debug(f"Could not convert signal time: {e}")
                    
                    all_signals.append(signal)
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
                
                # Add a small delay between batches to avoid rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_pairs}: {str(e)}")
                # Continue with next batch even if this one fails
                time.sleep(1)  # Slightly longer delay after error
        
        return all_signals
    
    def _scanner_thread(self):
        """Thread function for continuous market scanning."""
        last_process_time = {}  # Track last processing time for each timeframe
        
        while self.running:
            try:
                # Update pair list every refresh_period
                if hasattr(self, 'last_pairlist_update'):
                    elapsed = time.time() - self.last_pairlist_update
                    if elapsed >= self.pairlist_config.get("refresh_period", 1800):
                        self.update_pair_list()
                        self.last_pairlist_update = time.time()
                        # Print current list of assets being analyzed
                        logger.info(f"Whitelist pairs ({len(self.pairs)}): {self.pairs}")
                else:
                    self.update_pair_list()
                    self.last_pairlist_update = time.time()
                    # Print initial list of assets being analyzed
                    logger.info(f"Whitelist pairs ({len(self.pairs)}): {self.pairs}")
                
                # Get strategy timeframes
                timeframes = self.strategy.get_timeframes()
                current_time = time.time()
                should_process = False
                
                # Check if any timeframe needs processing
                for tf in timeframes:
                    tf_seconds = self.market_cache.get_timeframe_seconds(tf)
                    
                    # Initialize if not yet tracked
                    if tf not in last_process_time:
                        last_process_time[tf] = 0
                    
                    # Check if a new candle is available
                    elapsed = current_time - last_process_time[tf]
                    if elapsed >= tf_seconds:
                        should_process = True
                        last_process_time[tf] = current_time - (current_time % tf_seconds)
                        logger.info(f"Processing new {tf} candle")
                
                # Process data only if new candles are available
                if should_process:
                    # Scan the market
                    signals = self.scan_market()
                    
                    if signals:
                        logger.info(f"Found {len(signals)} signals in this iteration")
                    else:
                        logger.info("No signals found in this iteration")
                else:
                    logger.debug("No new candles available for processing")
                
                # Always wait exactly 1 minute before the next check
                logger.info("RUNNING")
                time.sleep(60)
                
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

        if scanner.telegram_config['enabled']:
            scanner.send_telegram_message("Crypto Trading Signal Scanner started successfully!")

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