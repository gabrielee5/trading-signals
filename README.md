# Crypto Trading Signal Scanner

A Python program that continuously scans the cryptocurrency market for trading signals based on configurable strategies. The program connects to a cryptocurrency exchange (Bybit by default), monitors selected pairs, and generates alerts when trading opportunities are detected according to the implemented strategies.

## Features

- Connects to cryptocurrency exchanges using the CCXT library
- Dynamically updates pair lists based on volume or other criteria
- Implements multiple trading strategies with customizable parameters
- Sends alerts via Telegram when signals are detected
- Runs continuously with configurable scanning intervals
- Logs all activities and signals for later review
- Efficient data caching system to minimize API calls and improve performance

## Setup

### Prerequisites

- Python 3.8 or higher
- TA-Lib (Technical Analysis Library)

### Installation

1. Clone the repository or download the source code
2. Install the required dependencies:

```
pip install -r requirements.txt
```

### TA-Lib Installation

TA-Lib can be challenging to install on some systems. Here are platform-specific instructions:

**Windows:**
Download and install the pre-built binary from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)

**macOS:**
```
brew install ta-lib
pip install TA-Lib
```

**Linux:**
```
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib
```

## Configuration

Update the `config.json` file with your exchange API credentials and preferences:

```json
{
    "exchange": {
        "name": "bybit",
        "asset_type": "perpetual",
        "key": "YOUR_API_KEY",
        "secret": "YOUR_API_SECRET",
        "pair_whitelist": [],
        "pair_blacklist": []
    },
    "pairlists": [
        {
            "entire_universe": false,
            "method": "VolumePairList",
            "number_assets": 25,
            "sort_key": "quoteVolume",
            "min_value": 0,
            "refresh_period": 1800
        }
    ],
    "telegram": {
        "enabled": false,
        "token": "YOUR_TELEGRAM_BOT_TOKEN",
        "chat_id": "YOUR_TELEGRAM_CHAT_ID"
    },
    "cache": {
        "max_size_mb": 300,
        "max_age_hours": 48
    }
}
```

### Pair Selection

- Leave `pair_whitelist` empty to use the dynamic pair selection
- Add specific pairs to `pair_whitelist` to focus on those pairs only
- Add pairs to `pair_blacklist` to exclude them from analysis

### Telegram Notifications

To enable Telegram notifications:

1. Create a Telegram bot using BotFather
2. Get your chat ID (you can use the GetIDs Bot)
3. Update the `telegram` section in the config file with your bot token and chat ID
4. Set `enabled` to `true`

## Strategies

The program includes two example strategies:

1. **CrossoverStrategy**: Generates signals based on moving average crossovers with volume confirmation
2. **RSIStrategy**: Identifies oversold and overbought conditions using RSI with multi-timeframe confirmation

You can customize or create new strategies by extending the `BaseStrategy` class in the `strategies.py` file.

### Strategy Selection

To change the active strategy, modify the parameter in the call to `get_strategy()` in the main file:

```python
# Load a specific strategy
self.strategy = get_strategy("RSIStrategy")
```

## Running the Program

Simply run the main Python file:

```
python trading_scanner.py
```

The program will:
1. Connect to the specified exchange
2. Update the pair list based on your configuration
3. Start scanning the market for signals
4. Log all activities and signals
5. Send notifications via Telegram if enabled

To stop the program, press `Ctrl+C`.

## Logs

All activities and signals are logged in the `trading_scanner.log` file. This includes:
- Connection status to the exchange
- Pair list updates
- Detected trading signals
- Errors and warnings

## Disclaimer

This software is for educational purposes only. Use it at your own risk. Cryptocurrency trading involves significant risk and can result in substantial financial loss. Past performance is not indicative of future results.