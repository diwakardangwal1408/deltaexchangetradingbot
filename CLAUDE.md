yes# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC Trading System for Delta Exchange - A comprehensive cryptocurrency trading bot supporting both BTC futures (long/short directional trades) and BTC options strategies. Features automated trading with web UI control, targeting high win rates (60-80%) using multi-timeframe technical analysis signals.

**🚀 LATEST PERFORMANCE OPTIMIZATIONS (2025):**
- **Ultra-Fast Backtesting**: 6x faster with pre-calculated indicators and parallel data loading
- **1M Precision Exits**: Minute-level exit timing for maximum profitability  
- **Comprehensive Historical Data**: 90+ days of 1M, 3M, and 180+ days of 1H data
- **Multi-Threaded Startup**: 3x faster server startup with parallel data processing

## Key Development Commands

### Installation and Setup
```bash
# Install basic strategy dependencies
pip install -r requirements_btc.txt

# Install Delta Exchange dependencies  
pip install -r requirements_delta.txt

# Install web UI dependencies
pip install -r requirements_flask.txt

# Initial setup with API credentials
python setup_delta_trading.py
```

### Running the System
```bash
# Start web UI (recommended)
python run_ui.py

# Run trading bot directly (headless)
python delta_btc_strategy.py

# Run strategy optimization/backtesting
python btc_strategy_optimization.py

# Test Delta Exchange API connection
python test_delta_api.py

# Data management and optimization commands
python data_downloader.py status          # Check current data status
python data_downloader.py parallel        # Test parallel data loading performance  
python data_manager.py parallel          # Test comprehensive data management
python data_downloader.py build --days 90 # Build 90-day historical dataset
```

### Testing Commands
```bash
# Test individual components
python test_config.py
python test_live_wallet.py
python test_lot_calculations.py
python test_neutral_strategy.py
python test_trend_alignment.py
```

## Architecture Overview

### Core Components

1. **Trading Engine** (`delta_btc_strategy.py`) - Main trading bot that orchestrates everything
2. **Exchange Client** (`delta_exchange_client.py`) - API wrapper for Delta Exchange with timestamp-based historical data
3. **Strategy Engine** (`btc_multi_timeframe_strategy.py`) - Multi-timeframe technical analysis
4. **Configuration** (`config_manager.py` + `application.config`) - Centralized config management
5. **Web UI** (`app.py` + templates) - Flask-based trading dashboard
6. **🆕 Backtest Engine** (`backtest_engine.py`) - Ultra-fast backtesting with 1M precision exits
7. **🆕 Data Manager** (`data_manager.py`) - In-memory data management with parallel loading
8. **🆕 Data Downloader** (`data_downloader.py`) - Multi-chunk historical data acquisition

### Strategy Variants
- `btc_final_strategy.py` - Optimized base strategy (60% win rate)
- `btc_ultra_selective_strategy.py` - Ultra-selective approach (fewer trades)
- `btc_trading_strategy.py` - Original strategy implementation
- `btc_options_trader.py` - Production options trader

### Data Flow
1. **🆕 Historical Data**: Parallel CSV loading (1M, 3M, 1H) → Pre-calculated indicators → In-memory cache
2. **Market Data**: Delta Exchange API → Real-time price feeds → Technical indicators
3. **Signal Generation**: Multi-timeframe analysis (1H trend + 3M signals) → Entry/exit decisions  
4. **🆕 Precision Exits**: 1M candle analysis → Exact stop/target execution → P&L optimization
5. **Risk Management**: Position sizing → Order placement → Real-time monitoring
6. **UI Updates**: Real-time status → Trade history → Performance metrics → Backtesting results

## Configuration System

### Primary Config File: `application.config`
- API credentials (encrypted storage)
- Trading parameters (position sizes, risk limits)
- Strategy settings (signal thresholds, timeframes)
- Risk management (stop loss, take profit, daily limits)

### Runtime State Files
- `delta_trading_state.json` - Current positions and trading state
- `delta_btc_trading.log` - Detailed trading logs
- `delta_config.json` - Legacy config (use application.config instead)

## Key Technical Concepts

### Multi-Timeframe Analysis
- **1H Timeframe**: Trend identification (Fisher Transform, TSI, Pivot Points, Dow Theory)
- **3M Timeframe**: Entry/exit signals (VWAP, Parabolic SAR, ATR, Price Action scoring)
- **🆕 1M Timeframe**: Precision exit timing (HIGH/LOW analysis for exact stop/target hits)
- Directional bias determines futures long/short and options call/put selection
- **🆕 Signal Scoring System**: 20-point scale (-20 to +20) for objective decision making

### Trading Strategy Types
- **Directional Strategy**: BTC futures long/short based on trend alignment
- **Options Strategy**: Daily expiry options with high-probability signals
- **Neutral Strategy**: Range-bound trading and volatility plays

### Risk Management
- Max 0.5-1% risk per trade
- Daily loss limits (1-2% of portfolio)  
- **🆕 1M Precision Trailing Stops**: Minute-level trailing stop adjustments
- **🆕 Multi-Level Exits**: Stop Loss → Take Profit → Quick Profit → Trailing Stops
- Maximum 2 concurrent positions
- **🆕 Exact Exit Pricing**: HIGH/LOW analysis ensures realistic fill simulation

### Trading Instruments

**BTC Futures (Directional Trades)**:
- Long/short positions based on trend analysis
- Perpetual and fixed-expiry futures contracts
- Leverage-based position sizing
- Trend-following entries with technical confirmations

**BTC Options Strategy**:
- Daily expiry BTC options on Delta Exchange
- Slightly OTM options for better risk/reward
- Time-based exits (4 hours before expiry)
- Volume and liquidity filtering
- Call/put selection based on directional bias

## 🚀 Performance Optimizations & Backtesting

### Ultra-Fast Backtesting System
```bash
# Pre-load 90 days of comprehensive data in parallel (1.35 seconds)
python data_manager.py parallel

# Run ultra-fast backtests with 1M precision exits  
python app.py # Use web UI for backtesting with progress tracking
```

**Performance Achievements:**
- **6x Faster Backtesting**: Pre-calculated indicators eliminate computation overhead
- **3x Faster Startup**: Parallel data loading (1M, 3M, 1H simultaneously)  
- **1M Exit Precision**: Minute-level exit timing vs 3-minute approximations
- **Comprehensive Dataset**: 129K+ 1M candles, 43K+ 3M candles, 4K+ 1H candles

### Historical Data Management
```bash
# Check current data status
python data_downloader.py status

# Build comprehensive historical datasets  
python data_downloader.py build --days 90    # 3-month dataset
python data_downloader.py chunks 1m --days 60 # 1M precision data
python data_downloader.py chunks 1h --days 180 # 1H trend data
```

**Data Architecture:**
- **1M Data**: Precision exit timing, stop-loss accuracy, trailing stops
- **3M Data**: Signal generation (VWAP, SAR, ATR, Price Action)
- **1H Data**: Trend identification (Fisher, TSI, Dow Theory)
- **Memory Usage**: ~11MB total for complete ultra-fast backtesting

### Backtesting Features
- **Every 3M Candle Analysis**: No signal skipping - complete coverage
- **Multi-Level Exit System**: Stop Loss → Take Profit → Quick Profit → Trailing Stops  
- **Realistic Fill Simulation**: HIGH/LOW price analysis for accurate exit timing
- **Complete Trade Tracking**: Entry/exit timestamps, P&L, duration, exit reasons
- **Performance Monitoring**: Processing rates, memory usage, ETA calculations

## Development Workflow

### Adding New Features
1. Update strategy parameters in `application.config`
2. Modify strategy logic in relevant strategy files
3. Update web UI if needed (`app.py` + templates)
4. Test with paper trading mode first
5. Run optimization backtests

### Testing Strategy Changes
1. Enable paper trading mode in config
2. Run backtests with `btc_strategy_optimization.py`
3. Monitor live paper trading for 1-2 weeks
4. Validate win rates and risk metrics before live deployment

### Debugging Issues
1. Check `delta_btc_trading.log` for detailed logs
2. Verify API connection with `test_delta_api.py`
3. Check config with `test_config.py`
4. Use web UI logs page for real-time monitoring

## Security and Safety

### API Security
- Credentials stored in `application.config` with basic encoding
- Paper trading mode as default safety measure
- API permissions limited to read and trade (no withdraw)

### Trading Safety
- Built-in position limits and loss limits
- Automatic trading halt on daily loss limit
- Paper trading validation before live deployment
- Emergency stop functionality in web UI

## File Structure Importance

### Core Files (Don't Delete)
- `delta_btc_strategy.py` - Main trading engine
- `delta_exchange_client.py` - Exchange API wrapper with timestamp-based historical data
- `config_manager.py` - Configuration management
- `application.config` - Main configuration file
- **🆕 `backtest_engine.py`** - Ultra-fast backtesting with 1M precision exits
- **🆕 `data_manager.py`** - In-memory data management and parallel loading
- **🆕 `data_downloader.py`** - Historical data acquisition and chunked downloads

### UI Files
- `app.py` - Flask web application
- `templates/` - HTML templates for web UI
- `static/` - CSS/JS assets for web UI

### Strategy Files (Modular)
- Multiple strategy implementations for different approaches
- Can be swapped by changing imports in main files

## Deployment Notes

### Production Checklist
1. Validate paper trading performance (60%+ win rate)
2. Set appropriate position sizes and risk limits
3. Configure API credentials with live account
4. Enable live trading mode in config
5. Monitor closely for first few trades

### Monitoring Requirements
- Daily review of trades and performance
- Weekly strategy parameter optimization
- Monthly backtest validation with recent data
- Continuous risk management oversight

This system is designed for automated cryptocurrency trading (both futures and options) with strong emphasis on risk management and user control.


While working with this repo do not assume anything , always ask user before doing any changes to the files, plan of action and only when user confirms then only persist the changed made to the files.

Think yourself as a Senior Software engineer while working on the code and always use efficient data structures and design patterns to come up with neat and clean code.

Do not use any fallback mechanism in the logic , clearly error out when something is not working

While fixing bugs do not remove any major functionality which can destroy the system, be surgical in your updates to file and do changes where required only.

Always remember you are working on windows machine so try to use windows commands on terminal and not unix commands

You will not start server on your own.


APIs for delta exchange to be used are placed at 

Products - https://docs.delta.exchange/#delta-exchange-api-v2-products

Orders - https://docs.delta.exchange/#delta-exchange-api-v2-products for your reference

Positions -https://docs.delta.exchange/#delta-exchange-api-v2-positions

Tradehistory -https://docs.delta.exchange/#delta-exchange-api-v2-tradehistoryNo 

