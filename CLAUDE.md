# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BTC Trading System for Delta Exchange - A comprehensive cryptocurrency trading bot supporting both BTC futures (long/short directional trades) and BTC options strategies. Features automated trading with web UI control, targeting high win rates (60-80%) using multi-timeframe technical analysis signals.

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
2. **Exchange Client** (`delta_exchange_client.py`) - API wrapper for Delta Exchange
3. **Strategy Engine** (`btc_multi_timeframe_strategy.py`) - Multi-timeframe technical analysis
4. **Configuration** (`config_manager.py` + `application.config`) - Centralized config management
5. **Web UI** (`app.py` + templates) - Flask-based trading dashboard

### Strategy Variants
- `btc_final_strategy.py` - Optimized base strategy (60% win rate)
- `btc_ultra_selective_strategy.py` - Ultra-selective approach (fewer trades)
- `btc_trading_strategy.py` - Original strategy implementation
- `btc_options_trader.py` - Production options trader

### Data Flow
1. **Market Data**: Delta Exchange API → Technical indicators calculation
2. **Signal Generation**: Multi-timeframe analysis → Entry/exit decisions
3. **Risk Management**: Position sizing → Order placement → Monitoring
4. **UI Updates**: Real-time status → Trade history → Performance metrics

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
- **1H Timeframe**: Trend identification (MA crossovers, Dow theory)
- **3M Timeframe**: Entry/exit signals (Bollinger Bands, RSI, MACD)
- Directional bias determines futures long/short and options call/put selection

### Trading Strategy Types
- **Directional Strategy**: BTC futures long/short based on trend alignment
- **Options Strategy**: Daily expiry options with high-probability signals
- **Neutral Strategy**: Range-bound trading and volatility plays

### Risk Management
- Max 0.5-1% risk per trade
- Daily loss limits (1-2% of portfolio)
- Trailing stops (15% default)
- Maximum 2 concurrent positions

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
- `delta_exchange_client.py` - Exchange API wrapper  
- `config_manager.py` - Configuration management
- `application.config` - Main configuration file

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


APIs for delta exchange to be used are placed at 

Products - https://docs.delta.exchange/#delta-exchange-api-v2-products

Orders - https://docs.delta.exchange/#delta-exchange-api-v2-products for your reference

Positions -https://docs.delta.exchange/#delta-exchange-api-v2-positions

Tradehistory -https://docs.delta.exchange/#delta-exchange-api-v2-tradehistoryNo 

