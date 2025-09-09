# BTC Trading Bot for Delta Exchange

A comprehensive cryptocurrency trading bot system supporting both BTC futures (long/short directional trades) and BTC options strategies with automated trading, web UI control, and advanced risk management.

## 🚀 Features

### Multi-Strategy Trading
- **BTC Futures Trading**: Long/short directional trades based on multi-timeframe technical analysis
- **BTC Options Trading**: Daily expiry options with high-probability signals  
- **Neutral Strategy**: Range-bound trading and volatility plays using strangle selling

### Advanced Exit Management
- **ATR-Based Dynamic Exits**: Market volatility-adaptive stop loss and take profit levels
- **Stop Hunt Protection**: Avoids psychological levels where stops typically cluster
- **Comprehensive Exit Types**: Stop Loss, Take Profit, Quick Profit, Trailing Stop, Max Risk Hit
- **Real-time Position Monitoring**: 30-second interval checks for responsive exit management

### Technical Analysis
- **Multi-Timeframe Analysis**: 3M entry signals with 1H trend confirmation
- **Dashboard Indicators**: VWAP, Parabolic SAR, ATR, Price Action, Fisher Transform, TSI, Pivot Points, Dow Theory, Williams Alligator
- **Signal Validation**: Requires trend alignment between timeframes for trade execution
- **High Win Rate Focus**: Targets 60-80% win rates using selective signal filtering

### Risk Management
- **Dollar-Based Risk**: Fixed USD amounts for predictable risk management
- **ATR-Based Risk**: Dynamic risk based on market volatility with hunt protection
- **Position Limits**: Configurable maximum concurrent positions (default: 1)
- **Daily Loss Limits**: Automatic trading halt on daily loss threshold
- **Portfolio Protection**: Maximum risk per trade and overall portfolio limits

### Web Interface
- **Real-time Dashboard**: Live trading signals, position monitoring, P&L tracking
- **Settings Management**: Complete configuration through web UI
- **Backtesting Engine**: Historical strategy validation with realistic trade simulation
- **Trade History**: Detailed logging and analysis of all trading activity

## 📋 Requirements

### Python Dependencies
```bash
# Basic strategy dependencies
pip install -r requirements_btc.txt

# Delta Exchange API
pip install -r requirements_delta.txt

# Web UI dependencies  
pip install -r requirements_flask.txt
```

### API Access
- Delta Exchange API credentials (API key and secret)
- Paper trading mode recommended for initial testing

## 🛠️ Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/diwakardangwal1408/deltaexchangetradingbot.git
cd deltaexchangetradingbot
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements_btc.txt
pip install -r requirements_delta.txt
pip install -r requirements_flask.txt
```

### 3. Initial Configuration
```bash
# Run setup script to configure API credentials
python setup_delta_trading.py
```

### 4. Test API Connection
```bash
python test_delta_api.py
```

## 🚀 Usage

### Web UI (Recommended)
```bash
# Start the web interface
python run_ui.py
```
Access the dashboard at `http://localhost:5000`

### Direct Trading Bot
```bash
# Run trading bot directly (headless mode)
python delta_btc_strategy.py
```

### Backtesting
```bash
# Run strategy optimization/backtesting
python btc_strategy_optimization.py
```

## ⚙️ Configuration

### Main Configuration File: `application.config`

#### API Configuration
```ini
[API_CONFIGURATION]
api_key = your_delta_exchange_api_key
api_secret = your_delta_exchange_api_secret
paper_trading = true  # Start with paper trading for safety
```

#### Trading Strategy Settings
```ini
[FUTURES_STRATEGY]
enabled = true
long_signal_threshold = 7      # Score threshold for long trades
short_signal_threshold = -7    # Score threshold for short trades
leverage = 100                 # Leverage for futures positions
position_size_usd = 600       # Position size in USD
require_trend_alignment = true # Require 1H trend confirmation
trend_bullish_threshold = 3   # 1H trend classification
trend_bearish_threshold = -3  # 1H trend classification
```

#### Risk Management - Choose Your Exit Strategy

##### Option 1: Fixed Dollar Exits (Simple & Predictable)
```ini
[DOLLAR_BASED_RISK]
enabled = true
stop_loss_usd = 200.0         # Maximum loss per trade
take_profit_usd = 300.0       # Profit target per trade
trailing_stop_usd = 100.0     # Trailing stop distance
quick_profit_usd = 200.0      # Early profit taking
max_risk_usd = 150.0          # Maximum risk per trade
daily_loss_limit_usd = 1000.0 # Daily loss limit
```

##### Option 2: ATR-Based Exits (Dynamic & Hunt-Resistant)
```ini
[ATR_EXITS]
enabled = true
atr_period = 14                    # ATR calculation period
stop_loss_atr_multiplier = 2.0     # Stop = Entry ± (ATR × multiplier)
take_profit_atr_multiplier = 3.0   # Profit = Entry ± (ATR × multiplier) 
trailing_atr_multiplier = 1.5      # Trailing = ATR × multiplier
buffer_zone_atr_multiplier = 0.3   # Anti-hunt buffer
volume_threshold_percentile = 70   # High volume detection
hunting_zone_offset = 5            # Distance from round numbers
```

#### Portfolio Settings
```ini
[PORTFOLIO_SETTINGS]
portfolio_size = 150000.0     # Total portfolio value
position_size_usd = 600.0     # Base position size
max_daily_loss = 600.0        # Daily loss limit
max_positions = 1             # Maximum concurrent positions
```

#### Trading Timing (TradingView Sync)
```ini
[TRADING_TIMING]
trading_start_time = 17:30    # When trading begins (affects candle timing)
timezone = Asia/Kolkata       # Trading timezone
```

### Advanced Settings

#### Neutral Options Strategy
```ini
[NEUTRAL_STRATEGY]
enabled = false               # Enable strangle selling
lot_size = 1                 # Options lot size
leverage_percentage = 50.0   # Leverage for margin calculation
strike_distance = 8          # Strikes away from ATM
profit_target_pct = 30.0     # Profit target percentage
stop_loss_pct = 50.0         # Stop loss percentage
```

## 📊 Trading Logic

### Signal Generation Process

1. **1H Timeframe Analysis** (Trend Identification)
   - Moving Average Crossovers
   - Dow Theory Analysis  
   - Fisher Transform
   - Williams Alligator
   - Pivot Points

2. **3M Timeframe Analysis** (Entry/Exit Signals)
   - Bollinger Bands
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - VWAP (Volume Weighted Average Price)
   - Parabolic SAR
   - ATR (Average True Range)
   - Price Action Analysis

3. **Signal Validation**
   - Multi-timeframe alignment required
   - Minimum signal strength filtering
   - Time-based trade spacing
   - Position limits enforcement

### Exit Management System

#### Exit Priority Order
1. **Max Risk Hit** (Priority 1) - Emergency exit at maximum loss
2. **Stop Loss** (Priority 2) - Price-based stop loss
3. **Take Profit** (Priority 3) - Profit target reached
4. **Quick Profit** (Priority 4) - Early profit taking
5. **Trailing Stop** (Automatic) - Follows favorable price movement

#### ATR-Based Exit Features
- **Dynamic Levels**: Stop/profit levels adapt to current market volatility
- **Hunt Protection**: Automatically avoids round numbers (45000, 45500, etc.)
- **Volume Awareness**: Increases protection during high-volume periods
- **Trend Following**: ATR-based trailing stops follow trends effectively

## 🧪 Testing & Validation

### Component Testing
```bash
# Test individual components
python test_config.py              # Configuration validation
python test_live_wallet.py         # Wallet connectivity
python test_lot_calculations.py    # Position sizing
python test_neutral_strategy.py    # Options strategy
python test_trend_alignment.py     # Multi-timeframe logic
```

### Backtesting
The system includes a comprehensive backtesting engine that:
- Uses real historical data from Delta Exchange
- Simulates realistic trade execution and slippage
- Processes trades chronologically candle-by-candle
- Validates exit criteria with proper timing
- Provides detailed performance metrics

#### Backtest Results Include:
- Total trades executed and win rate
- P&L analysis and drawdown metrics  
- Exit reason breakdown (Stop Loss, Take Profit, etc.)
- Risk-adjusted returns and Sharpe ratio
- Trade-by-trade detailed analysis

### Paper Trading
Always test strategies in paper trading mode before live deployment:
```ini
[API_CONFIGURATION]
paper_trading = true
```

## 🔒 Security & Safety

### Trading Safety Features
- **Paper Trading Default**: Always starts in safe simulation mode
- **Position Limits**: Built-in maximum position and loss limits
- **Emergency Stops**: Automatic halt on daily loss threshold
- **API Permissions**: Limited to trading only (no withdraw permissions)

### Security Best Practices
- API credentials stored in configuration file with basic encoding
- No sensitive information logged or transmitted
- Read-only dashboard access for monitoring
- Configurable risk limits for all position sizes

## 📁 Project Structure

```
├── README.md                           # This file
├── CLAUDE.md                          # Development instructions for Claude Code
├── application.config                 # Main configuration file
│
├── Core Trading System/
│   ├── delta_btc_strategy.py          # Main trading engine
│   ├── delta_exchange_client.py       # Delta Exchange API wrapper
│   ├── btc_multi_timeframe_strategy.py # Technical analysis engine
│   └── config_manager.py              # Configuration management
│
├── Backtesting System/
│   ├── backtest_engine.py             # Backtesting engine
│   ├── btc_strategy_optimization.py   # Strategy optimization
│   └── candle_timing.py               # TradingView timing sync
│
├── Web Interface/
│   ├── app.py                         # Flask web application
│   ├── run_ui.py                      # Web UI launcher
│   ├── templates/                     # HTML templates
│   │   ├── dashboard.html             # Main trading dashboard
│   │   ├── settings.html              # Configuration interface
│   │   ├── backtesting.html           # Backtesting interface
│   │   └── base.html                  # Base template
│   └── static/                        # CSS/JS assets
│
├── Strategy Variants/
│   ├── btc_final_strategy.py          # Optimized strategy (60% win rate)
│   ├── btc_ultra_selective_strategy.py # Ultra-selective approach
│   ├── btc_trading_strategy.py        # Original strategy
│   └── btc_options_trader.py          # Production options trader
│
├── Testing & Utilities/
│   ├── test_*.py                      # Component tests
│   ├── setup_delta_trading.py        # Initial setup script
│   └── logger_config.py              # Logging configuration
│
└── Requirements/
    ├── requirements_btc.txt           # Basic strategy dependencies
    ├── requirements_delta.txt         # Delta Exchange dependencies
    └── requirements_flask.txt         # Web UI dependencies
```

## 📈 Performance Metrics

### Target Performance
- **Win Rate**: 60-80%
- **Risk Per Trade**: 0.5-1% of portfolio
- **Daily Risk Limit**: 1-2% of portfolio  
- **Maximum Drawdown**: <10%
- **Profit Factor**: >1.5

### Recent Improvements (Latest Update)
- ✅ Fixed "Backtest end" exit issue - now shows real exit reasons
- ✅ Implemented ATR-based dynamic exits with stop-hunt protection
- ✅ Enhanced live trading with comprehensive position monitoring
- ✅ Added proper candle timing synchronization with TradingView
- ✅ Unified risk management UI with clear exit strategy selection
- ✅ Set maximum 1 concurrent position for clearer trading behavior

### Sample Backtest Results
```
SUCCESS: Backtest completed with 21 trades
Exit Reason Summary:
  Stop Loss: 12 trades
  Take Profit: 4 trades  
  Quick Profit: 3 trades
  Max Risk Hit: 2 trades
Total P&L: $XX.XX
Win Rate: XX%
```

## 🚨 Important Warnings

### Risk Disclosure
- **High Risk**: Cryptocurrency trading involves substantial risk of loss
- **Leverage Risk**: Using leverage amplifies both gains and losses
- **No Guarantees**: Past performance does not guarantee future results
- **Test First**: Always validate strategies in paper trading mode

### Operational Risks
- **API Connectivity**: Ensure stable internet connection for live trading
- **Exchange Risks**: Delta Exchange operational status affects trading
- **Configuration**: Incorrect settings can lead to unintended behavior
- **Monitoring**: Active supervision recommended during live trading

## 🤝 Contributing

### Development Workflow
1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Test** changes thoroughly with paper trading
4. **Commit** changes (`git commit -m 'Add amazing feature'`)
5. **Push** to branch (`git push origin feature/amazing-feature`)
6. **Create** Pull Request

### Code Standards
- Follow existing code structure and naming conventions
- Include comprehensive error handling
- Add logging for important operations
- Test all changes in paper trading mode
- Document configuration changes

## 📞 Support

### Getting Help
- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: Check CLAUDE.md for development guidelines
- **Testing**: Use paper trading mode for safe testing
- **Configuration**: Review application.config for all settings

### Troubleshooting
1. **API Connection Issues**: Check credentials and network connectivity
2. **Configuration Errors**: Validate settings in application.config
3. **Trading Issues**: Review logs in delta_btc_trading.log
4. **Web UI Problems**: Check Flask application logs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this trading bot. Always conduct thorough testing and risk assessment before deploying with real funds.

---

**Built with ❤️ for algorithmic trading enthusiasts**

*Last Updated: September 2025*