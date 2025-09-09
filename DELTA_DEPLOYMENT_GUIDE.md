# Delta Exchange BTC Options Trading - Deployment Guide

## Overview

This guide will help you deploy the BTC options trading strategy on Delta Exchange for paper trading. The system is designed to trade daily expiry BTC options with high win-rate signals.

## Prerequisites

1. **Delta Exchange Account** - You need an active Delta Exchange India account
2. **API Access** - Enable API access in your Delta Exchange account
3. **Python 3.8+** - Ensure you have Python installed
4. **Stable Internet** - Required for real-time trading

## Files Created

The following files have been created in `C:\Users\diwak\`:

### Core Trading Files
- `delta_exchange_client.py` - Delta Exchange API wrapper
- `delta_btc_strategy.py` - Main trading bot
- `delta_config.json` - Configuration file
- `setup_delta_trading.py` - Setup and configuration script

### Strategy Files (Dependencies)
- `btc_final_strategy.py` - BTC trading strategy logic
- `btc_trading_strategy.py` - Base strategy implementation
- `requirements_btc.txt` - Python dependencies

## Step-by-Step Deployment

### Step 1: Install Dependencies

```bash
cd C:\Users\diwak
pip install -r requirements_btc.txt
pip install requests asyncio
```

### Step 2: Get Delta Exchange API Credentials

1. **Login to Delta Exchange**
   - Go to https://www.delta.exchange/
   - Login to your account

2. **Create API Key**
   - Navigate to Settings → API Management
   - Click "Create New API Key"
   - Set permissions:
     - ✅ Read
     - ✅ Trade
     - ❌ Withdraw (Not needed)
   - Copy your **API Key** and **API Secret**

3. **Enable Testnet (for Paper Trading)**
   - Go to https://testnet.delta.exchange/
   - Create testnet account if needed
   - Generate testnet API credentials

### Step 3: Run Setup Script

```bash
python setup_delta_trading.py
```

The setup script will:
- Prompt for your API credentials
- Test the connection
- Configure risk parameters
- Create configuration file

**Sample Setup Process:**
```
Enter your Delta Exchange API Key: your_api_key_here
Enter your Delta Exchange API Secret: [hidden]
Do you want to use paper trading mode? (Y/n): Y
Enter your portfolio size in USD (default 100000): 100000
Enter position size in USD per trade (default 500): 500
Enter maximum daily loss in USD (default 2000): 2000
```

### Step 4: Verify Configuration

Check that `delta_config.json` has been created with your settings:

```json
{
  "api_key": "your_api_key",
  "api_secret": "your_api_secret", 
  "paper_trading": true,
  "portfolio_size": 100000,
  "position_size_usd": 500,
  "max_daily_loss": 2000
}
```

### Step 5: Test the Strategy

Run a quick test:

```bash
python delta_exchange_client.py
```

Expected output:
```
Testing Delta Exchange connection...
Current BTC Price: $43250.50
Found 12 daily expiry BTC options
Found 3 suitable CALL options
Sample option: BTC-43000-C-18JAN24
```

### Step 6: Start Paper Trading

```bash
python delta_btc_strategy.py
```

## Understanding the Trading System

### Strategy Logic
1. **Signal Generation**: Uses Bollinger Bands, ATR, Volume, RSI, MACD
2. **Entry Criteria**: Requires 4+ confirmations for high-probability trades
3. **Risk Management**: 0.5% risk per trade, 2% daily loss limit
4. **Exit Strategy**: 100% profit target, 50% stop loss, time-based exits

### Position Management
- **Maximum Positions**: 2 concurrent positions
- **Position Size**: $500 per trade (configurable)
- **Options Selection**: Daily expiry, slightly out-of-the-money
- **Time Management**: Closes positions 4 hours before expiry

### Monitoring and Logs

The system creates detailed logs in `delta_btc_trading.log`:

```
2024-01-18 10:30:15 - INFO - Connected to Delta Exchange. Current BTC price: $43250.50
2024-01-18 10:31:22 - INFO - NEW SIGNAL DETECTED: CALL - Strength: 5
2024-01-18 10:31:25 - INFO - TRADE EXECUTED: trade_1705567885
2024-01-18 10:31:25 - INFO - Symbol: BTC-43500-C-18JAN24
2024-01-18 10:31:25 - INFO - Contracts: 2
2024-01-18 10:31:25 - INFO - Entry Price: $0.008500
2024-01-18 10:31:25 - INFO - Total Cost: $17.00
```

## Risk Management Features

### Built-in Safety Controls
- **Paper Trading Default**: Starts in testnet mode
- **Daily Loss Limits**: Stops trading after daily loss limit
- **Position Limits**: Maximum 2 concurrent positions
- **Time Controls**: Minimum 1 hour between trades
- **Premium Filters**: Only trades options within price range

### Monitoring Tools
- **Real-time Logging**: All actions logged with timestamps
- **State Persistence**: Trading state saved in JSON file
- **Performance Tracking**: P&L tracking and statistics
- **Error Handling**: Robust error handling and recovery

## Configuration Options

### Risk Settings (in delta_config.json)
```json
"risk_management": {
  "stop_loss_pct": 50,           // 50% stop loss
  "take_profit_pct": 100,        // 100% profit target
  "quick_profit_pct": 30,        // 30% quick profit in 1 hour
  "time_exit_hours": 20          // Close 4 hours before expiry
}
```

### Strategy Settings
```json
"strategy_params": {
  "min_signal_strength": 4,      // Minimum confirmations required
  "volume_threshold": 2.0,       // 2x volume requirement
  "min_time_between_trades": 3600 // 1 hour minimum between trades
}
```

## Common Operations

### Start Trading
```bash
python delta_btc_strategy.py
```

### Stop Trading
Press `Ctrl+C` to stop gracefully

### Check Logs
```bash
tail -f delta_btc_trading.log
```

### View Trading State
```bash
cat delta_trading_state.json
```

### Update Configuration
Edit `delta_config.json` and restart the system

## Paper Trading Best Practices

### Phase 1: Initial Testing (1-2 weeks)
- Run paper trading continuously
- Monitor signal generation and execution
- Check option selection logic
- Verify risk management controls

### Phase 2: Parameter Tuning (1-2 weeks)
- Adjust signal strength requirements
- Fine-tune position sizing
- Optimize entry/exit criteria
- Monitor win rate and profit factor

### Phase 3: Performance Validation (2-4 weeks)
- Track performance metrics
- Compare with backtesting results
- Assess market condition adaptability
- Document any issues or improvements

## Transitioning to Live Trading

⚠️ **Only after thorough paper trading validation**

1. **Update Configuration**:
   ```json
   "paper_trading": false
   ```

2. **Use Live API Credentials**:
   - Replace testnet credentials with live ones
   - Ensure live account has sufficient balance

3. **Start with Smaller Positions**:
   - Reduce position_size_usd to $100-200 initially
   - Gradually increase after confidence builds

4. **Enhanced Monitoring**:
   - Monitor more frequently
   - Set up notifications/alerts
   - Keep manual override capability

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API credentials
   - Check internet connection
   - Confirm API permissions

2. **No Options Found**
   - Check if daily expiry options are available
   - Verify BTC options are trading
   - Check time of day (options may expire)

3. **No Signals Generated**
   - Lower min_signal_strength temporarily
   - Check BTC price movement and volatility
   - Verify strategy data updates

4. **Orders Not Executing**
   - Check account balance
   - Verify option liquidity
   - Check order size limits

### Log Analysis

Look for these key log messages:
- `Connected to Delta Exchange` - System startup
- `NEW SIGNAL DETECTED` - Trading opportunity found
- `TRADE EXECUTED` - Position opened
- `POSITION CLOSED` - Position closed with P&L

### Getting Help

1. **Check Logs**: Most issues are logged with details
2. **Review Configuration**: Ensure all settings are correct
3. **Test Components**: Run individual scripts to isolate issues
4. **Delta Exchange Support**: Contact support for API-related issues

## Performance Monitoring

### Key Metrics to Track
- **Win Rate**: Target 60-80%
- **Profit Factor**: Target > 1.5
- **Daily P&L**: Monitor against limits
- **Signal Quality**: Average signal strength
- **Execution Success**: Order fill rates

### Weekly Review
- Analyze trade history
- Check market condition impact
- Adjust parameters if needed
- Update strategy if market changes

## Security Considerations

### API Security
- Never share API credentials
- Use IP restrictions if available
- Monitor API usage regularly
- Rotate keys periodically

### System Security
- Keep trading system private
- Use secure networks only
- Regular backup of configuration
- Monitor for unauthorized access

## Disclaimer

⚠️ **Important Risk Warning**:
- Cryptocurrency trading involves significant risk
- Past performance does not guarantee future results
- Only trade with money you can afford to lose
- The strategy may not perform as expected in all market conditions
- Consider consulting with a financial advisor

---

**Support**: For technical issues, check the logs and configuration first. For Delta Exchange API issues, contact their support team.

**Updates**: Keep the strategy files updated and monitor for any changes to the Delta Exchange API.