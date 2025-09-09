# BTC Options Trader - Web UI Setup Guide

## Overview

A modern, intuitive web interface for managing your BTC options trading bot. The UI provides real-time monitoring, configuration management, trade history, and live logs.

## 🎯 Features

### **Dashboard**
- Real-time bot status and controls (Start/Stop)
- Trading mode indicator (Paper/Live)
- API connection status
- Daily P&L tracking
- Portfolio overview
- Current positions and trade count
- Quick actions panel

### **Settings**
- Secure API credential configuration
- Paper/Live trading toggle with safety warnings
- Risk management parameters
- Strategy configuration (signal strength, volume thresholds)
- Exit strategy settings (stop loss, take profit)
- Real-time API connection testing

### **Trade History**
- Complete trade history with P&L
- Performance statistics and win rate
- Interactive charts (P&L over time, trade outcomes)
- Trade filtering and sorting
- Export capabilities

### **Live Logs**
- Real-time trading log monitoring
- Log filtering by level (INFO, WARNING, ERROR)
- Search functionality with highlighting
- Auto-refresh capability
- Log level color coding

## 🚀 Quick Start

### **Step 1: Install Dependencies**
```bash
pip install -r requirements_flask.txt
```

### **Step 2: Ensure Configuration Exists**
```bash
python setup_delta_trading.py
```
*(Skip if you've already configured your API credentials)*

### **Step 3: Launch the Web UI**
```bash
python run_ui.py
```

The UI will automatically open in your browser at `http://127.0.0.1:5000`

## 📁 Files Created

All files are located in `C:\Users\diwak\`:

### **Core Application**
- `app.py` - Main Flask application
- `run_ui.py` - Launch script with dependency checking
- `requirements_flask.txt` - Web UI dependencies

### **Templates** (in `templates/` folder)
- `base.html` - Base template with navigation
- `dashboard.html` - Main dashboard interface  
- `settings.html` - Configuration and API setup
- `trades.html` - Trade history with charts
- `logs.html` - Live log viewer

### **Static Files** (in `static/` folder)
- `css/style.css` - Custom styling
- `js/app.js` - JavaScript functionality

## 🔧 Configuration

### **Environment Variables**
You can customize the UI with these environment variables:

```bash
# Windows
set FLASK_HOST=127.0.0.1
set FLASK_PORT=5000
set FLASK_DEBUG=false

# Linux/Mac
export FLASK_HOST=127.0.0.1
export FLASK_PORT=5000
export FLASK_DEBUG=false
```

### **Security Settings**
- UI runs locally by default (`127.0.0.1`)
- Change `FLASK_HOST` to `0.0.0.0` only if you need network access
- Never expose to internet without proper security measures
- API credentials are stored locally in encrypted format

## 🎮 Using the Web UI

### **Initial Setup**
1. **Access Settings**: Navigate to Settings page
2. **Enter API Credentials**: Add your Delta Exchange API key and secret
3. **Test Connection**: Use "Test API Connection" button
4. **Configure Risk**: Set portfolio size, position size, daily loss limits
5. **Verify Paper Trading**: Ensure paper trading is enabled for testing

### **Starting Trading**
1. **Dashboard**: Go to main dashboard
2. **Check Status**: Verify all indicators are green
3. **Start Bot**: Click "Start" button
4. **Monitor**: Watch real-time status updates

### **Monitoring Trades**
1. **Trade History**: View all executed trades
2. **Performance**: Check win rate and P&L statistics
3. **Charts**: Analyze performance trends
4. **Logs**: Monitor detailed trading activity

### **Stopping Trading**
1. **Stop Bot**: Click "Stop" button on dashboard
2. **Wait**: Allow current trades to close naturally
3. **Review**: Check final P&L and performance

## 🛡️ Safety Features

### **Built-in Protections**
- **Paper Trading Default**: Starts in safe testing mode
- **API Testing**: Validates credentials before use
- **Risk Warnings**: Alerts for high-risk settings
- **Position Limits**: Enforces maximum concurrent positions
- **Loss Limits**: Automatic trading halt at daily loss limit

### **User Interface Safety**
- **Confirmation Dialogs**: For risky actions (live trading mode)
- **Visual Indicators**: Clear status indicators throughout
- **Real-time Updates**: Live monitoring of bot status
- **Error Handling**: Graceful handling of connection issues

## 📊 Dashboard Features

### **Real-time Status Cards**
- **Bot Status**: Running/Stopped with start/stop controls
- **Trading Mode**: Paper/Live with safety indicators
- **API Status**: Connection health to Delta Exchange
- **Daily P&L**: Current day's profit/loss

### **Portfolio Overview**
- **Account Balance**: Real wallet balances from Delta Exchange
- **Portfolio Settings**: Configured risk parameters
- **Active Positions**: Current open trades
- **Performance Metrics**: Win rate, total trades, cumulative P&L

### **Quick Actions**
- Refresh wallet balance
- Test API connection
- View trade history
- Access trading logs
- Emergency stop trading

## ⚙️ Advanced Settings

### **Strategy Configuration**
- **Signal Strength**: Minimum confirmations required (2-8)
- **Volume Threshold**: Required volume multiple (1.0x - 5.0x)
- **Time Between Trades**: Minimum interval (5min - 4hours)

### **Risk Management**
- **Stop Loss**: Percentage loss to close position (10% - 90%)
- **Take Profit**: Percentage gain to close position (20% - 500%)
- **Quick Profit**: Fast profit target for momentum (10% - 100%)
- **Time Exit**: Hours before expiry to force close (1-23)

### **Options Settings**
- **Premium Range**: Min/max option premiums to consider
- **Order Timeout**: Maximum wait time for order execution
- **Position Sizing**: Risk-based calculation parameters

## 🔍 Troubleshooting

### **Common Issues**

1. **UI Won't Start**
   ```bash
   # Check dependencies
   pip install -r requirements_flask.txt
   
   # Check Python version (3.8+ required)
   python --version
   ```

2. **API Connection Failed**
   - Verify API credentials in Settings
   - Check Delta Exchange API status
   - Ensure paper/live mode matches your credentials

3. **Bot Won't Start**
   - Check API connection status
   - Verify sufficient account balance
   - Review trading logs for error details

4. **No Trade History**
   - Ensure bot has been running
   - Check if trades were executed successfully
   - Verify log files for trade confirmations

### **Log Files**
- **Application Logs**: `delta_btc_trading.log`
- **Flask Logs**: Console output when running `run_ui.py`
- **Trading State**: `delta_trading_state.json`

### **Port Conflicts**
If port 5000 is in use:
```bash
# Windows
set FLASK_PORT=8080
python run_ui.py

# Linux/Mac
export FLASK_PORT=8080
python run_ui.py
```

## 🔄 Updates and Maintenance

### **Updating the UI**
1. Backup your configuration: `delta_config.json`
2. Update files as needed
3. Restart the UI: `python run_ui.py`

### **Data Backup**
Important files to backup:
- `delta_config.json` - Configuration
- `delta_trading_state.json` - Trading state
- `delta_btc_trading.log` - Trading history

### **Performance Optimization**
- UI updates every 10 seconds automatically
- Log viewer shows last 100 lines
- Charts limit data points for performance
- Auto-cleanup of old alert notifications

## 🌐 Network Access

### **Local Access Only (Recommended)**
```bash
# Default - localhost only
python run_ui.py
```

### **Network Access (Advanced Users)**
```bash
# WARNING: Only use on secure networks
set FLASK_HOST=0.0.0.0
python run_ui.py
```

⚠️ **Security Warning**: Only enable network access on trusted networks. The UI contains sensitive trading information and API credentials.

## 📱 Mobile Compatibility

The UI is fully responsive and works on:
- Desktop browsers (Chrome, Firefox, Safari, Edge)
- Tablet devices (iPad, Android tablets)
- Mobile phones (iOS Safari, Android Chrome)

Optimized mobile features:
- Touch-friendly buttons and controls
- Responsive charts and tables
- Simplified navigation for small screens
- Fast loading for mobile connections

## 🆘 Support

### **Getting Help**
1. Check the logs in the UI for error details
2. Review this guide for common solutions
3. Ensure all dependencies are installed correctly
4. Verify API credentials and connectivity

### **Best Practices**
- Always start with paper trading
- Monitor performance regularly  
- Keep position sizes conservative
- Review logs daily for any issues
- Backup configuration files regularly

---

**🎉 You're Ready!** The web UI provides a professional interface for managing your BTC options trading bot with safety, monitoring, and control features built-in.