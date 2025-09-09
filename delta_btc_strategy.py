import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from delta_exchange_client import DeltaExchangeClient
from btc_multi_timeframe_strategy import BTCMultiTimeframeStrategy
# Excel logging removed - using Delta Exchange only
from config_manager import config_manager
from logger_config import get_logger, TradingLogger

class DeltaBTCOptionsTrader:
    """
    BTC Options Trading Bot for Delta Exchange
    Integrates high win-rate strategy with Delta Exchange API
    """
    
    def __init__(self, config_file="application.config"):
        # Load configuration from application.config
        self.config = config_manager.get_all_config()
        
        # Initialize Delta Exchange client
        self.delta_client = DeltaExchangeClient(
            api_key=self.config['api_key'],
            api_secret=self.config['api_secret'],
            paper_trading=self.config.get('paper_trading', True)
        )
        
        # Initialize multi-timeframe trading strategy  
        self.strategy = BTCMultiTimeframeStrategy(
            api_key=self.config['api_key'],
            api_secret=self.config['api_secret'],
            paper_trading=self.config.get('paper_trading', True)
        )
        
        # Trading parameters from config
        self.max_positions = self.config.get('max_positions', 2)
        
        # Excel logging removed - trade history now maintained on Delta Exchange
        
        # Position tracking with trailing stops
        self.current_positions = {}
        self.open_orders = {}
        self.trailing_stops = {}  # Track trailing stop data for each position
        self.daily_pnl = 0
        self.trade_history = []
        self.available_options = []  # Initialize options list (for strangle trades only)
        self.available_futures = {}  # Initialize futures contracts (for directional trades)
        
        # Setup logging using centralized configuration
        logging_config = self.config.get('logging', {})
        console_level = logging_config.get('console_level', 'INFO')
        log_file = logging_config.get('log_file', 'delta_btc_trading.log')
        
        self.logger = get_logger(__name__, console_level, log_file)
        TradingLogger.log_system_info(self.logger)
        
        # Dollar-based risk management
        dollar_risk_config = self.config.get('dollar_based_risk', {})
        self.max_daily_loss = dollar_risk_config.get('daily_loss_limit_usd', 500)  # USD
        self.position_size_usd = self.config.get('position_size_usd', 500)  # USD per position
        self.stop_loss_usd = dollar_risk_config.get('stop_loss_usd', 100)  # USD
        self.take_profit_usd = dollar_risk_config.get('take_profit_usd', 200)  # USD
        self.trailing_stop_usd = dollar_risk_config.get('trailing_stop_usd', 50)  # USD
        self.quick_profit_usd = dollar_risk_config.get('quick_profit_usd', 60)  # USD
        self.max_risk_usd = dollar_risk_config.get('max_risk_usd', 150)  # USD
        
        self.logger.info("Delta BTC Options Trader initialized")
        if self.config.get('paper_trading', True):
            self.logger.info("Running in PAPER TRADING mode")
        else:
            self.logger.warning("Running in LIVE TRADING mode")
    
    async def initialize(self):
        """Initialize the trading system"""
        try:
            # Test connection
            btc_price = self.delta_client.get_current_btc_price()
            if not btc_price:
                raise Exception("Failed to connect to Delta Exchange API")
            
            self.logger.info(f"Connected to Delta Exchange. Current BTC price: ${btc_price}")
            
            # Load available options
            await self.update_available_options()
            
            # Get account info
            balance = self.delta_client.get_account_balance()
            portfolio = self.delta_client.get_portfolio_summary()
            
            self.logger.info(f"Account initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize: {e}")
            return False
    
    async def update_available_options(self):
        """Update list of available daily expiry options"""
        try:
            self.available_options = self.delta_client.get_daily_expiry_options()
            self.logger.info(f"Found {len(self.available_options)} daily expiry BTC options")
            
            # Log some sample options
            for i, option in enumerate(self.available_options[:3]):
                self.logger.info(f"Sample option {i+1}: {option['symbol']} - Strike: {option.get('strike_price')}")
                
        except Exception as e:
            self.logger.error(f"Error updating available options: {e}")
    
    async def update_available_futures(self):
        """Update list of available BTC futures contracts for directional trading"""
        try:
            self.available_futures = self.delta_client.get_btc_futures_specifications()
            self.logger.info(f"Found {len(self.available_futures)} BTC futures contracts")
            
            # Log available futures contracts
            for symbol, contract in self.available_futures.items():
                self.logger.info(f"Futures contract: {symbol} - Type: {contract.get('contract_type')}")
                
        except Exception as e:
            self.logger.error(f"Error updating available futures: {e}")
    
    def get_dashboard_3m_signals(self):
        """Get 3m signals using same logic as dashboard API"""
        try:
            from app import calculate_technical_indicators
            
            # Get 3m candle data (same as dashboard)
            live_data = self.delta_client.get_historical_candles('BTCUSD', '3m', 200)
            
            if not live_data or len(live_data) < 50:
                self.logger.warning("Insufficient 3m candle data")
                return None
                
            # Convert to DataFrame (same as dashboard)
            df = pd.DataFrame(live_data)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
            df = df.sort_values('timestamp')
            
            # Calculate technical indicators (same as dashboard)
            df = calculate_technical_indicators(df)
            
            # Get latest candle values (same as dashboard)
            latest = df.iloc[-1]
            
            # Extract signal data (same as dashboard)
            base_signal = int(latest['Signal']) if not pd.isna(latest['Signal']) else 0
            base_strength = int(latest['Signal_Strength']) if not pd.isna(latest['Signal_Strength']) else 0
            total_score = int(latest['Total_Score']) if not pd.isna(latest['Total_Score']) else 0
            base_entry_type = latest['Entry_Type'] if not pd.isna(latest['Entry_Type']) else 'None'
            
            # Get futures configuration
            futures_config = self.config.get('futures_strategy', {})
            long_threshold = futures_config.get('long_signal_threshold', 5)
            short_threshold = futures_config.get('short_signal_threshold', -7)
            
            self.logger.info(f"3M SIGNALS: Total Score={total_score}, Long Threshold={long_threshold}, Short Threshold={short_threshold}")
            
            # Check thresholds (same logic as dashboard should use)
            if total_score >= long_threshold:
                return {
                    'signal': 1,
                    'entry_type': 'long_futures',
                    'strength': base_strength,
                    'confidence': min(base_strength / 10.0, 1.0),
                    'btc_price': self.delta_client.get_current_btc_price(),
                    'total_score': total_score,
                    'threshold_used': long_threshold,
                    'reasons': [f"Total Score {total_score} >= Long Threshold {long_threshold}"],
                    'timestamp': datetime.now()
                }
            elif total_score <= short_threshold:
                return {
                    'signal': -1,
                    'entry_type': 'short_futures',
                    'strength': base_strength,
                    'confidence': min(base_strength / 10.0, 1.0),
                    'btc_price': self.delta_client.get_current_btc_price(),
                    'total_score': total_score,
                    'threshold_used': short_threshold,
                    'reasons': [f"Total Score {total_score} <= Short Threshold {short_threshold}"],
                    'timestamp': datetime.now()
                }
            else:
                self.logger.info(f"3M NO SIGNAL: Total Score {total_score} between thresholds ({short_threshold} to {long_threshold})")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting dashboard 3m signals: {e}")
            return None
    
    def update_strategy_data(self):
        """Update strategy with multi-timeframe analysis (LEGACY - kept for compatibility)"""
        try:
            # Run multi-timeframe analysis (15m trend + 3m signals)
            analysis = self.strategy.run_analysis()
            
            if not analysis:
                self.logger.warning("No analysis data received")
                return None
            
            # Extract signal information
            signal_data = analysis['signal_3m']
            trend_data = analysis['trend_1h']
            
            self.logger.info(f"Multi-timeframe analysis: 1H Trend={trend_data['direction']} ({trend_data['strength']:.1f}/10), 3m Signal={signal_data['type']} ({signal_data['strength']}/10)")
            
            if signal_data['signal'] != 0:
                return {
                    'signal': signal_data['signal'],
                    'entry_type': signal_data['type'],
                    'strength': signal_data['strength'],
                    'confidence': min(signal_data['strength'] / 10.0, 1.0),
                    'btc_price': analysis['current_price'],
                    'trend_direction': trend_data['direction'],
                    'trend_strength': trend_data['strength'],
                    'bb_position': signal_data.get('bb_position', 0),
                    'rsi': signal_data.get('rsi', 0),
                    'volume_ratio': signal_data.get('volume_ratio', 0),
                    'atr_pct': signal_data.get('atr_pct', 0),
                    'reasons': signal_data.get('reasons', []),
                    'timestamp': datetime.now()
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error updating strategy data: {e}")
            return None
    
    def validate_trading_conditions(self, signal_data):
        """Validate if trading conditions are met"""
        # Check daily loss limit
        if abs(self.daily_pnl) >= self.max_daily_loss:
            return False, f"Daily loss limit reached: ${self.daily_pnl}"
        
        # Check maximum positions
        if len(self.current_positions) >= self.max_positions:
            return False, f"Maximum positions ({self.max_positions}) reached"
        
        # Check signal strength
        if signal_data['strength'] < 4:
            return False, f"Signal strength too low: {signal_data['strength']}"
        
        # Check market hours (optional - crypto trades 24/7)
        current_hour = datetime.now().hour
        if not (8 <= current_hour <= 22):  # Trade only during active hours
            return False, "Outside preferred trading hours"
        
        return True, "All conditions met"
    
    
    def find_best_futures_contract(self, signal_data):
        """Find the best futures contract for directional trading"""
        try:
            if not self.available_futures:
                return None, "No futures contracts available"
            
            current_price = self.delta_client.get_current_btc_price()
            if not current_price:
                return None, "Could not get current BTC price"
            
            # Look for active perpetual futures first (preferred for directional trades)
            best_contract = None
            best_symbol = None
            
            # Prioritize BTCUSD perpetual futures
            priority_symbols = ['BTCUSD', 'BTCUSDT', 'BTCUSD_PERP', 'BTCUSDT_PERP']
            
            for symbol in priority_symbols:
                if symbol in self.available_futures:
                    contract = self.available_futures[symbol]
                    if contract.get('trading_status') == 'operational':
                        # Get current orderbook to check liquidity
                        try:
                            orderbook = self.delta_client.get_orderbook(symbol)
                            if orderbook and orderbook.get('buy') and orderbook.get('sell'):
                                best_contract = contract
                                best_symbol = symbol
                                break
                        except Exception as e:
                            self.logger.warning(f"Could not get orderbook for {symbol}: {e}")
                            continue
            
            # If no priority symbol found, try any available futures
            if not best_contract:
                for symbol, contract in self.available_futures.items():
                    if contract.get('trading_status') == 'operational':
                        try:
                            orderbook = self.delta_client.get_orderbook(symbol)
                            if orderbook and orderbook.get('buy') and orderbook.get('sell'):
                                best_contract = contract
                                best_symbol = symbol
                                break
                        except Exception as e:
                            continue
            
            if best_contract:
                # Get current market price
                orderbook = self.delta_client.get_orderbook(best_symbol)
                if signal_data['signal'] > 0:  # Bullish - buy futures (long)
                    price = float(orderbook['sell'][0]['price'])  # Ask price for buying
                    side = 'buy'
                else:  # Bearish - sell futures (short)
                    price = float(orderbook['buy'][0]['price'])   # Bid price for selling
                    side = 'sell'
                
                return {
                    'contract': best_contract,
                    'symbol': best_symbol,
                    'price': price,
                    'side': side,
                    'contract_type': 'futures'
                }, "Best futures contract found"
            else:
                return None, "No operational futures contracts with liquidity found"
                
        except Exception as e:
            self.logger.error(f"Error finding best futures contract: {e}")
            return None, f"Error: {str(e)}"
    
    def calculate_position_size(self, symbol, option_price, side='buy'):
        """Calculate number of lots to trade using proper Delta Exchange lot sizes"""
        try:
            # Get lot size for this specific option symbol
            lot_size_btc = self.delta_client.get_lot_size_for_symbol(symbol)
            
            # Calculate premium in USDT for 1 lot
            premium_per_lot_usdt = self.delta_client.calculate_premium_in_usdt(symbol, option_price, 1)
            
            # Calculate maximum lots based on position size in USD
            if premium_per_lot_usdt > 0:
                max_lots = max(1, int(self.position_size_usd / premium_per_lot_usdt))
            else:
                max_lots = 1
            
            # Apply risk limits - maximum 10 lots per trade
            lots = min(max_lots, 10)
            
            # Calculate total costs and margins
            total_premium_usdt = self.delta_client.calculate_premium_in_usdt(symbol, option_price, lots)
            total_margin_usdt = self.delta_client.calculate_margin_requirement(symbol, option_price, lots, side)
            
            self.logger.info(f"Position size calculation for {symbol}:")
            self.logger.info(f"  Lot size: {lot_size_btc} BTC per lot")
            self.logger.info(f"  Option price: {option_price:.6f}")
            self.logger.info(f"  Lots to trade: {lots}")
            self.logger.info(f"  Total premium: ${total_premium_usdt:.2f}")
            self.logger.info(f"  Total margin required: ${total_margin_usdt:.2f}")
            
            return {
                'lots': lots,
                'lot_size_btc': lot_size_btc,
                'total_premium_usdt': total_premium_usdt,
                'total_margin_usdt': total_margin_usdt,
                'premium_per_lot': premium_per_lot_usdt,
                'option_price': option_price,
                
                # Legacy compatibility
                'contracts': lots,  # For backward compatibility
                'total_cost': total_premium_usdt
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return None
    
    async def execute_trade(self, signal_data):
        """Execute a directional trade based on signal using futures (not options)"""
        try:
            # For directional trades, use futures instead of options
            best_contract, message = self.find_best_futures_contract(signal_data)
            if not best_contract:
                self.logger.warning(f"Cannot execute directional trade: {message}")
                return None
            
            # Calculate position size for futures contract
            # For futures, we'll use a simplified position sizing based on portfolio percentage
            portfolio_size = self.position_size_usd
            leverage = 10  # Conservative leverage for directional trades
            position_value = portfolio_size * leverage
            quantity = max(1, int(position_value / best_contract['price']))
            if quantity <= 0:
                self.logger.error("Invalid position size calculated")
                return None
            
            # Place futures order
            order_result = self.delta_client.place_order(
                symbol=best_contract['symbol'],
                side=best_contract['side'],
                quantity=quantity,
                order_type='limit',
                price=best_contract['price']
            )
            
            if order_result.get('success'):
                trade_id = f"trade_{int(time.time())}"
                
                trade_info = {
                    'trade_id': trade_id,
                    'symbol': best_contract['symbol'],
                    'signal': signal_data['signal'],
                    'entry_type': signal_data['entry_type'],
                    'contracts': quantity,
                    'entry_price': best_contract['price'],
                    'side': best_contract['side'],
                    'strategy': 'directional_futures',
                    'entry_time': datetime.now(),
                    'signal_strength': signal_data['strength'],
                    'confidence': signal_data['confidence'],
                    'order_id': order_result['result'].get('id'),
                    'total_cost': quantity * best_contract['price'],
                    'leverage': leverage,
                    'status': 'open'
                }
                
                self.current_positions[trade_id] = trade_info
                self.trade_history.append(trade_info.copy())
                
                # Trade logging removed - using Delta Exchange order history only
                
                self.logger.info(f"DIRECTIONAL FUTURES TRADE EXECUTED: {trade_id}")
                self.logger.info(f"Symbol: {best_contract['symbol']}")
                self.logger.info(f"Quantity: {quantity}")
                self.logger.info(f"Side: {best_contract['side']}")
                self.logger.info(f"Entry Price: ${best_contract['price']:.2f}")
                self.logger.info(f"Total Value: ${quantity * best_contract['price']:.2f}")
                
                return trade_info
            
            else:
                self.logger.error(f"Order failed: {order_result}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    async def monitor_positions(self):
        """Monitor open positions for exit conditions"""
        positions_to_close = []
        
        for trade_id, position in self.current_positions.items():
            try:
                # Get current option price
                orderbook = self.delta_client.get_orderbook(position['symbol'])
                if not orderbook or not orderbook.get('buy'):
                    continue
                
                current_price = float(orderbook['buy'][0]['price'])  # Bid price for selling
                
                # Calculate P&L
                entry_price = position['entry_price']
                contracts = position['contracts']
                pnl = (current_price - entry_price) * contracts
                pnl_pct = (current_price - entry_price) / entry_price * 100
                
                # Check exit conditions
                time_in_position = (datetime.now() - position['entry_time']).total_seconds() / 3600  # hours
                
                exit_reason = None
                
                # Time-based exit (close before expiry)
                if time_in_position >= 20:  # Close 4 hours before daily expiry
                    exit_reason = "Time exit (approaching expiry)"
                
                # Dollar-based profit target
                elif pnl >= self.take_profit_usd:
                    exit_reason = f"Take profit target (${self.take_profit_usd})"
                
                # Dollar-based stop loss
                elif pnl <= -self.stop_loss_usd:
                    exit_reason = f"Stop loss (${self.stop_loss_usd})"
                
                # Dollar-based quick profit (in first hour)
                elif time_in_position <= 1 and pnl >= self.quick_profit_usd:
                    exit_reason = f"Quick profit (${self.quick_profit_usd} in 1 hour)"
                
                if exit_reason:
                    positions_to_close.append((trade_id, exit_reason, current_price, pnl))
                else:
                    # Log position status
                    self.logger.info(f"Position {trade_id}: {position['symbol']} - P&L: ${pnl:.2f} ({pnl_pct:.1f}%) - Time: {time_in_position:.1f}h")
            
            except Exception as e:
                self.logger.error(f"Error monitoring position {trade_id}: {e}")
        
        # Close positions that meet exit criteria
        for trade_id, reason, exit_price, pnl in positions_to_close:
            await self.close_position(trade_id, reason, exit_price, pnl)
    
    async def close_position(self, trade_id, reason, exit_price, pnl):
        """Close a position"""
        try:
            if trade_id not in self.current_positions:
                return
            
            position = self.current_positions[trade_id]
            
            # Place sell order
            order_result = self.delta_client.place_order(
                symbol=position['symbol'],
                side='sell',
                quantity=position['contracts'],
                order_type='market'
            )
            
            if order_result.get('success'):
                # Update position info
                position['exit_time'] = datetime.now()
                position['exit_price'] = exit_price
                position['exit_reason'] = reason
                position['pnl'] = pnl
                position['status'] = 'closed'
                
                # Update daily P&L
                self.daily_pnl += pnl
                
                # Update Excel database with exit information
                exit_data = {
                    'btc_price_at_exit': exit_price * position.get('strike_price', 1),  # Approximate
                    'exit_premium_primary': exit_price,
                    'total_premium_paid': position.get('total_cost', 0),
                    'gross_pnl': pnl,
                    'commission_fees': 0,  # Update if commission tracking is added
                    'net_pnl': pnl,
                    'roi_percentage': (pnl / position.get('total_cost', 1)) * 100 if position.get('total_cost', 0) > 0 else 0,
                    'exit_reason': reason,
                    'duration_minutes': (datetime.now() - position['entry_time']).total_seconds() / 60,
                    'notes': f"Closed via: {reason}"
                }
                # Position closed, logging complete
                
                self.logger.info(f"POSITION CLOSED: {trade_id}")
                self.logger.info(f"Reason: {reason}")
                self.logger.info(f"Exit Price: ${exit_price:.6f}")
                self.logger.info(f"P&L: ${pnl:.2f}")
                self.logger.info(f"Daily P&L: ${self.daily_pnl:.2f}")
                
                # Remove from current positions
                del self.current_positions[trade_id]
            
            else:
                self.logger.error(f"Failed to close position {trade_id}: {order_result}")
        
        except Exception as e:
            self.logger.error(f"Error closing position {trade_id}: {e}")
    
    def save_state(self):
        """Save trading state to file"""
        try:
            state = {
                'current_positions': {k: {**v, 'entry_time': v['entry_time'].isoformat(), 
                                         'exit_time': v.get('exit_time', datetime.now()).isoformat() if v.get('exit_time') else None} 
                                    for k, v in self.current_positions.items()},
                'daily_pnl': self.daily_pnl,
                'trade_history': [
                    {**trade, 'entry_time': trade['entry_time'].isoformat(),
                     'exit_time': trade.get('exit_time', datetime.now()).isoformat() if trade.get('exit_time') else None}
                    for trade in self.trade_history
                ],
                'last_update': datetime.now().isoformat()
            }
            
            with open('delta_trading_state.json', 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def get_strangle_strikes(self, current_price, expiry_date):
        """Get call and put strike prices for strangle selling"""
        try:
            neutral_config = self.config.get('neutral_strategy', {})
            strike_distance = neutral_config.get('strike_distance', 8)
            
            # Get all available options for the expiry date
            call_options = []
            put_options = []
            
            for option in self.available_options:
                if option.get('expiry_date') == expiry_date:
                    if option.get('option_type') == 'call':
                        call_options.append(option)
                    elif option.get('option_type') == 'put':
                        put_options.append(option)
            
            # Find call strike (8 strikes above ATM)
            call_strikes = sorted([float(opt['strike_price']) for opt in call_options])
            put_strikes = sorted([float(opt['strike_price']) for opt in put_options])
            
            # Find closest ATM strike
            atm_call = min(call_strikes, key=lambda x: abs(x - current_price))
            atm_put = min(put_strikes, key=lambda x: abs(x - current_price))
            
            # Find strike that is 8 positions away from ATM
            atm_call_idx = call_strikes.index(atm_call)
            atm_put_idx = put_strikes.index(atm_put)
            
            # Call strike: 8 strikes above ATM (OTM)
            call_strike_idx = min(atm_call_idx + strike_distance, len(call_strikes) - 1)
            call_strike = call_strikes[call_strike_idx]
            
            # Put strike: 8 strikes below ATM (OTM)  
            put_strike_idx = max(atm_put_idx - strike_distance, 0)
            put_strike = put_strikes[put_strike_idx]
            
            # Find corresponding option objects
            call_option = next((opt for opt in call_options if float(opt['strike_price']) == call_strike), None)
            put_option = next((opt for opt in put_options if float(opt['strike_price']) == put_strike), None)
            
            if call_option and put_option:
                return {
                    'call_option': call_option,
                    'put_option': put_option,
                    'call_strike': call_strike,
                    'put_strike': put_strike,
                    'current_price': current_price,
                    'expiry_date': expiry_date
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting strangle strikes: {e}")
            return None
    
    async def execute_neutral_strangle(self, trend_data):
        """Execute neutral strangle selling strategy"""
        try:
            neutral_config = self.config.get('neutral_strategy', {})
            
            if not neutral_config.get('enabled', True):
                return None
            
            # Get current BTC price
            current_price = self.delta_client.get_current_btc_price()
            if not current_price:
                return None
            
            # Calculate next day expiry - use DDMMYY format to match options data
            from datetime import datetime, timedelta
            expiry_days = neutral_config.get('expiry_days', 1)
            target_date = datetime.now() + timedelta(days=expiry_days)
            target_expiry = target_date.strftime('%d%m%y')  # Format: 080925 to match options
            
            # Get strangle strikes
            strangle_info = self.get_strangle_strikes(current_price, target_expiry)
            if not strangle_info:
                self.logger.warning("Could not find suitable strangle strikes")
                return None
            
            # Get option prices from orderbook
            call_symbol = strangle_info['call_option']['symbol']
            put_symbol = strangle_info['put_option']['symbol']
            
            call_orderbook = self.delta_client.get_orderbook(call_symbol)
            put_orderbook = self.delta_client.get_orderbook(put_symbol)
            
            if not (call_orderbook.get('sell') and put_orderbook.get('sell')):
                self.logger.warning("No sell orders available for strangle options")
                return None
            
            call_bid_price = float(call_orderbook['buy'][0]['price']) if call_orderbook.get('buy') else 0
            put_bid_price = float(put_orderbook['buy'][0]['price']) if put_orderbook.get('buy') else 0
            
            if call_bid_price <= 0 or put_bid_price <= 0:
                self.logger.warning("Invalid bid prices for strangle options")
                return None
            
            # Calculate position sizing using proper Delta Exchange lot calculations
            lot_size = neutral_config.get('lot_size', 1)
            leverage_pct = neutral_config.get('leverage_percentage', 50.0)
            
            # Use Delta Exchange API to calculate proper premium and margins
            call_premium_usdt = self.delta_client.calculate_premium_in_usdt(call_symbol, call_bid_price, lot_size)
            put_premium_usdt = self.delta_client.calculate_premium_in_usdt(put_symbol, put_bid_price, lot_size)
            total_premium = call_premium_usdt + put_premium_usdt
            
            # Calculate margin requirements for selling options
            call_margin = self.delta_client.calculate_margin_requirement(call_symbol, call_bid_price, lot_size, 'sell')
            put_margin = self.delta_client.calculate_margin_requirement(put_symbol, put_bid_price, lot_size, 'sell')
            total_margin = call_margin + put_margin
            
            self.logger.info(f"Strangle margin calculation:")
            self.logger.info(f"  Call premium: ${call_premium_usdt:.2f}, Put premium: ${put_premium_usdt:.2f}")
            self.logger.info(f"  Total premium received: ${total_premium:.2f}")
            self.logger.info(f"  Call margin: ${call_margin:.2f}, Put margin: ${put_margin:.2f}")
            self.logger.info(f"  Total margin required: ${total_margin:.2f}")
            
            # Execute both legs of the strangle
            call_order = self.delta_client.place_order(
                symbol=call_symbol,
                side='sell',
                quantity=lot_size,
                order_type='limit',
                price=call_bid_price
            )
            
            put_order = self.delta_client.place_order(
                symbol=put_symbol,
                side='sell',
                quantity=lot_size,
                order_type='limit',
                price=put_bid_price
            )
            
            if call_order.get('success') and put_order.get('success'):
                trade_id = f"strangle_{int(time.time())}"
                
                # Create strangle position
                strangle_position = {
                    'trade_id': trade_id,
                    'strategy': 'neutral_strangle',
                    'entry_time': datetime.now(),
                    'call_leg': {
                        'symbol': call_symbol,
                        'strike': strangle_info['call_strike'],
                        'order_id': call_order['result']['id'],
                        'entry_price': call_bid_price,
                        'quantity': lot_size
                    },
                    'put_leg': {
                        'symbol': put_symbol,
                        'strike': strangle_info['put_strike'],
                        'order_id': put_order['result']['id'],
                        'entry_price': put_bid_price,
                        'quantity': lot_size
                    },
                    'total_premium_received': total_premium,
                    'margin_deployed': total_margin,
                    'current_price_at_entry': current_price,
                    'expiry_date': target_expiry,
                    'status': 'open',
                    'trailing_stop': {
                        'enabled': True,
                        'best_profit_pct': 0,
                        'stop_loss_pct': neutral_config.get('trailing_stop_loss_pct', 20.0)
                    },
                    'exit_conditions': {
                        'profit_target_pct': neutral_config.get('profit_target_pct', 30.0),
                        'stop_loss_pct': neutral_config.get('stop_loss_pct', 50.0)
                    }
                }
                
                self.current_positions[trade_id] = strangle_position
                
                # Strangle trade executed, logging complete
                
                self.logger.info(f"NEUTRAL STRANGLE EXECUTED: {trade_id}")
                self.logger.info(f"Call Strike: {strangle_info['call_strike']} @ ${call_bid_price:.6f}")
                self.logger.info(f"Put Strike: {strangle_info['put_strike']} @ ${put_bid_price:.6f}")
                self.logger.info(f"Total Premium: ${total_premium:.2f}")
                self.logger.info(f"Margin Deployed: ${total_margin:.2f}")
                
                return strangle_position
            else:
                self.logger.error("Failed to execute strangle - one or both orders failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing neutral strangle: {e}")
            return None
    
    async def monitor_strangle_position(self, trade_id, position):
        """Monitor and manage strangle position with trailing stops"""
        try:
            if position['status'] != 'open':
                return
            
            # Get current option prices
            call_symbol = position['call_leg']['symbol']
            put_symbol = position['put_leg']['symbol']
            
            call_orderbook = self.delta_client.get_orderbook(call_symbol)
            put_orderbook = self.delta_client.get_orderbook(put_symbol)
            
            if not (call_orderbook.get('sell') and put_orderbook.get('sell')):
                return
            
            # Current market prices to buy back (we sold initially)
            call_ask_price = float(call_orderbook['sell'][0]['price'])
            put_ask_price = float(put_orderbook['sell'][0]['price'])
            
            # Calculate current cost to close position using proper lot-based calculations
            quantity = position['call_leg']['quantity']  # Number of lots
            
            call_close_cost = self.delta_client.calculate_premium_in_usdt(call_symbol, call_ask_price, quantity)
            put_close_cost = self.delta_client.calculate_premium_in_usdt(put_symbol, put_ask_price, quantity)
            current_close_cost = call_close_cost + put_close_cost
            
            # Calculate current P&L (positive when options lose value, since we sold initially)
            current_pnl = position['total_premium_received'] - current_close_cost
            current_pnl_pct = (current_pnl / position['margin_deployed']) * 100 if position['margin_deployed'] > 0 else 0
            
            # Update trailing stop
            if current_pnl_pct > position['trailing_stop']['best_profit_pct']:
                position['trailing_stop']['best_profit_pct'] = current_pnl_pct
            
            # Check exit conditions
            exit_reason = None
            
            # Profit target hit
            if current_pnl_pct >= position['exit_conditions']['profit_target_pct']:
                exit_reason = f"Profit target reached: {current_pnl_pct:.1f}%"
            
            # Stop loss hit
            elif current_pnl_pct <= -position['exit_conditions']['stop_loss_pct']:
                exit_reason = f"Stop loss hit: {current_pnl_pct:.1f}%"
            
            # Trailing stop triggered
            elif (position['trailing_stop']['best_profit_pct'] > 10 and  # Only if we had some profit
                  current_pnl_pct <= position['trailing_stop']['best_profit_pct'] - position['trailing_stop']['stop_loss_pct']):
                exit_reason = f"Trailing stop triggered: {current_pnl_pct:.1f}% (was {position['trailing_stop']['best_profit_pct']:.1f}%)"
            
            if exit_reason:
                await self.close_strangle_position(trade_id, position, exit_reason, current_close_cost)
            
        except Exception as e:
            self.logger.error(f"Error monitoring strangle position {trade_id}: {e}")
    
    async def close_strangle_position(self, trade_id, position, reason, close_cost):
        """Close strangle position by buying back both legs"""
        try:
            call_symbol = position['call_leg']['symbol']
            put_symbol = position['put_leg']['symbol']
            quantity = position['call_leg']['quantity']
            
            # Buy back both legs
            call_close_order = self.delta_client.place_order(
                symbol=call_symbol,
                side='buy',
                quantity=quantity,
                order_type='market'
            )
            
            put_close_order = self.delta_client.place_order(
                symbol=put_symbol,
                side='buy',
                quantity=quantity,
                order_type='market'
            )
            
            if call_close_order.get('success') and put_close_order.get('success'):
                # Calculate final P&L
                final_pnl = position['total_premium_received'] - close_cost
                final_pnl_pct = (final_pnl / position['margin_deployed']) * 100
                
                # Update position
                position['exit_time'] = datetime.now()
                position['exit_reason'] = reason
                position['close_cost'] = close_cost
                position['final_pnl'] = final_pnl
                position['final_pnl_pct'] = final_pnl_pct
                position['status'] = 'closed'
                
                # Update daily P&L
                self.daily_pnl += final_pnl
                
                # Add to trade history
                self.trade_history.append(position.copy())
                
                # Update Excel database with strangle exit information
                exit_data = {
                    'btc_price_at_exit': position.get('current_price_at_entry', 0),  # Should get current BTC price
                    'exit_premium_primary': close_cost / 2,  # Approximate call cost
                    'exit_premium_secondary': close_cost / 2,  # Approximate put cost
                    'total_premium_paid': close_cost,
                    'gross_pnl': final_pnl,
                    'commission_fees': 0,  # Update if commission tracking is added
                    'net_pnl': final_pnl,
                    'roi_percentage': final_pnl_pct,
                    'exit_reason': reason,
                    'duration_minutes': (datetime.now() - position['entry_time']).total_seconds() / 60,
                    'max_drawdown': 0,  # Could track this during monitoring
                    'peak_profit': position['trailing_stop'].get('best_profit_pct', 0),
                    'notes': f"Strangle closed: {reason}"
                }
                # Strangle position closed, logging complete
                
                self.logger.info(f"STRANGLE CLOSED: {trade_id}")
                self.logger.info(f"Reason: {reason}")
                self.logger.info(f"Final P&L: ${final_pnl:.2f} ({final_pnl_pct:.1f}%)")
                
                # Remove from current positions
                del self.current_positions[trade_id]
            else:
                self.logger.error(f"Failed to close strangle {trade_id}")
                
        except Exception as e:
            self.logger.error(f"Error closing strangle position {trade_id}: {e}")
    
    async def trading_loop(self):
        """Main trading loop"""
        self.logger.info("Starting trading loop...")
        
        last_signal_time = None
        # Get futures-specific min time between trades
        futures_config = self.config.get('futures_strategy', {})
        min_signal_interval = futures_config.get('min_time_between_trades', 3600)  # Configurable minimum between signals
        
        while True:
            try:
                # Update available options periodically (for strangle trades)
                if len(self.available_options) == 0:
                    await self.update_available_options()
                
                # Update available futures periodically (for directional trades)
                if len(self.available_futures) == 0:
                    await self.update_available_futures()
                
                # Monitor existing positions (both regular and strangle positions)
                await self.monitor_positions()
                
                # Monitor strangle positions separately
                for trade_id, position in list(self.current_positions.items()):
                    if position.get('strategy') == 'neutral_strangle':
                        await self.monitor_strangle_position(trade_id, position)
                
                # Check for new signals (only if we have capacity)
                if len(self.current_positions) < self.max_positions:
                    # Step 1: Check 1-hour timeframe trend
                    from app import calculate_higher_timeframe_indicators
                    import pandas as pd
                    
                    signal_data = None
                    
                    try:
                        # Get 1H candle data for trend analysis
                        live_data = self.delta_client.get_historical_candles('BTCUSD', '1h', 200)
                        if live_data and len(live_data) >= 100:
                            df = pd.DataFrame(live_data)
                            if 'time' in df.columns:
                                df = df.rename(columns={'time': 'timestamp'})
                            if 'timestamp' in df.columns:
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', utc=True)
                            df = df.sort_values('timestamp')
                            
                            # Get 1H trend
                            trend_result = calculate_higher_timeframe_indicators(df)
                            trend_1h = trend_result['overall_trend']
                            
                            self.logger.info(f"1H TREND: {trend_1h}")
                            
                            # Step 2: Get 3-minute technical indicators (same as dashboard)
                            signal_data = self.get_dashboard_3m_signals()
                            
                            # Step 3: Check if both point to same direction
                            if signal_data and trend_1h != 'Neutral':
                                signal_direction = 'Bullish' if signal_data['signal'] > 0 else 'Bearish'
                                
                                if trend_1h == signal_direction:
                                    self.logger.info(f"BOTH ALIGN: 1H={trend_1h}, 3m={signal_direction} → TRADE EXECUTION APPROVED")
                                else:
                                    self.logger.info(f"ALIGNMENT MISMATCH: 1H={trend_1h}, 3m={signal_direction} → NO TRADE")
                                    signal_data = None
                            else:
                                self.logger.info(f"INSUFFICIENT SIGNALS: 1H={trend_1h}, 3m={'None' if not signal_data else 'Valid'} → NO TRADE")
                                signal_data = None
                            
                            current_time = time.time()
                            neutral_config = self.config.get('neutral_strategy', {})
                            min_neutral_interval = neutral_config.get('min_time_between_neutral_trades', 7200)
                            
                            # Check for neutral trend and execute strangle strategy
                            if (trend_1h == 'Neutral' and 
                                neutral_config.get('enabled', True) and
                                (last_signal_time is None or current_time - last_signal_time >= min_neutral_interval)):
                                
                                self.logger.info(f"NEUTRAL TREND DETECTED - Executing strangle strategy")
                                strangle_result = await self.execute_neutral_strangle(trend_result)
                                if strangle_result:
                                    last_signal_time = current_time
                                    self.logger.info(f"Neutral strangle executed successfully")
                            
                            # Regular directional signals when trend is not neutral
                            elif signal_data and trend_1h != 'Neutral':
                                # Check minimum time between signals
                                if (last_signal_time is None or 
                                    current_time - last_signal_time >= min_signal_interval):
                                    
                                    # Validate trading conditions
                                    is_valid, message = self.validate_trading_conditions(signal_data)
                                    
                                    if is_valid:
                                        self.logger.info(f"NEW DIRECTIONAL SIGNAL: {signal_data['entry_type']} - Strength: {signal_data['strength']} (Trend: {trend})")
                                        
                                        # Execute directional trade
                                        trade_result = await self.execute_trade(signal_data)
                                        if trade_result:
                                            last_signal_time = current_time
                                    else:
                                        self.logger.info(f"Signal validation failed: {message}")
                            
                    except Exception as e:
                        self.logger.error(f"Error in trend analysis: {e}")
                        
                        # Fallback to regular signal processing if trend analysis fails
                        if signal_data:
                            current_time = time.time()
                            
                            # Check minimum time between signals
                            if (last_signal_time is None or 
                                current_time - last_signal_time >= min_signal_interval):
                                
                                # Validate trading conditions
                                is_valid, message = self.validate_trading_conditions(signal_data)
                                
                                if is_valid:
                                    self.logger.info(f"NEW SIGNAL DETECTED: {signal_data['entry_type']} - Strength: {signal_data['strength']}")
                                    
                                    # Execute trade
                                    trade_result = await self.execute_trade(signal_data)
                                    if trade_result:
                                        last_signal_time = current_time
                                else:
                                    self.logger.info(f"Signal validation failed: {message}")
                
                # Save state
                self.save_state()
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except KeyboardInterrupt:
                self.logger.info("Trading loop interrupted by user")
                break
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)
        
        self.logger.info("Trading loop ended")
    

async def main():
    """Main function"""
    trader = DeltaBTCOptionsTrader()
    
    # Initialize
    if await trader.initialize():
        # Start trading
        await trader.trading_loop()
    else:
        logger = get_logger(__name__)
        logger.error("Failed to initialize trader")

if __name__ == "__main__":
    asyncio.run(main())