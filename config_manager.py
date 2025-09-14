#!/usr/bin/env python3
"""
Configuration Manager for BTC Options Trading Application
Handles loading, saving, and managing configuration from application.config file
"""

import configparser
import os
from typing import Dict, Any, Union

class ConfigManager:
    def __init__(self, config_file: str = 'application.config'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from application.config file"""
        if not os.path.exists(self.config_file):
            # Create default config if file doesn't exist
            self.create_default_config()
        
        self.config.read(self.config_file)
        return self.get_all_config()
    
    def create_default_config(self):
        """Create default configuration file"""
        default_config = """[API_CONFIGURATION]
api_key = 
api_secret = 
paper_trading = true

[PORTFOLIO_SETTINGS]
portfolio_size = 100000.0
position_size_usd = 500.0
max_daily_loss = 2000.0
max_positions = 2
leverage = 100


[STRATEGY_PARAMETERS]


[DIRECTIONAL_STRATEGY]
enabled = true
bullish_signal_threshold = 11
bearish_signal_threshold = -11
require_trend_alignment = true
min_trend_strength = 5
allow_counter_trend_trades = false

[NEUTRAL_STRATEGY]
enabled = true
lot_size = 1
leverage_percentage = 50.0
strike_distance = 8
expiry_days = 1
trailing_stop_loss_pct = 20.0
profit_target_pct = 30.0
stop_loss_pct = 50.0
min_time_between_neutral_trades = 7200"""
        
        with open(self.config_file, 'w') as f:
            f.write(default_config)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration as a nested dictionary"""
        config_dict = {
            # API Configuration
            'api_key': self.config.get('API_CONFIGURATION', 'api_key', fallback=''),
            'api_secret': self.config.get('API_CONFIGURATION', 'api_secret', fallback=''),
            'paper_trading': self.config.getboolean('API_CONFIGURATION', 'paper_trading', fallback=True),
            
            # Portfolio Settings
            'portfolio_size': self.config.getfloat('PORTFOLIO_SETTINGS', 'portfolio_size', fallback=100000.0),
            'position_size_usd': self.config.getfloat('PORTFOLIO_SETTINGS', 'position_size_usd', fallback=500.0),
            'max_daily_loss': self.config.getfloat('PORTFOLIO_SETTINGS', 'max_daily_loss', fallback=2000.0),
            'max_positions': self.config.getint('PORTFOLIO_SETTINGS', 'max_positions', fallback=2),
            'leverage': self.config.getint('PORTFOLIO_SETTINGS', 'leverage', fallback=100),
            
            
            
            
            # BTC Futures Strategy
            'futures_strategy': {
                'enabled': self.config.getboolean('FUTURES_STRATEGY', 'enabled', fallback=True),
                'long_signal_threshold': self.config.getint('FUTURES_STRATEGY', 'long_signal_threshold', fallback=5),
                'short_signal_threshold': self.config.getint('FUTURES_STRATEGY', 'short_signal_threshold', fallback=-7),
                'leverage': self.config.getint('FUTURES_STRATEGY', 'leverage', fallback=10),
                'position_size_usd': self.config.getint('FUTURES_STRATEGY', 'position_size_usd', fallback=500),
                'min_signal_strength': self.config.getint('FUTURES_STRATEGY', 'min_signal_strength', fallback=4),
                'require_trend_alignment': self.config.getboolean('FUTURES_STRATEGY', 'require_trend_alignment', fallback=True),
                'min_trend_strength': self.config.getint('FUTURES_STRATEGY', 'min_trend_strength', fallback=5),
                'min_time_between_trades': self.config.getint('FUTURES_STRATEGY', 'min_time_between_trades', fallback=3600),
                'trend_bullish_threshold': self.config.getint('FUTURES_STRATEGY', 'trend_bullish_threshold', fallback=3),
                'trend_bearish_threshold': self.config.getint('FUTURES_STRATEGY', 'trend_bearish_threshold', fallback=-3)
            },
            
            # Neutral Strategy
            'neutral_strategy': {
                'enabled': self.config.getboolean('NEUTRAL_STRATEGY', 'enabled', fallback=True),
                'lot_size': self.config.getint('NEUTRAL_STRATEGY', 'lot_size', fallback=1),
                'leverage_percentage': self.config.getfloat('NEUTRAL_STRATEGY', 'leverage_percentage', fallback=50.0),
                'strike_distance': self.config.getint('NEUTRAL_STRATEGY', 'strike_distance', fallback=8),
                'expiry_days': self.config.getint('NEUTRAL_STRATEGY', 'expiry_days', fallback=1),
                'trailing_stop_loss_pct': self.config.getfloat('NEUTRAL_STRATEGY', 'trailing_stop_loss_pct', fallback=20.0),
                'profit_target_pct': self.config.getfloat('NEUTRAL_STRATEGY', 'profit_target_pct', fallback=30.0),
                'stop_loss_pct': self.config.getfloat('NEUTRAL_STRATEGY', 'stop_loss_pct', fallback=50.0),
                'min_time_between_neutral_trades': self.config.getint('NEUTRAL_STRATEGY', 'min_time_between_neutral_trades', fallback=7200)
            },
            
            # Dollar-Based Risk Management
            'dollar_based_risk': {
                'enabled': self.config.getboolean('DOLLAR_BASED_RISK', 'enabled', fallback=False),
                'stop_loss_usd': self.config.getfloat('DOLLAR_BASED_RISK', 'stop_loss_usd', fallback=100.0),
                'take_profit_usd': self.config.getfloat('DOLLAR_BASED_RISK', 'take_profit_usd', fallback=200.0),
                'trailing_stop_usd': self.config.getfloat('DOLLAR_BASED_RISK', 'trailing_stop_usd', fallback=50.0),
                'quick_profit_usd': self.config.getfloat('DOLLAR_BASED_RISK', 'quick_profit_usd', fallback=60.0),
                'max_risk_usd': self.config.getfloat('DOLLAR_BASED_RISK', 'max_risk_usd', fallback=150.0),
                'daily_loss_limit_usd': self.config.getfloat('DOLLAR_BASED_RISK', 'daily_loss_limit_usd', fallback=500.0)
            },
            
            # Currency Conversion
            'USD': self.config.getfloat('DOLLAR_COVERSION_FACTOR', 'USD', fallback=85.0),
            
            # Trading Timing Configuration
            'trading_timing': {
                'trading_start_time': self.config.get('TRADING_TIMING', 'trading_start_time', fallback='17:30'),
                'timezone': self.config.get('TRADING_TIMING', 'timezone', fallback='Asia/Kolkata')
            },
            
            # ATR-Based Exits Configuration
            'atr_exits': {
                'enabled': self.config.getboolean('ATR_EXITS', 'enabled', fallback=False),
                'atr_period': self.config.getint('ATR_EXITS', 'atr_period', fallback=14),
                'stop_loss_atr_multiplier': self.config.getfloat('ATR_EXITS', 'stop_loss_atr_multiplier', fallback=2.0),
                'take_profit_atr_multiplier': self.config.getfloat('ATR_EXITS', 'take_profit_atr_multiplier', fallback=3.0),
                'trailing_atr_multiplier': self.config.getfloat('ATR_EXITS', 'trailing_atr_multiplier', fallback=1.5),
                'buffer_zone_atr_multiplier': self.config.getfloat('ATR_EXITS', 'buffer_zone_atr_multiplier', fallback=0.3),
                'volume_threshold_percentile': self.config.getfloat('ATR_EXITS', 'volume_threshold_percentile', fallback=70),
                'hunting_zone_offset': self.config.getfloat('ATR_EXITS', 'hunting_zone_offset', fallback=5)
            },
            
            # Logging Configuration
            'logging': {
                'console_level': self.config.get('LOGGING', 'console_level', fallback='INFO'),
                'log_file': self.config.get('LOGGING', 'log_file', fallback='delta_btc_trading.log'),
                'file_level': self.config.get('LOGGING', 'file_level', fallback='DEBUG')
            }
        }
        
        return config_dict
    
    def save_config(self, config_data: Dict[str, Any]) -> bool:
        """Save configuration data to application.config file"""
        try:
            # Clear existing config
            for section in self.config.sections():
                self.config.remove_section(section)
            
            # Add API Configuration
            self.config.add_section('API_CONFIGURATION')
            self.config.set('API_CONFIGURATION', 'api_key', str(config_data.get('api_key', '')))
            self.config.set('API_CONFIGURATION', 'api_secret', str(config_data.get('api_secret', '')))
            self.config.set('API_CONFIGURATION', 'paper_trading', str(config_data.get('paper_trading', True)).lower())
            
            # Add Portfolio Settings
            self.config.add_section('PORTFOLIO_SETTINGS')
            self.config.set('PORTFOLIO_SETTINGS', 'portfolio_size', str(config_data.get('portfolio_size', 100000.0)))
            self.config.set('PORTFOLIO_SETTINGS', 'position_size_usd', str(config_data.get('position_size_usd', 500.0)))
            self.config.set('PORTFOLIO_SETTINGS', 'max_daily_loss', str(config_data.get('max_daily_loss', 2000.0)))
            self.config.set('PORTFOLIO_SETTINGS', 'max_positions', str(config_data.get('max_positions', 2)))
            self.config.set('PORTFOLIO_SETTINGS', 'leverage', str(config_data.get('leverage', 100)))
            
            
            
            
            # Add Futures Strategy
            self.config.add_section('FUTURES_STRATEGY')
            futures = config_data.get('futures_strategy', {})
            self.config.set('FUTURES_STRATEGY', 'enabled', str(futures.get('enabled', True)).lower())
            self.config.set('FUTURES_STRATEGY', 'long_signal_threshold', str(futures.get('long_signal_threshold', 5)))
            self.config.set('FUTURES_STRATEGY', 'short_signal_threshold', str(futures.get('short_signal_threshold', -7)))
            self.config.set('FUTURES_STRATEGY', 'leverage', str(futures.get('leverage', 10)))
            self.config.set('FUTURES_STRATEGY', 'position_size_usd', str(futures.get('position_size_usd', 500)))
            self.config.set('FUTURES_STRATEGY', 'min_signal_strength', str(futures.get('min_signal_strength', 4)))
            self.config.set('FUTURES_STRATEGY', 'require_trend_alignment', str(futures.get('require_trend_alignment', True)).lower())
            self.config.set('FUTURES_STRATEGY', 'min_trend_strength', str(futures.get('min_trend_strength', 5)))
            self.config.set('FUTURES_STRATEGY', 'min_time_between_trades', str(futures.get('min_time_between_trades', 3600)))
            self.config.set('FUTURES_STRATEGY', 'trend_bullish_threshold', str(futures.get('trend_bullish_threshold', 3)))
            self.config.set('FUTURES_STRATEGY', 'trend_bearish_threshold', str(futures.get('trend_bearish_threshold', -3)))
            
            # Add Neutral Strategy
            self.config.add_section('NEUTRAL_STRATEGY')
            neutral = config_data.get('neutral_strategy', {})
            self.config.set('NEUTRAL_STRATEGY', 'enabled', str(neutral.get('enabled', True)).lower())
            self.config.set('NEUTRAL_STRATEGY', 'lot_size', str(neutral.get('lot_size', 1)))
            self.config.set('NEUTRAL_STRATEGY', 'leverage_percentage', str(neutral.get('leverage_percentage', 50.0)))
            self.config.set('NEUTRAL_STRATEGY', 'strike_distance', str(neutral.get('strike_distance', 8)))
            self.config.set('NEUTRAL_STRATEGY', 'expiry_days', str(neutral.get('expiry_days', 1)))
            self.config.set('NEUTRAL_STRATEGY', 'trailing_stop_loss_pct', str(neutral.get('trailing_stop_loss_pct', 20.0)))
            self.config.set('NEUTRAL_STRATEGY', 'profit_target_pct', str(neutral.get('profit_target_pct', 30.0)))
            self.config.set('NEUTRAL_STRATEGY', 'stop_loss_pct', str(neutral.get('stop_loss_pct', 50.0)))
            self.config.set('NEUTRAL_STRATEGY', 'min_time_between_neutral_trades', str(neutral.get('min_time_between_neutral_trades', 7200)))
            
            # Add Dollar-Based Risk Management
            self.config.add_section('DOLLAR_BASED_RISK')
            dollar_risk = config_data.get('dollar_based_risk', {})
            self.config.set('DOLLAR_BASED_RISK', 'enabled', str(dollar_risk.get('enabled', False)).lower())
            self.config.set('DOLLAR_BASED_RISK', 'stop_loss_usd', str(dollar_risk.get('stop_loss_usd', 100.0)))
            self.config.set('DOLLAR_BASED_RISK', 'take_profit_usd', str(dollar_risk.get('take_profit_usd', 200.0)))
            self.config.set('DOLLAR_BASED_RISK', 'trailing_stop_usd', str(dollar_risk.get('trailing_stop_usd', 50.0)))
            self.config.set('DOLLAR_BASED_RISK', 'quick_profit_usd', str(dollar_risk.get('quick_profit_usd', 60.0)))
            self.config.set('DOLLAR_BASED_RISK', 'max_risk_usd', str(dollar_risk.get('max_risk_usd', 150.0)))
            self.config.set('DOLLAR_BASED_RISK', 'daily_loss_limit_usd', str(dollar_risk.get('daily_loss_limit_usd', 500.0)))
            
            # Add Trading Timing Configuration
            self.config.add_section('TRADING_TIMING')
            trading_timing = config_data.get('trading_timing', {})
            self.config.set('TRADING_TIMING', 'trading_start_time', str(trading_timing.get('trading_start_time', '17:30')))
            self.config.set('TRADING_TIMING', 'timezone', str(trading_timing.get('timezone', 'Asia/Kolkata')))
            
            # Add ATR Exits Configuration
            self.config.add_section('ATR_EXITS')
            atr_exits = config_data.get('atr_exits', {})
            self.config.set('ATR_EXITS', 'enabled', str(atr_exits.get('enabled', False)).lower())
            self.config.set('ATR_EXITS', 'atr_period', str(atr_exits.get('atr_period', 14)))
            self.config.set('ATR_EXITS', 'stop_loss_atr_multiplier', str(atr_exits.get('stop_loss_atr_multiplier', 2.0)))
            self.config.set('ATR_EXITS', 'take_profit_atr_multiplier', str(atr_exits.get('take_profit_atr_multiplier', 3.0)))
            self.config.set('ATR_EXITS', 'trailing_atr_multiplier', str(atr_exits.get('trailing_atr_multiplier', 1.5)))
            self.config.set('ATR_EXITS', 'buffer_zone_atr_multiplier', str(atr_exits.get('buffer_zone_atr_multiplier', 0.3)))
            self.config.set('ATR_EXITS', 'volume_threshold_percentile', str(atr_exits.get('volume_threshold_percentile', 70)))
            self.config.set('ATR_EXITS', 'hunting_zone_offset', str(atr_exits.get('hunting_zone_offset', 5)))
            
            # Add Logging Configuration
            self.config.add_section('LOGGING')
            logging_config = config_data.get('logging', {})
            self.config.set('LOGGING', 'console_level', str(logging_config.get('console_level', 'INFO')))
            self.config.set('LOGGING', 'log_file', str(logging_config.get('log_file', 'delta_btc_trading.log')))
            self.config.set('LOGGING', 'file_level', str(logging_config.get('file_level', 'DEBUG')))
            
            # Write to file
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_config_value(self, section: str, key: str, fallback: Any = None) -> Any:
        """Get a specific configuration value"""
        try:
            if section == 'API_CONFIGURATION':
                if key == 'paper_trading':
                    return self.config.getboolean(section, key, fallback=fallback)
                else:
                    return self.config.get(section, key, fallback=fallback)
            elif section in ['PORTFOLIO_SETTINGS', 'NEUTRAL_STRATEGY', 'DOLLAR_BASED_RISK', 'FUTURES_STRATEGY']:
                if key in ['portfolio_size', 'position_size_usd', 'max_daily_loss', 'min_premium', 'max_premium',
                          'stop_loss_pct', 'take_profit_pct', 'quick_profit_pct', 'quick_profit_time_hours',
                          'time_exit_hours', 'volume_threshold', 'leverage_percentage', 'trailing_stop_loss_pct',
                          'profit_target_pct', 'max_risk_per_trade', 'stop_loss_usd', 'take_profit_usd', 
                          'trailing_stop_usd', 'quick_profit_usd', 'max_risk_usd', 'daily_loss_limit_usd']:
                    return self.config.getfloat(section, key, fallback=fallback)
                elif key in ['max_positions', 'order_timeout', 'min_signal_strength', 'min_time_between_trades',
                            'bullish_signal_threshold', 'bearish_signal_threshold', 'min_trend_strength',
                            'lot_size', 'strike_distance', 'expiry_days', 'min_time_between_neutral_trades']:
                    return self.config.getint(section, key, fallback=fallback)
                else:
                    return self.config.getboolean(section, key, fallback=fallback)
            else:
                return self.config.get(section, key, fallback=fallback)
        except Exception:
            return fallback
    
    def update_config_value(self, section: str, key: str, value: Any) -> bool:
        """Update a specific configuration value"""
        try:
            if not self.config.has_section(section):
                self.config.add_section(section)
            
            self.config.set(section, key, str(value))
            
            # Write to file
            with open(self.config_file, 'w') as configfile:
                self.config.write(configfile)
            
            return True
        except Exception as e:
            print(f"Error updating configuration: {e}")
            return False

# Global instance
config_manager = ConfigManager()