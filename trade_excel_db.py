"""
Trade History Excel Database Module
Handles reading and writing trade data to Excel file
"""

import pandas as pd
import os
from datetime import datetime
import logging
from typing import Dict, List, Optional, Any

class TradeExcelDB:
    def __init__(self, excel_path: str = "Trade History.xlsx"):
        self.excel_path = excel_path
        self.logger = logging.getLogger(__name__)
        self.columns = self._get_column_structure()
        self._initialize_excel_file()
    
    def _get_column_structure(self) -> List[str]:
        """Define the complete column structure for the trade history Excel"""
        return [
            # Trade Identification (A-E)
            'Trade_ID', 'Entry_Date', 'Entry_Time', 'Exit_Date', 'Exit_Time',
            
            # Strategy & Signal (F-M)
            'Strategy_Type', 'Signal_Strength_5m', 'Trend_Direction_1h', 'Trend_Strength_1h',
            'Trade_Logic', 'Technical_Indicators', 'Volume_Confirmation', 'Risk_Score',
            
            # Options Details (N-W)
            'Primary_Symbol', 'Secondary_Symbol', 'Primary_Strike', 'Secondary_Strike',
            'Primary_Option_Type', 'Secondary_Option_Type', 'Expiry_Date', 'Days_to_Expiry',
            'BTC_Price_at_Entry', 'BTC_Price_at_Exit',
            
            # Position & Risk Management (X-AH)
            'Lot_Size', 'Quantity_Primary', 'Quantity_Secondary', 'Entry_Premium_Primary',
            'Entry_Premium_Secondary', 'Total_Premium_Received', 'Margin_Deployed',
            'Stop_Loss_Percentage', 'Take_Profit_Percentage', 'Trailing_Stop_Percentage',
            
            # Performance (AI-AR)
            'Exit_Premium_Primary', 'Exit_Premium_Secondary', 'Total_Premium_Paid',
            'Gross_PnL', 'Commission_Fees', 'Net_PnL', 'ROI_Percentage',
            'Duration_Minutes', 'Max_Drawdown', 'Peak_Profit',
            
            # Exit & Analysis (AS-BA)
            'Exit_Reason', 'Trade_Status', 'Market_Condition', 'Volatility_Impact',
            'Greeks_Delta', 'Greeks_Gamma', 'Greeks_Theta', 'Greeks_Vega', 'Notes'
        ]
    
    def _initialize_excel_file(self):
        """Initialize Excel file with headers if it doesn't exist"""
        try:
            if not os.path.exists(self.excel_path):
                # Create empty DataFrame with column structure
                df = pd.DataFrame(columns=self.columns)
                df.to_excel(self.excel_path, index=False, sheet_name='Trade_History')
                self.logger.info(f"Created new Excel file: {self.excel_path}")
            else:
                # Verify existing file has correct structure
                existing_df = pd.read_excel(self.excel_path, sheet_name='Trade_History')
                missing_columns = set(self.columns) - set(existing_df.columns)
                if missing_columns:
                    self.logger.warning(f"Missing columns in Excel: {missing_columns}")
                    # Add missing columns
                    for col in missing_columns:
                        existing_df[col] = None
                    # Reorder columns to match structure
                    existing_df = existing_df.reindex(columns=self.columns)
                    existing_df.to_excel(self.excel_path, index=False, sheet_name='Trade_History')
                    self.logger.info(f"Updated Excel file structure")
        except Exception as e:
            self.logger.error(f"Error initializing Excel file: {e}")
            raise
    
    def add_trade_entry(self, trade_data: Dict[str, Any]) -> bool:
        """Add a new trade entry to Excel"""
        try:
            # Read existing data
            df = pd.read_excel(self.excel_path, sheet_name='Trade_History')
            
            # Create new trade row
            new_trade = self._format_trade_data(trade_data)
            
            # Add to DataFrame
            df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
            
            # Save to Excel
            df.to_excel(self.excel_path, index=False, sheet_name='Trade_History')
            
            self.logger.info(f"Added trade entry: {new_trade['Trade_ID']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding trade entry: {e}")
            return False
    
    def update_trade_exit(self, trade_id: str, exit_data: Dict[str, Any]) -> bool:
        """Update an existing trade with exit information"""
        try:
            # Read existing data
            df = pd.read_excel(self.excel_path, sheet_name='Trade_History')
            
            # Find trade by ID
            trade_index = df[df['Trade_ID'] == trade_id].index
            if len(trade_index) == 0:
                self.logger.warning(f"Trade ID not found: {trade_id}")
                return False
            
            # Update exit information
            idx = trade_index[0]
            exit_datetime = datetime.now()
            
            df.at[idx, 'Exit_Date'] = exit_datetime.strftime('%Y-%m-%d')
            df.at[idx, 'Exit_Time'] = exit_datetime.strftime('%H:%M:%S')
            df.at[idx, 'BTC_Price_at_Exit'] = exit_data.get('btc_price_at_exit', 0)
            df.at[idx, 'Exit_Premium_Primary'] = exit_data.get('exit_premium_primary', 0)
            df.at[idx, 'Exit_Premium_Secondary'] = exit_data.get('exit_premium_secondary', 0)
            df.at[idx, 'Total_Premium_Paid'] = exit_data.get('total_premium_paid', 0)
            df.at[idx, 'Gross_PnL'] = exit_data.get('gross_pnl', 0)
            df.at[idx, 'Commission_Fees'] = exit_data.get('commission_fees', 0)
            df.at[idx, 'Net_PnL'] = exit_data.get('net_pnl', 0)
            df.at[idx, 'ROI_Percentage'] = exit_data.get('roi_percentage', 0)
            df.at[idx, 'Exit_Reason'] = exit_data.get('exit_reason', 'UNKNOWN')
            df.at[idx, 'Trade_Status'] = 'CLOSED'
            df.at[idx, 'Duration_Minutes'] = exit_data.get('duration_minutes', 0)
            df.at[idx, 'Max_Drawdown'] = exit_data.get('max_drawdown', 0)
            df.at[idx, 'Peak_Profit'] = exit_data.get('peak_profit', 0)
            df.at[idx, 'Notes'] = exit_data.get('notes', '')
            
            # Calculate ROI if not provided
            if df.at[idx, 'ROI_Percentage'] == 0 and df.at[idx, 'Margin_Deployed'] > 0:
                roi = (df.at[idx, 'Net_PnL'] / df.at[idx, 'Margin_Deployed']) * 100
                df.at[idx, 'ROI_Percentage'] = round(roi, 2)
            
            # Save to Excel
            df.to_excel(self.excel_path, index=False, sheet_name='Trade_History')
            
            self.logger.info(f"Updated trade exit: {trade_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating trade exit: {e}")
            return False
    
    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Get all trades from Excel file"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='Trade_History')
            # Convert to list of dictionaries
            trades = df.to_dict('records')
            
            # Clean up NaN values
            for trade in trades:
                for key, value in trade.items():
                    if pd.isna(value):
                        trade[key] = None
            
            self.logger.info(f"Retrieved {len(trades)} trades from Excel")
            return trades
            
        except Exception as e:
            self.logger.error(f"Error reading trades from Excel: {e}")
            return []
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get only active trades"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='Trade_History')
            active_df = df[df['Trade_Status'] == 'ACTIVE']
            trades = active_df.to_dict('records')
            
            # Clean up NaN values
            for trade in trades:
                for key, value in trade.items():
                    if pd.isna(value):
                        trade[key] = None
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error reading active trades: {e}")
            return []
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get trade statistics summary"""
        try:
            df = pd.read_excel(self.excel_path, sheet_name='Trade_History')
            
            if len(df) == 0:
                return {
                    'total_trades': 0,
                    'active_trades': 0,
                    'closed_trades': 0,
                    'total_pnl': 0,
                    'win_rate': 0,
                    'avg_roi': 0
                }
            
            # Calculate statistics
            total_trades = len(df)
            active_trades = len(df[df['Trade_Status'] == 'ACTIVE'])
            closed_trades = len(df[df['Trade_Status'] == 'CLOSED'])
            
            # PnL calculations (only for closed trades)
            closed_df = df[df['Trade_Status'] == 'CLOSED']
            total_pnl = closed_df['Net_PnL'].sum() if len(closed_df) > 0 else 0
            winning_trades = len(closed_df[closed_df['Net_PnL'] > 0])
            win_rate = (winning_trades / len(closed_df) * 100) if len(closed_df) > 0 else 0
            avg_roi = closed_df['ROI_Percentage'].mean() if len(closed_df) > 0 else 0
            
            return {
                'total_trades': total_trades,
                'active_trades': active_trades,
                'closed_trades': closed_trades,
                'total_pnl': round(total_pnl, 2),
                'win_rate': round(win_rate, 2),
                'avg_roi': round(avg_roi, 2),
                'winning_trades': winning_trades,
                'losing_trades': len(closed_df[closed_df['Net_PnL'] <= 0])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating trade statistics: {e}")
            return {}
    
    def _format_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format trade data to match Excel column structure"""
        entry_datetime = datetime.now()
        
        # Create formatted trade entry
        formatted_trade = {
            # Trade Identification
            'Trade_ID': trade_data.get('trade_id', f"T_{entry_datetime.strftime('%Y%m%d_%H%M%S')}"),
            'Entry_Date': entry_datetime.strftime('%Y-%m-%d'),
            'Entry_Time': entry_datetime.strftime('%H:%M:%S'),
            'Exit_Date': None,
            'Exit_Time': None,
            
            # Strategy & Signal
            'Strategy_Type': trade_data.get('strategy_type', 'UNKNOWN'),
            'Signal_Strength_5m': trade_data.get('signal_strength_5m', 0),
            'Trend_Direction_1h': trade_data.get('trend_direction_1h', 'NEUTRAL'),
            'Trend_Strength_1h': trade_data.get('trend_strength_1h', 0),
            'Trade_Logic': trade_data.get('trade_logic', ''),
            'Technical_Indicators': trade_data.get('technical_indicators', ''),
            'Volume_Confirmation': trade_data.get('volume_confirmation', ''),
            'Risk_Score': trade_data.get('risk_score', 5),
            
            # Options Details
            'Primary_Symbol': trade_data.get('primary_symbol', ''),
            'Secondary_Symbol': trade_data.get('secondary_symbol', ''),
            'Primary_Strike': trade_data.get('primary_strike', 0),
            'Secondary_Strike': trade_data.get('secondary_strike', 0),
            'Primary_Option_Type': trade_data.get('primary_option_type', ''),
            'Secondary_Option_Type': trade_data.get('secondary_option_type', ''),
            'Expiry_Date': trade_data.get('expiry_date', ''),
            'Days_to_Expiry': trade_data.get('days_to_expiry', 0),
            'BTC_Price_at_Entry': trade_data.get('btc_price_at_entry', 0),
            'BTC_Price_at_Exit': None,
            
            # Position & Risk Management
            'Lot_Size': trade_data.get('lot_size', 0),
            'Quantity_Primary': trade_data.get('quantity_primary', 0),
            'Quantity_Secondary': trade_data.get('quantity_secondary', 0),
            'Entry_Premium_Primary': trade_data.get('entry_premium_primary', 0),
            'Entry_Premium_Secondary': trade_data.get('entry_premium_secondary', 0),
            'Total_Premium_Received': trade_data.get('total_premium_received', 0),
            'Margin_Deployed': trade_data.get('margin_deployed', 0),
            'Stop_Loss_Percentage': trade_data.get('stop_loss_percentage', 0),
            'Take_Profit_Percentage': trade_data.get('take_profit_percentage', 0),
            'Trailing_Stop_Percentage': trade_data.get('trailing_stop_percentage', 0),
            
            # Performance (will be filled on exit)
            'Exit_Premium_Primary': None,
            'Exit_Premium_Secondary': None,
            'Total_Premium_Paid': None,
            'Gross_PnL': None,
            'Commission_Fees': None,
            'Net_PnL': None,
            'ROI_Percentage': None,
            'Duration_Minutes': None,
            'Max_Drawdown': None,
            'Peak_Profit': None,
            
            # Exit & Analysis
            'Exit_Reason': None,
            'Trade_Status': 'ACTIVE',
            'Market_Condition': trade_data.get('market_condition', ''),
            'Volatility_Impact': trade_data.get('volatility_impact', ''),
            'Greeks_Delta': trade_data.get('greeks_delta', 0),
            'Greeks_Gamma': trade_data.get('greeks_gamma', 0),
            'Greeks_Theta': trade_data.get('greeks_theta', 0),
            'Greeks_Vega': trade_data.get('greeks_vega', 0),
            'Notes': trade_data.get('notes', '')
        }
        
        return formatted_trade

# Example usage and testing
if __name__ == "__main__":
    # Test the Excel database
    logging.basicConfig(level=logging.INFO)
    
    db = TradeExcelDB("Trade History.xlsx")
    
    # Test adding a sample trade
    sample_trade = {
        'trade_id': 'TEST_001',
        'strategy_type': 'DIRECTIONAL_CALL',
        'signal_strength_5m': 7,
        'trend_direction_1h': 'BULLISH',
        'trend_strength_1h': 8,
        'trade_logic': 'Strong bullish signal with trend alignment. VWAP above, Parabolic SAR bullish.',
        'technical_indicators': 'VWAP: Above, SAR: Bullish, RSI: 65',
        'volume_confirmation': 'Above average volume',
        'risk_score': 3,
        'primary_symbol': 'C-BTC-112000-080925',
        'primary_strike': 112000,
        'primary_option_type': 'CALL',
        'expiry_date': '2025-09-08',
        'days_to_expiry': 1,
        'btc_price_at_entry': 111500,
        'lot_size': 10,
        'quantity_primary': 10,
        'entry_premium_primary': 0.005,
        'total_premium_received': 0.05,
        'margin_deployed': 500,
        'stop_loss_percentage': 50,
        'take_profit_percentage': 100,
        'trailing_stop_percentage': 20,
        'market_condition': 'Trending Bull Market',
        'notes': 'Test trade entry'
    }
    
    # Add trade
    success = db.add_trade_entry(sample_trade)
    print(f"Trade added: {success}")
    
    # Get statistics
    stats = db.get_trade_statistics()
    print(f"Statistics: {stats}")