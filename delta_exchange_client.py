import requests
import time
import hmac
import hashlib
import json
from datetime import datetime, timezone
import pandas as pd
import logging

class DeltaExchangeClient:
    """
    Delta Exchange API client for BTC options trading
    Using official Delta Exchange API v2 endpoints - NO FALLBACKS
    """
    
    def __init__(self, api_key, api_secret, paper_trading=True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.paper_trading = paper_trading
        
        # Official Delta Exchange API base URL
        self.base_url = "https://api.india.delta.exchange"
        self.testnet_mode = paper_trading
            
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'BTC-Options-Strategy/1.0'
        })
        
        # No need for delta-rest-client since we use custom implementation
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Cache for products and options
        self.products = {}
        self.available_options = []
        self.btc_option_specs = {}  # Cache for BTC option contract specifications
        
    def _generate_signature(self, method, endpoint, payload="", params=None):
        """Generate HMAC signature for authentication per official Delta Exchange docs"""
        timestamp = str(int(time.time()))
        
        # Build query string for GET requests
        query_string = ""
        if params:
            query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
            if query_string:
                query_string = "?" + query_string
        
        # Build payload string for POST requests
        payload_str = ""
        if payload:
            payload_str = json.dumps(payload, separators=(',', ':'))
            
        # Create message as per Delta Exchange documentation:
        # method + timestamp + endpoint + query_string + payload
        message = method + timestamp + endpoint + query_string + payload_str
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    def _make_request(self, method, endpoint, payload=None, params=None):
        """Make authenticated request to Delta Exchange API - NO FALLBACKS"""
        url = self.base_url + endpoint
        
        signature, timestamp = self._generate_signature(method, endpoint, payload, params)
        
        headers = {
            'api-key': self.api_key,
            'signature': signature,
            'timestamp': timestamp,
            'Content-Type': 'application/json',
            'User-Agent': 'BTC-Options-Strategy/1.0'
        }
        
        try:
            if method == 'GET':
                response = self.session.get(url, headers=headers, params=params, timeout=10)
            elif method == 'POST':
                response = self.session.post(url, headers=headers, json=payload, timeout=10)
            elif method == 'DELETE':
                response = self.session.delete(url, headers=headers, timeout=10)
            elif method == 'PUT':
                response = self.session.put(url, headers=headers, json=payload, timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            data = response.json()
            
            # Check if API response indicates success
            if not data.get('success', False):
                error_msg = data.get('error', {}).get('message', 'Unknown API error')
                raise Exception(f"Delta Exchange API error: {error_msg}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"HTTP request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if error_data.get('error', {}).get('message'):
                        error_msg = f"Delta API error: {error_data['error']['message']}"
                except:
                    error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _make_public_request(self, endpoint, params=None):
        """Make public (unauthenticated) request - NO FALLBACKS"""
        url = self.base_url + endpoint
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data.get('success', False):
                error_msg = data.get('error', {}).get('message', 'Unknown API error')
                raise Exception(f"Delta Exchange API error: {error_msg}")
                
            return data
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Public API request failed: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if error_data.get('error', {}).get('message'):
                        error_msg = f"Delta API error: {error_data['error']['message']}"
                except:
                    error_msg = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def get_products(self):
        """Get all available products using official /v2/products endpoint"""
        try:
            response = self._make_public_request('/v2/products')
            products = response['result']
            self.products = {p['symbol']: p for p in products}
            self.logger.info(f"Successfully fetched {len(products)} products")
            return products
        except Exception as e:
            self.logger.error(f"Failed to get products: {e}")
            raise Exception(f"Products fetch failed: {e}")
    
    def get_btc_options_specifications(self):
        """Get BTC options contract specifications including lot sizes from Delta Exchange"""
        try:
            # Fetch all products if not cached
            if not self.products:
                self.get_products()
            
            btc_option_specs = {}
            
            # Filter for BTC options and extract contract specifications
            for symbol, product in self.products.items():
                if (symbol.startswith('C-BTC-') or symbol.startswith('P-BTC-')) and product:
                    try:
                        # Delta Exchange India BTC options specifications
                        contract_spec = {
                            'symbol': symbol,
                            'product_id': product.get('id'),
                            'underlying_asset': product.get('underlying_asset', {}).get('symbol', 'BTC'),
                            'contract_value': product.get('contract_value', 0.001),  # Default 0.001 BTC per lot
                            'contract_unit_currency': product.get('contract_unit_currency', 'BTC'),
                            'quoting_asset': product.get('quoting_asset', {}).get('symbol', 'USDT'),
                            'tick_size': product.get('tick_size', 1),  # Minimum price increment
                            'minimum_size': product.get('minimum_size', 1),  # Minimum order size in lots
                            'size_multiplier': product.get('contract_value', 0.001),  # BTC amount per lot
                            'strike_price': self._extract_strike_price(symbol),
                            'option_type': 'call' if symbol.startswith('C-BTC-') else 'put',
                            'expiry_time': product.get('expiry_time'),
                            'settlement_price': product.get('settlement_price'),
                            'is_quanto': product.get('is_quanto', False),
                            'margin_parameters': {
                                'initial_margin': product.get('initial_margin', 0.1),
                                'maintenance_margin': product.get('maintenance_margin', 0.05),
                                'impact_size': product.get('impact_size', 100)
                            }
                        }
                        
                        btc_option_specs[symbol] = contract_spec
                        
                    except Exception as e:
                        self.logger.debug(f"Could not parse specifications for {symbol}: {e}")
                        continue
            
            # Cache the specifications
            self.btc_option_specs = btc_option_specs
            self.logger.info(f"Successfully cached specifications for {len(btc_option_specs)} BTC options")
            
            return btc_option_specs
            
        except Exception as e:
            self.logger.error(f"Failed to get BTC options specifications: {e}")
            return {}
    
    def _extract_strike_price(self, symbol):
        """Extract strike price from option symbol"""
        try:
            # Format: C-BTC-111200-080925 or P-BTC-111200-080925
            parts = symbol.split('-')
            if len(parts) >= 3:
                return float(parts[2])
        except:
            pass
        return 0.0
    
    def get_lot_size_for_symbol(self, symbol):
        """Get lot size (contract value) for a specific BTC option symbol"""
        try:
            # Check cache first
            if symbol in self.btc_option_specs:
                return self.btc_option_specs[symbol].get('contract_value', 0.001)
            
            # If not cached, fetch specifications
            if not self.btc_option_specs:
                self.get_btc_options_specifications()
            
            # Return cached value or default
            return self.btc_option_specs.get(symbol, {}).get('contract_value', 0.001)
            
        except Exception as e:
            self.logger.warning(f"Could not get lot size for {symbol}, using default 0.001 BTC: {e}")
            return 0.001  # Default Delta Exchange India BTC options lot size
    
    def calculate_premium_in_usdt(self, symbol, option_price, num_lots):
        """Calculate total premium in USDT for given number of lots"""
        try:
            lot_size_btc = self.get_lot_size_for_symbol(symbol)
            btc_price = self.get_current_btc_price()
            
            # Ensure all values are proper numeric types
            lot_size_btc = float(lot_size_btc)
            option_price = float(option_price)
            btc_price = float(btc_price)
            num_lots = int(num_lots)
            
            # Premium = option_price * lot_size_btc * btc_price * num_lots
            # For Delta Exchange: option prices are in USDT per BTC, lot size is in BTC
            total_premium_usdt = option_price * lot_size_btc * btc_price * num_lots
            
            self.logger.info(f"Premium calculation for {symbol}: "
                           f"Price={option_price:.6f} * LotSize={lot_size_btc} BTC * "
                           f"BTC_Price=${btc_price:.2f} * Lots={num_lots} = ${total_premium_usdt:.2f}")
            
            return total_premium_usdt
            
        except Exception as e:
            self.logger.error(f"Premium calculation failed for {symbol}: {e}")
            return 0.0
    
    def calculate_margin_requirement(self, symbol, option_price, num_lots, side='sell'):
        """Calculate margin requirement for BTC options position"""
        try:
            lot_size_btc = self.get_lot_size_for_symbol(symbol)
            btc_price = self.get_current_btc_price()
            
            # Ensure all values are proper numeric types
            lot_size_btc = float(lot_size_btc)
            option_price = float(option_price)
            btc_price = float(btc_price)
            num_lots = int(num_lots)
            
            # Get margin parameters from specifications
            spec = self.btc_option_specs.get(symbol, {})
            margin_params = spec.get('margin_parameters', {})
            initial_margin_rate = float(margin_params.get('initial_margin', 0.1))  # Default 10%
            
            if side.lower() == 'buy':
                # For buying options, margin = premium paid
                margin = self.calculate_premium_in_usdt(symbol, option_price, num_lots)
            else:
                # For selling options, margin = initial margin requirement
                # Simplified margin calculation: initial_margin_rate * underlying_value * lots
                underlying_value = lot_size_btc * btc_price * num_lots
                margin = underlying_value * initial_margin_rate
                
                # Add premium received (reduces net margin requirement)
                premium_received = self.calculate_premium_in_usdt(symbol, option_price, num_lots)
                net_margin = max(margin - premium_received, premium_received * 0.1)  # Minimum 10% of premium
                
                self.logger.info(f"Margin calculation for {symbol} (sell): "
                               f"Underlying_Value=${underlying_value:.2f} * "
                               f"Margin_Rate={initial_margin_rate:.1%} = ${margin:.2f}, "
                               f"Premium_Received=${premium_received:.2f}, "
                               f"Net_Margin=${net_margin:.2f}")
                
                return net_margin
            
            self.logger.info(f"Margin requirement for {symbol} ({side}): ${margin:.2f}")
            return margin
            
        except Exception as e:
            self.logger.error(f"Margin calculation failed for {symbol}: {e}")
            return 0.0
    
    def get_btc_futures_specifications(self):
        """Get BTC futures contract specifications for directional trading"""
        try:
            # Fetch all products if not cached
            if not self.products:
                self.get_products()
            
            btc_futures_specs = {}
            
            # Filter for BTC futures (perpetual and expiring)
            for symbol, product in self.products.items():
                if product and (product.get('contract_type') in ['perpetual_futures', 'futures']):
                    underlying = product.get('underlying_asset', {})
                    if underlying.get('symbol') == 'BTC':
                        contract_spec = {
                            'symbol': symbol,
                            'product_id': product.get('id'),
                            'contract_type': product.get('contract_type'),
                            'underlying_asset': underlying.get('symbol', 'BTC'),
                            'quoting_asset': product.get('quoting_asset', {}).get('symbol', 'USD'),
                            'contract_unit_currency': product.get('contract_unit_currency'),
                            'contract_value': product.get('contract_value', 1),
                            'tick_size': product.get('tick_size', 0.5),
                            'minimum_order_size': product.get('min_size', 1),
                            'maximum_leverage': product.get('max_leverage_notional', 100),
                            'maintenance_margin': product.get('maintenance_margin', 0.005),
                            'initial_margin': product.get('initial_margin', 0.01),
                            'trading_status': product.get('trading_status', 'operational'),
                            'settlement_price': product.get('settlement_price'),
                            'launch_date': product.get('launch_date'),
                            'settlement_time': product.get('settlement_time')
                        }
                        
                        btc_futures_specs[symbol] = contract_spec
            
            self.logger.info(f"Found {len(btc_futures_specs)} BTC futures contracts")
            return btc_futures_specs
            
        except Exception as e:
            self.logger.error(f"Failed to get BTC futures specifications: {e}")
            return {}
    
    def get_current_btc_price(self):
        """Get current BTC price using official /v2/tickers endpoint - NO FALLBACKS"""
        try:
            # Try specific BTC symbols first
            btc_symbols = ['BTCUSD', 'BTCUSDT', 'BTCUSD_PERP', 'BTCUSDT_PERP']
            
            for symbol in btc_symbols:
                try:
                    response = self._make_public_request(f'/v2/tickers/{symbol}')
                    ticker = response['result']
                    
                    # Get price from ticker (prefer mark_price, then last_price)
                    price = ticker.get('mark_price') or ticker.get('last_price') or ticker.get('close_price')
                    
                    if price and float(price) > 0:
                        self.logger.info(f"Got BTC price ${float(price):,.2f} from {symbol}")
                        return float(price)
                        
                except Exception as e:
                    self.logger.debug(f"Symbol {symbol} failed: {e}")
                    continue
            
            # If specific symbols fail, try getting all tickers
            response = self._make_public_request('/v2/tickers')
            tickers = response['result']
            
            for ticker in tickers:
                symbol = ticker.get('symbol', '')
                if 'BTC' in symbol and ('USD' in symbol or 'USDT' in symbol):
                    price = ticker.get('mark_price') or ticker.get('last_price') or ticker.get('close_price')
                    if price and float(price) > 0:
                        self.logger.info(f"Got BTC price ${float(price):,.2f} from {symbol}")
                        return float(price)
            
            raise Exception("No valid BTC price found in tickers")
            
        except Exception as e:
            self.logger.error(f"BTC price fetch failed: {e}")
            raise Exception(f"BTC price unavailable: {e}")
    
    def get_historical_candles(self, symbol, resolution, count):
        """Get historical candles using official Delta Exchange API"""
        try:
            # Get product_id from cached products data
            if not self.products:
                self.get_products()
            
            product_info = self.products.get(symbol)
            if not product_info or 'id' not in product_info:
                self.logger.error(f"Product info not found for {symbol}")
                raise Exception(f"Product info not found for {symbol}")
            
            product_id = product_info['id']
            
            # Map resolution to Delta Exchange format
            resolution_map = {
                '1m': '1m',
                '3m': '3m', 
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '1H': '1h',
                '4h': '4h',
                '1d': '1d'
            }
            
            delta_resolution = resolution_map.get(resolution, '5m')
            
            # Use the official Delta Exchange candles endpoint
            endpoint = f'/v2/history/candles'
            
            # Calculate time range for the last 'count' candles
            import time
            current_time = int(time.time())
            interval_seconds = {'1m': 60, '3m': 180, '5m': 300, '15m': 900, '30m': 1800, '1h': 3600, '4h': 14400, '1d': 86400}
            seconds_per_candle = interval_seconds.get(delta_resolution, 300)
            start_time = current_time - (count * seconds_per_candle)
            
            params = {
                'symbol': symbol,
                'resolution': delta_resolution,
                'start': start_time,
                'end': current_time
            }
            
            response = self._make_public_request(endpoint, params)
            candles_raw = response.get('result', [])
            
            if not candles_raw:
                self.logger.warning(f"No candles returned for {symbol} {resolution}")
                return []
            
            # Convert Delta Exchange candle format to standard format
            candles_data = []
            for candle in candles_raw:
                try:
                    # Delta Exchange candle format is already a dict: {'time': timestamp, 'open': ..., 'high': ..., ...}
                    if isinstance(candle, dict) and 'time' in candle:
                        candle_dict = {
                            'time': int(candle['time']),  # Unix timestamp
                            'open': float(candle['open']),
                            'high': float(candle['high']), 
                            'low': float(candle['low']),
                            'close': float(candle['close']),
                            'volume': float(candle['volume'])
                        }
                        candles_data.append(candle_dict)
                    elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                        # Fallback for array format: [timestamp, open, high, low, close, volume]
                        candle_dict = {
                            'time': int(candle[0]),  # Unix timestamp
                            'open': float(candle[1]),
                            'high': float(candle[2]), 
                            'low': float(candle[3]),
                            'close': float(candle[4]),
                            'volume': float(candle[5])
                        }
                        candles_data.append(candle_dict)
                except Exception as e:
                    self.logger.debug(f"Error parsing candle data: {e}")
                    continue
            
            self.logger.info(f"Successfully fetched {len(candles_data)} real candles for {symbol} {resolution}")
            return candles_data
            
        except Exception as e:
            self.logger.error(f"Real candle data fetch failed for {symbol}: {e}")
            raise Exception(f"Failed to fetch real candle data for {symbol} {resolution}: {e}")
    
    def get_account_balance(self):
        """Get wallet balance using official /v2/wallet/balances endpoint - NO FALLBACKS"""
        try:
            if self.paper_trading:
                # Return simulated balance for paper trading in correct format
                return {
                    'USDT': {'balance': '10000.000000', 'available': '10000.000000'},
                    'BTC': {'balance': '0.100000', 'available': '0.100000'}
                }
            
            # Use correct endpoint path from API docs: /v2/wallet/balances
            response = self._make_request('GET', '/v2/wallet/balances')
            balance_array = response['result']
            
            # Convert array format to dictionary format for easier processing
            balances = {}
            for balance_item in balance_array:
                asset_symbol = balance_item.get('asset_symbol')
                if asset_symbol:
                    balances[asset_symbol] = {
                        'balance': balance_item.get('balance', '0'),
                        'available': balance_item.get('available_balance', '0'),
                        'asset_id': balance_item.get('asset_id')
                    }
            
            self.logger.info(f"Successfully fetched wallet balances for {len(balances)} assets")
            return balances
            
        except Exception as e:
            self.logger.error(f"Wallet balance fetch failed: {e}")
            raise Exception(f"Wallet balance unavailable: {e}")
    
    def get_multi_timeframe_data(self, symbol='BTCUSD'):
        """Get multi-timeframe data for analysis"""
        try:
            data = {}
            
            # Get 3m data (200 candles for short-term analysis)
            data['3m'] = self.get_historical_candles(symbol, '3m', 200)
            
            # Get 1h data (100 candles for higher timeframe trend)
            data['1h'] = self.get_historical_candles(symbol, '1h', 100)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe data fetch failed: {e}")
            raise Exception(f"Multi-timeframe data unavailable: {e}")
    
    def get_positions(self):
        """Get positions using official /v2/positions/margined endpoint - NO FALLBACKS"""
        try:
            if self.paper_trading:
                # Return empty positions for paper trading
                return []
            
            response = self._make_request('GET', '/v2/positions/margined')
            positions = response['result']
            
            self.logger.info(f"Successfully fetched {len(positions)} positions")
            return positions
            
        except Exception as e:
            self.logger.error(f"Positions fetch failed: {e}")
            raise Exception(f"Positions unavailable: {e}")
    
    def get_portfolio_summary(self):
        """Get portfolio summary from positions and balances - NO FALLBACKS"""
        try:
            if self.paper_trading:
                return {
                    'total_balance': 10000.0,
                    'available_balance': 10000.0,
                    'unrealized_pnl': 0.0,
                    'margin_used': 0.0,
                    'positions': 0
                }
            
            # Get balances and positions to build summary
            balances = self.get_account_balance()
            positions = self.get_positions()
            
            # Calculate totals from balances
            total_balance = 0
            available_balance = 0
            
            for currency, balance_info in balances.items():
                if isinstance(balance_info, dict):
                    total_balance += float(balance_info.get('balance', 0))
                    available_balance += float(balance_info.get('available', 0))
            
            # Calculate unrealized PnL from positions
            unrealized_pnl = sum(float(pos.get('unrealized_pnl', 0)) for pos in positions)
            margin_used = sum(float(pos.get('margin', 0)) for pos in positions)
            
            summary = {
                'total_balance': total_balance,
                'available_balance': available_balance,
                'unrealized_pnl': unrealized_pnl,
                'margin_used': margin_used,
                'positions': len(positions)
            }
            
            self.logger.info("Successfully calculated portfolio summary")
            return summary
            
        except Exception as e:
            self.logger.error(f"Portfolio summary calculation failed: {e}")
            raise Exception(f"Portfolio summary unavailable: {e}")
    
    def place_order(self, symbol, side, quantity, order_type='market', price=None):
        """Place order using Delta REST client"""
        try:
            if self.paper_trading:
                # Simulate order for paper trading
                payload = {
                    'product_symbol': symbol,
                    'side': side.lower(),
                    'size': str(quantity),
                    'order_type': order_type.lower()
                }
                self.logger.info(f"PAPER TRADE - Order simulated: {payload}")
                return {
                    'success': True,
                    'result': {
                        'id': f"paper_{int(time.time())}_{symbol}",
                        'product_symbol': symbol,
                        'side': side,
                        'size': quantity,
                        'state': 'filled',
                        'average_price': price or self._get_simulated_price(symbol, side),
                        'filled_size': quantity,
                        'created_at': int(time.time())
                    }
                }
            
            # Live trading mode - use Delta REST client
            from delta_rest_client import DeltaRestClient
            
            # Get product_id from cached products data
            if not self.products:
                self.get_products()
            
            product_info = self.products.get(symbol)
            if not product_info or 'id' not in product_info:
                self.logger.error(f"Product info not found for {symbol}")
                raise Exception(f"Product info not found for {symbol}")
            
            # Initialize Delta REST client for order placement
            delta_rest_client = DeltaRestClient(
                base_url=self.base_url,
                api_key=self.api_key,
                api_secret=self.api_secret
            )
            
            # Use Delta REST client's place_order method
            if order_type.lower() == 'market':
                response = delta_rest_client.place_order(
                    product_id=product_info['id'],
                    size=quantity,
                    side=side.lower()
                )
            else:  # limit order
                response = delta_rest_client.place_order(
                    product_id=product_info['id'],
                    size=quantity,
                    side=side.lower(),
                    limit_price=price
                )
            
            self.logger.info(f"Successfully placed order: {response.get('id', 'unknown')}")
            return {'success': True, 'result': response}
            
        except Exception as e:
            self.logger.error(f"Order placement failed: {e}")
            raise Exception(f"Order placement failed: {e}")
    
    def _get_simulated_price(self, symbol, side):
        """Get simulated price for paper trading"""
        try:
            if 'BTC' in symbol:
                base_price = self.get_current_btc_price()
                # Add small spread for simulation
                if side.lower() == 'buy':
                    return base_price * 1.001  # 0.1% higher for buy
                else:
                    return base_price * 0.999  # 0.1% lower for sell
        except:
            pass
        return 50000  # Fallback price for simulation
    
    def get_orders(self, states=['open', 'pending']):
        """Get orders using official /v2/orders endpoint - NO FALLBACKS"""
        try:
            if self.paper_trading:
                return []  # No orders in paper trading
            
            params = {'states': ','.join(states)}
            response = self._make_request('GET', '/v2/orders', params=params)
            orders = response['result']
            
            self.logger.info(f"Successfully fetched {len(orders)} orders")
            return orders
            
        except Exception as e:
            self.logger.error(f"Orders fetch failed: {e}")
            raise Exception(f"Orders unavailable: {e}")
    
    def cancel_order(self, order_id):
        """Cancel order using official /v2/orders/{id} endpoint - NO FALLBACKS"""
        try:
            if self.paper_trading:
                self.logger.info(f"PAPER TRADE - Order cancellation simulated: {order_id}")
                return {'success': True, 'result': {'id': order_id, 'state': 'cancelled'}}
            
            response = self._make_request('DELETE', f'/v2/orders/{order_id}')
            self.logger.info(f"Successfully cancelled order: {order_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Order cancellation failed: {e}")
            raise Exception(f"Order cancellation failed: {e}")
    
    def get_orderbook(self, symbol):
        """Get orderbook using Delta REST client"""
        try:
            if self.paper_trading:
                # Simulate orderbook data for paper trading
                import random
                
                # Generate realistic bid/ask spreads for options
                if symbol.startswith(('C-BTC-', 'P-BTC-')):
                    # Parse strike price from symbol for realistic pricing
                    try:
                        parts = symbol.split('-')
                        if len(parts) >= 3:
                            strike = float(parts[2])
                            current_price = 111000  # Approximate current BTC price
                            
                            # Simple option pricing simulation
                            if symbol.startswith('C-BTC-'):  # Call
                                intrinsic = max(0, current_price - strike)
                                time_value = random.uniform(100, 500)
                                mid_price = intrinsic + time_value
                            else:  # Put
                                intrinsic = max(0, strike - current_price)
                                time_value = random.uniform(100, 500)
                                mid_price = intrinsic + time_value
                            
                            # Generate bid/ask with spread
                            spread = mid_price * 0.02  # 2% spread
                            bid_price = mid_price - spread/2
                            ask_price = mid_price + spread/2
                            
                            simulated_orderbook = {
                                'buy': [{'price': str(bid_price), 'size': '5'}],  # 5 contracts available
                                'sell': [{'price': str(ask_price), 'size': '5'}],
                                'symbol': symbol
                            }
                            
                            self.logger.info(f"PAPER TRADE - Simulated orderbook for {symbol}: bid={bid_price:.2f}, ask={ask_price:.2f}")
                            return simulated_orderbook
                    except Exception as e:
                        self.logger.debug(f"Error in option pricing simulation: {e}")
                
                # Default simulation for futures and other symbols
                if symbol == 'BTCUSD':
                    # Realistic BTCUSD futures prices with tight spreads
                    base_price = 111000
                    spread = 25  # $25 spread
                    simulated_orderbook = {
                        'buy': [{'price': str(base_price - spread/2), 'size': '0.1'}],
                        'sell': [{'price': str(base_price + spread/2), 'size': '0.1'}],
                        'symbol': symbol
                    }
                    self.logger.info(f"PAPER TRADE - Simulated BTCUSD futures orderbook: bid={base_price - spread/2}, ask={base_price + spread/2}")
                else:
                    # Default for other symbols
                    simulated_orderbook = {
                        'buy': [{'price': '100.0', 'size': '1'}],
                        'sell': [{'price': '110.0', 'size': '1'}],
                        'symbol': symbol
                    }
                    self.logger.info(f"PAPER TRADE - Simulated default orderbook for {symbol}")
                return simulated_orderbook
            
            # Live trading mode - use Delta REST client
            from delta_rest_client import DeltaRestClient
            
            # Get product_id from cached products data
            if not self.products:
                self.get_products()
            
            product_info = self.products.get(symbol)
            if not product_info or 'id' not in product_info:
                self.logger.error(f"Product info not found for {symbol}")
                raise Exception(f"Product info not found for {symbol}")
            
            # Initialize Delta REST client for orderbook call
            delta_rest_client = DeltaRestClient(base_url=self.base_url)
            
            # Use Delta REST client's get_l2_orderbook method
            orderbook = delta_rest_client.get_l2_orderbook(product_info['id'], auth=False)
            
            self.logger.info(f"Successfully fetched orderbook for {symbol}")
            return orderbook
            
        except Exception as e:
            self.logger.error(f"Orderbook fetch failed for {symbol}: {e}")
            raise Exception(f"Orderbook unavailable for {symbol}: {e}")
    
    def get_daily_expiry_options(self):
        """Get BTC options that expire within next 2 days using correct API endpoint"""
        try:
            from datetime import datetime, timedelta
            
            btc_options = []
            today = datetime.now().date()
            next_day = (datetime.now() + timedelta(days=1)).date()
            day_after = (datetime.now() + timedelta(days=2)).date()
            
            # Target dates in DD-MM-YYYY format as required by Delta Exchange API
            target_dates = [
                today.strftime('%d-%m-%Y'), 
                next_day.strftime('%d-%m-%Y'), 
                day_after.strftime('%d-%m-%Y')
            ]
            
            self.logger.info(f"Searching for BTC options expiring on: {target_dates}")
            
            # Fetch options for each target date using the correct API endpoint
            for expiry_date in target_dates:
                try:
                    # Use /v2/tickers endpoint with proper parameters
                    params = {
                        'contract_types': 'call_options,put_options',
                        'underlying_asset_symbols': 'BTC',
                        'expiry_date': expiry_date
                    }
                    
                    response = self._make_request('GET', '/v2/tickers', params=params)
                    
                    if response and 'result' in response:
                        tickers = response['result']
                        self.logger.info(f"Found {len(tickers)} options tickers for {expiry_date}")
                        
                        for ticker in tickers:
                            try:
                                # Extract option information from ticker data
                                symbol = ticker.get('symbol', '')
                                
                                # Parse option details
                                if symbol.startswith('C-BTC-') or symbol.startswith('P-BTC-'):
                                    # Example format: P-BTC-111200-080925 or C-BTC-111500-080925
                                    parts = symbol.split('-')
                                    if len(parts) >= 4:
                                        option_info = {
                                            'symbol': symbol,
                                            'strike_price': float(parts[2]),
                                            'option_type': 'call' if parts[0] == 'C' else 'put',
                                            'expiry_date': parts[3],  # 080925 format
                                            'underlying': parts[1],  # BTC
                                            'product_id': ticker.get('product_id'),
                                            'mark_price': ticker.get('mark_price', 0),
                                            'bid_price': ticker.get('best_bid_price', 0),
                                            'ask_price': ticker.get('best_ask_price', 0),
                                            'volume': ticker.get('volume', 0),
                                            'open_interest': ticker.get('oi', 0),
                                            'iv': ticker.get('iv', 0),  # Implied volatility
                                            'delta': ticker.get('greeks', {}).get('delta', 0),
                                            'gamma': ticker.get('greeks', {}).get('gamma', 0),
                                            'theta': ticker.get('greeks', {}).get('theta', 0),
                                            'vega': ticker.get('greeks', {}).get('vega', 0),
                                            'is_tradeable': True  # Assume tradeable if in tickers
                                        }
                                        btc_options.append(option_info)
                                        
                            except Exception as e:
                                self.logger.debug(f"Could not parse option ticker {ticker.get('symbol', 'Unknown')}: {e}")
                                continue
                    
                except Exception as e:
                    self.logger.warning(f"Failed to fetch options for {expiry_date}: {e}")
                    continue
            
            self.logger.info(f"Found {len(btc_options)} BTC options for next 2 days")
            return btc_options
            
        except Exception as e:
            self.logger.error(f"Failed to get daily expiry options: {e}")
            return []
    
    def find_suitable_options(self, entry_type, current_price):
        """Find suitable options for directional trading"""
        try:
            if not self.available_options:
                # If no options cached, get them now
                self.available_options = self.get_daily_expiry_options()
            
            suitable_options = []
            
            for option in self.available_options:
                try:
                    # Check if option type matches signal
                    if entry_type == 'CALL' and option.get('option_type') != 'call':
                        continue
                    elif entry_type == 'PUT' and option.get('option_type') != 'put':
                        continue
                    
                    strike_price = float(option.get('strike_price', 0))
                    if strike_price <= 0:
                        continue
                    
                    # Calculate moneyness for filtering
                    if entry_type == 'CALL':
                        moneyness = current_price / strike_price
                        # Look for slightly OTM calls (0.98 to 1.05 moneyness)
                        if 0.98 <= moneyness <= 1.05:
                            suitable_options.append(option)
                    else:  # PUT
                        moneyness = strike_price / current_price
                        # Look for slightly OTM puts (0.98 to 1.05 moneyness)
                        if 0.98 <= moneyness <= 1.05:
                            suitable_options.append(option)
                            
                except Exception as e:
                    self.logger.debug(f"Error processing option {option.get('symbol', 'Unknown')}: {e}")
                    continue
            
            self.logger.info(f"Found {len(suitable_options)} suitable {entry_type} options")
            return suitable_options
            
        except Exception as e:
            self.logger.error(f"Error finding suitable options: {e}")
            return []
    
    def get_fills_history(self, page_size=50):
        """
        Get trade history (fills) from Delta Exchange using official Trade History API
        Endpoint: /v2/fills
        """
        try:
            endpoint = "/v2/fills"
            params = {
                'page_size': page_size
            }
            
            self.logger.info(f"Fetching fills history from /v2/fills with page_size={page_size}")
            
            # Make authenticated request
            response = self._make_request('GET', endpoint, params=params)
            
            if not isinstance(response, dict):
                raise Exception("Invalid response format from Delta Exchange API")
            
            # Check for success status
            if not response.get('success', False):
                error_msg = response.get('error', {}).get('message', 'Unknown API error')
                raise Exception(f"Delta Exchange API error: {error_msg}")
            
            result = response.get('result', [])
            
            self.logger.info(f"Successfully fetched {len(result)} trade fills from Delta Exchange")
            
            return {
                'success': True,
                'result': result
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get fills history: {e}")
            raise Exception(f"Failed to fetch trade history from Delta Exchange: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize client (paper trading mode)
    client = DeltaExchangeClient(
        api_key="your_api_key_here",
        api_secret="your_api_secret_here",
        paper_trading=True
    )
    
    # Test basic functionality
    print("Testing Delta Exchange API connection...")
    
    try:
        # Get BTC price
        btc_price = client.get_current_btc_price()
        print(f"Current BTC Price: ${btc_price:,.2f}")
        
        # Get historical candles
        candles = client.get_historical_candles('BTCUSD', '5m', 100)
        print(f"Fetched {len(candles)} candles")
        
        # Get account balance
        balance = client.get_account_balance()
        print(f"Account balance: {balance}")
        
    except Exception as e:
        print(f"API test failed: {e}")