#!/usr/bin/env python3
"""
Historical Data Downloader for BTC Trading Bot
Downloads and stores historical candle data locally in CSV format for fast backtesting
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager
from logger_config import get_logger


class DataDownloader:
    """Downloads and manages historical trading data storage"""
    
    def __init__(self):
        """Initialize the data downloader"""
        self.config = config_manager.get_all_config()
        
        # Setup Delta Exchange client
        self.delta_client = DeltaExchangeClient(
            api_key=self.config['api_key'],
            api_secret=self.config['api_secret'],
            paper_trading=True  # Always use paper trading for data download
        )
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Data directory
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.metadata_file = os.path.join(self.data_dir, 'last_update.json')
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Supported timeframes and their API mappings
        self.timeframes = {
            '1m': {'api_name': '1m', 'csv_file': 'BTCUSD_1m.csv'},
            '3m': {'api_name': '3m', 'csv_file': 'BTCUSD_3m.csv'},
            '1h': {'api_name': '1h', 'csv_file': 'BTCUSD_1h.csv'}
        }
        
        # How much historical data to download (2 years)
        self.history_days = 730
        
    def get_metadata(self) -> Dict:
        """Get metadata about last update times"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not read metadata file: {e}")
        
        return {
            'last_updates': {},
            'data_ranges': {},
            'created': datetime.now().isoformat()
        }
    
    def save_metadata(self, metadata: Dict):
        """Save metadata about updates"""
        try:
            metadata['last_modified'] = datetime.now().isoformat()
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Could not save metadata: {e}")
    
    def download_timeframe_data(self, timeframe: str, force_refresh: bool = False) -> bool:
        """
        Download historical data for a specific timeframe
        
        Args:
            timeframe: '1m', '3m', or '1h'
            force_refresh: If True, download all data fresh
            
        Returns:
            True if successful, False otherwise
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return False
        
        config = self.timeframes[timeframe]
        csv_file = os.path.join(self.data_dir, config['csv_file'])
        
        try:
            self.logger.info(f"Downloading {timeframe} data for BTCUSD...")
            
            # Calculate how many candles we need for 3 months
            minutes_per_candle = {'1m': 1, '3m': 3, '1h': 60}[timeframe]
            total_minutes = self.history_days * 24 * 60
            candles_needed = total_minutes // minutes_per_candle
            
            # Limit to reasonable amounts to avoid API issues  
            max_candles = {'1m': 50000, '3m': 25000, '1h': 18000}[timeframe]  # Increased limits for 2 years
            candles_to_fetch = min(candles_needed, max_candles)
            
            self.logger.info(f"Fetching {candles_to_fetch} candles for {timeframe}")
            
            # Download data from Delta Exchange
            candle_data = self.delta_client.get_historical_candles(
                symbol='BTCUSD',
                resolution=config['api_name'],
                count=candles_to_fetch
            )
            
            if not candle_data:
                self.logger.error(f"No data received for {timeframe}")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(candle_data)
            
            # Ensure proper column names and types
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            
            # Sort by timestamp (oldest first)
            df = df.sort_values('timestamp')
            
            # Select and reorder columns for CSV
            df_csv = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Save to CSV
            df_csv.to_csv(csv_file, index=False)
            
            # Update metadata
            metadata = self.get_metadata()
            metadata['last_updates'][timeframe] = datetime.now().isoformat()
            metadata['data_ranges'][timeframe] = {
                'start_time': df['timestamp'].min().isoformat(),
                'end_time': df['timestamp'].max().isoformat(),
                'total_candles': len(df_csv)
            }
            self.save_metadata(metadata)
            
            self.logger.info(f"Successfully saved {len(df_csv)} {timeframe} candles to {csv_file}")
            self.logger.info(f"Data range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {timeframe} data: {e}")
            return False
    
    def download_all_timeframes(self, force_refresh: bool = False) -> Dict[str, bool]:
        """
        Download data for all supported timeframes
        
        Args:
            force_refresh: If True, download all data fresh
            
        Returns:
            Dictionary with success status for each timeframe
        """
        results = {}
        
        self.logger.info("=== Starting Historical Data Download ===")
        
        for timeframe in self.timeframes.keys():
            self.logger.info(f"Processing {timeframe} timeframe...")
            results[timeframe] = self.download_timeframe_data(timeframe, force_refresh)
            
            # Add delay between API calls to be respectful
            if timeframe != list(self.timeframes.keys())[-1]:  # Not the last one
                time.sleep(1)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"=== Download Complete: {successful}/{total} timeframes successful ===")
        
        if successful == total:
            self.logger.info("All data downloaded successfully! Ready for fast backtesting.")
        else:
            failed = [tf for tf, success in results.items() if not success]
            self.logger.warning(f"Failed timeframes: {failed}")
        
        return results
    
    def get_data_status(self) -> Dict:
        """Get status of locally stored data"""
        metadata = self.get_metadata()
        status = {
            'metadata': metadata,
            'files': {}
        }
        
        for timeframe, config in self.timeframes.items():
            csv_file = os.path.join(self.data_dir, config['csv_file'])
            
            if os.path.exists(csv_file):
                try:
                    # Get file info
                    stat = os.stat(csv_file)
                    file_size = stat.st_size
                    
                    # Quick read to get row count
                    df = pd.read_csv(csv_file)
                    row_count = len(df)
                    
                    if row_count > 0:
                        # Get data range
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        data_start = df['timestamp'].min()
                        data_end = df['timestamp'].max()
                        
                        status['files'][timeframe] = {
                            'exists': True,
                            'file_size': file_size,
                            'row_count': row_count,
                            'data_start': data_start.isoformat(),
                            'data_end': data_end.isoformat(),
                            'age_hours': (datetime.now() - datetime.fromisoformat(data_end.isoformat().replace('T', ' ').replace('+00:00', ''))).total_seconds() / 3600
                        }
                    else:
                        status['files'][timeframe] = {
                            'exists': True,
                            'file_size': file_size,
                            'row_count': 0,
                            'error': 'Empty file'
                        }
                        
                except Exception as e:
                    status['files'][timeframe] = {
                        'exists': True,
                        'error': str(e)
                    }
            else:
                status['files'][timeframe] = {
                    'exists': False
                }
        
        return status
    
    def should_update_data(self, max_age_hours: int = 1) -> Dict[str, bool]:
        """
        Check which timeframes need updating based on age
        
        Args:
            max_age_hours: Maximum age in hours before data is considered stale
            
        Returns:
            Dictionary indicating which timeframes need updates
        """
        status = self.get_data_status()
        needs_update = {}
        
        for timeframe in self.timeframes.keys():
            file_status = status['files'].get(timeframe, {})
            
            if not file_status.get('exists', False):
                needs_update[timeframe] = True
            elif file_status.get('row_count', 0) == 0:
                needs_update[timeframe] = True
            elif 'age_hours' in file_status and file_status['age_hours'] > max_age_hours:
                needs_update[timeframe] = True
            else:
                needs_update[timeframe] = False
        
        return needs_update
    
    def download_incremental_update(self, timeframe: str) -> bool:
        """
        Download only new data since last update (incremental)
        
        Args:
            timeframe: '1m', '3m', or '1h'
            
        Returns:
            True if successful, False otherwise
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return False
        
        config = self.timeframes[timeframe]
        csv_file = os.path.join(self.data_dir, config['csv_file'])
        
        try:
            # Check if we have existing data
            if not os.path.exists(csv_file):
                self.logger.info(f"No existing data for {timeframe}, doing full download")
                return self.download_timeframe_data(timeframe)
            
            # Read existing data to find the latest timestamp
            existing_df = pd.read_csv(csv_file)
            if existing_df.empty:
                self.logger.info(f"Empty existing data for {timeframe}, doing full download")
                return self.download_timeframe_data(timeframe)
            
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            latest_timestamp = existing_df['timestamp'].max()
            
            self.logger.info(f"Latest {timeframe} data: {latest_timestamp}")
            
            # Calculate how many new candles we might need (1 day worth + buffer)
            minutes_per_candle = {'1m': 1, '3m': 3, '1h': 60}[timeframe]
            candles_per_day = (24 * 60) // minutes_per_candle
            candles_to_fetch = candles_per_day + 100  # Extra buffer
            
            self.logger.info(f"Fetching {candles_to_fetch} recent candles for incremental update")
            
            # Download recent data
            candle_data = self.delta_client.get_historical_candles(
                symbol='BTCUSD',
                resolution=config['api_name'],
                count=candles_to_fetch
            )
            
            if not candle_data:
                self.logger.warning(f"No new data received for {timeframe}")
                return True  # Not an error, just no new data
            
            # Convert to DataFrame
            new_df = pd.DataFrame(candle_data)
            new_df['timestamp'] = pd.to_datetime(new_df['time'], unit='s')
            new_df['open'] = new_df['open'].astype(float)
            new_df['high'] = new_df['high'].astype(float)
            new_df['low'] = new_df['low'].astype(float)
            new_df['close'] = new_df['close'].astype(float)
            new_df['volume'] = new_df['volume'].astype(float)
            
            # Filter out data we already have
            new_data = new_df[new_df['timestamp'] > latest_timestamp]
            
            if new_data.empty:
                self.logger.info(f"No new data available for {timeframe}")
                return True
            
            self.logger.info(f"Found {len(new_data)} new candles for {timeframe}")
            
            # Append new data to existing
            new_data_csv = new_data[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            new_data_csv = new_data_csv.sort_values('timestamp')
            
            # Append to existing CSV
            new_data_csv.to_csv(csv_file, mode='a', header=False, index=False)
            
            # Update metadata
            metadata = self.get_metadata()
            metadata['last_updates'][timeframe] = datetime.now().isoformat()
            if timeframe not in metadata['data_ranges']:
                metadata['data_ranges'][timeframe] = {}
            
            # Update data range info
            all_data = pd.concat([existing_df, new_data_csv], ignore_index=True)
            metadata['data_ranges'][timeframe]['end_time'] = all_data['timestamp'].max().isoformat()
            metadata['data_ranges'][timeframe]['total_candles'] = len(all_data)
            
            self.save_metadata(metadata)
            
            self.logger.info(f"Successfully appended {len(new_data)} new {timeframe} candles")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in incremental update for {timeframe}: {e}")
            return False
    
    def daily_update_all_timeframes(self) -> Dict[str, bool]:
        """
        Perform daily incremental updates for all timeframes
        
        Returns:
            Dictionary with success status for each timeframe
        """
        results = {}
        
        self.logger.info("=== Starting Daily Incremental Update ===")
        
        for timeframe in self.timeframes.keys():
            self.logger.info(f"Incremental update for {timeframe}...")
            results[timeframe] = self.download_incremental_update(timeframe)
            
            # Add delay between API calls
            if timeframe != list(self.timeframes.keys())[-1]:
                time.sleep(1)
        
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"=== Daily Update Complete: {successful}/{total} timeframes updated ===")
        
        return results
    
    def download_historical_chunks(self, timeframe: str, target_days: int = 730) -> bool:
        """
        Download historical data in multiple chunks to overcome API limits
        
        Args:
            timeframe: '1m', '3m', or '1h'
            target_days: Number of days of historical data to download
            
        Returns:
            True if successful, False otherwise
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return False
        
        config = self.timeframes[timeframe]
        csv_file = os.path.join(self.data_dir, config['csv_file'])
        
        try:
            self.logger.info(f"Starting chunked download for {timeframe} ({target_days} days)...")
            
            # Determine chunk size based on API limits
            api_limit = 2000  # Delta Exchange API limit: max 2000 candles per response
            minutes_per_candle = {'1m': 1, '3m': 3, '1h': 60}[timeframe]
            
            # Calculate candles per day and chunk size in days
            candles_per_day = (24 * 60) // minutes_per_candle
            days_per_chunk = max(1, api_limit // candles_per_day)  # How many days we can get per chunk
            
            self.logger.info(f"Chunk strategy: {days_per_chunk} days per chunk ({candles_per_day} candles/day)")
            
            # Calculate number of chunks needed
            num_chunks = (target_days + days_per_chunk - 1) // days_per_chunk  # Round up
            
            all_chunks = []
            successful_chunks = 0
            
            # Calculate timestamps going backwards from current time
            import time
            current_timestamp = int(time.time())
            seconds_per_day = 24 * 60 * 60
            
            for chunk_idx in range(num_chunks):
                try:
                    # Calculate candles to fetch for this chunk
                    remaining_days = target_days - (chunk_idx * days_per_chunk)
                    chunk_days = min(days_per_chunk, remaining_days)
                    
                    # Calculate explicit start and end timestamps for this chunk
                    end_timestamp = current_timestamp - (chunk_idx * days_per_chunk * seconds_per_day)
                    start_timestamp = end_timestamp - (chunk_days * seconds_per_day)
                    
                    # Convert to readable dates for logging
                    from datetime import datetime
                    start_date = datetime.fromtimestamp(start_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    end_date = datetime.fromtimestamp(end_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    
                    self.logger.info(f"Chunk {chunk_idx + 1}/{num_chunks}: Requesting {chunk_days} days from {start_date} to {end_date}")
                    self.logger.info(f"Timestamps: start={start_timestamp}, end={end_timestamp}")
                    
                    # Download this chunk with explicit timestamps
                    candle_data = self.delta_client.get_historical_candles_by_timestamps(
                        symbol='BTCUSD',
                        resolution=config['api_name'],
                        start_timestamp=start_timestamp,
                        end_timestamp=end_timestamp
                    )
                    
                    if not candle_data:
                        self.logger.warning(f"No data received for chunk {chunk_idx + 1}")
                        continue
                    
                    # Convert to DataFrame
                    chunk_df = pd.DataFrame(candle_data)
                    chunk_df['timestamp'] = pd.to_datetime(chunk_df['time'], unit='s')
                    
                    # Process columns
                    chunk_df['open'] = chunk_df['open'].astype(float)
                    chunk_df['high'] = chunk_df['high'].astype(float)
                    chunk_df['low'] = chunk_df['low'].astype(float)
                    chunk_df['close'] = chunk_df['close'].astype(float)
                    chunk_df['volume'] = chunk_df['volume'].astype(float)
                    
                    all_chunks.append(chunk_df)
                    successful_chunks += 1
                    
                    self.logger.info(f"Chunk {chunk_idx + 1} successful: {len(chunk_df)} candles "
                                   f"({chunk_df['timestamp'].min()} to {chunk_df['timestamp'].max()})")
                    
                    # Add delay between chunks to be respectful to API
                    if chunk_idx < num_chunks - 1:
                        time.sleep(2)
                        
                except Exception as e:
                    self.logger.error(f"Error downloading chunk {chunk_idx + 1}: {e}")
                    continue
            
            if not all_chunks:
                self.logger.error(f"No successful chunks downloaded for {timeframe}")
                return False
            
            # Combine all chunks
            self.logger.info(f"Combining {successful_chunks} chunks...")
            combined_df = pd.concat(all_chunks, ignore_index=True)
            
            # Remove duplicates and sort
            combined_df = combined_df.drop_duplicates(subset=['timestamp'])
            combined_df = combined_df.sort_values('timestamp')
            
            # Select final columns for CSV
            final_df = combined_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
            
            # Save to CSV
            final_df.to_csv(csv_file, index=False)
            
            # Update metadata
            metadata = self.get_metadata()
            metadata['last_updates'][timeframe] = datetime.now().isoformat()
            metadata['data_ranges'][timeframe] = {
                'start_time': final_df['timestamp'].min().isoformat(),
                'end_time': final_df['timestamp'].max().isoformat(),
                'total_candles': len(final_df),
                'chunks_downloaded': successful_chunks,
                'method': 'chunked_download'
            }
            self.save_metadata(metadata)
            
            self.logger.info(f"Successfully saved {len(final_df)} {timeframe} candles from {successful_chunks} chunks")
            self.logger.info(f"Date range: {final_df['timestamp'].min()} to {final_df['timestamp'].max()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in chunked download for {timeframe}: {e}")
            return False
    
    def build_full_historical_dataset(self, target_days: int = 730) -> Dict[str, bool]:
        """
        Build complete historical dataset using chunked downloads
        
        Args:
            target_days: Number of days of historical data to download
            
        Returns:
            Dictionary with success status for each timeframe
        """
        results = {}
        
        self.logger.info(f"=== Building Full Historical Dataset ({target_days} days) ===")
        
        # Only download 3m and 1h (skip 1m as it's not needed and has very short range)
        priority_timeframes = ['1h', '3m']
        
        for timeframe in priority_timeframes:
            self.logger.info(f"Starting chunked download for {timeframe}...")
            results[timeframe] = self.download_historical_chunks(timeframe, target_days)
            
            if results[timeframe]:
                self.logger.info(f"✅ {timeframe} chunked download completed successfully")
            else:
                self.logger.error(f"❌ {timeframe} chunked download failed")
            
            # Add delay between timeframes
            time.sleep(5)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        if successful == total:
            self.logger.info(f"🎉 Full historical dataset build complete: {successful}/{total} timeframes successful")
        else:
            self.logger.warning(f"⚠️ Partial success: {successful}/{total} timeframes completed")
        
        return results


if __name__ == "__main__":
    # CLI interface for manual data download
    import sys
    
    downloader = DataDownloader()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'download':
            # Download all data
            force = '--force' in sys.argv
            results = downloader.download_all_timeframes(force_refresh=force)
            
        elif command == 'status':
            # Show data status
            status = downloader.get_data_status()
            print("\n=== Data Status ===")
            for timeframe, info in status['files'].items():
                if info.get('exists', False):
                    if 'row_count' in info:
                        print(f"{timeframe}: {info['row_count']} candles ({info['file_size']/1024:.1f} KB)")
                        if 'age_hours' in info:
                            print(f"  Age: {info['age_hours']:.1f} hours")
                    else:
                        print(f"{timeframe}: ERROR - {info.get('error', 'Unknown')}")
                else:
                    print(f"{timeframe}: Not downloaded")
            
        elif command == 'check':
            # Check what needs updating
            needs_update = downloader.should_update_data()
            print("\n=== Update Check ===")
            for timeframe, needs in needs_update.items():
                status = "NEEDS UPDATE" if needs else "OK"
                print(f"{timeframe}: {status}")
                
        elif command == 'build':
            # Build full historical dataset with chunked downloads
            target_days = 730  # Default 2 years
            if '--days' in sys.argv:
                try:
                    days_idx = sys.argv.index('--days')
                    if days_idx + 1 < len(sys.argv):
                        target_days = int(sys.argv[days_idx + 1])
                except (ValueError, IndexError):
                    print("Invalid --days parameter, using default 730 days")
            
            print(f"\n=== Building Full Historical Dataset ({target_days} days) ===")
            results = downloader.build_full_historical_dataset(target_days)
            
            print("\n=== Build Results ===")
            for timeframe, success in results.items():
                status = "SUCCESS" if success else "FAILED"
                print(f"{timeframe}: {status}")
                
        elif command == 'chunks':
            # Build using chunks for a specific timeframe
            if len(sys.argv) < 3:
                print("Usage: python data_downloader.py chunks [1h|3m] [--days N]")
                sys.exit(1)
                
            timeframe = sys.argv[2]
            target_days = 730
            
            if '--days' in sys.argv:
                try:
                    days_idx = sys.argv.index('--days')
                    if days_idx + 1 < len(sys.argv):
                        target_days = int(sys.argv[days_idx + 1])
                except (ValueError, IndexError):
                    print("Invalid --days parameter, using default 730 days")
            
            print(f"\n=== Chunked Download: {timeframe} ({target_days} days) ===")
            success = downloader.download_historical_chunks(timeframe, target_days)
            status = "SUCCESS" if success else "FAILED"
            print(f"Result: {status}")
        
        else:
            print("Usage: python data_downloader.py [download|status|check|build|chunks] [options]")
            print("Options:")
            print("  --force: Force download even if data exists")
            print("  --days N: Specify number of days for build/chunks command")
            print("")
            print("Examples:")
            print("  python data_downloader.py build --days 365")
            print("  python data_downloader.py chunks 3m --days 180")
    else:
        print("Usage: python data_downloader.py [download|status|check|build|chunks] [options]")
        print("Commands:")
        print("  download: Download standard dataset")
        print("  status: Show current data status")
        print("  check: Check what needs updating")
        print("  build: Build full historical dataset using chunks")
        print("  chunks [timeframe]: Download chunks for specific timeframe")