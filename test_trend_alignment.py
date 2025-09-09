#!/usr/bin/env python3
"""
Test trend alignment protection for directional strategies
"""

import requests
import json

def test_trend_alignment():
    """Test the trend alignment logic"""
    print("Testing Trend Alignment Protection")
    print("=" * 50)
    
    try:
        # Get current signal data
        response = requests.get('http://localhost:5000/api/signal_data')
        data = response.json()
        
        if data['success']:
            print(f"5-Minute Signal Score: {data['scoring']['total_score']}")
            print(f"Higher Timeframe Trend: {data['trend_alignment']['higher_timeframe_trend']}")
            print(f"Trend Strength: {data['trend_alignment']['trend_strength']}")
            print(f"Current Signal: {data['entry_type']} (Signal: {data['current_signal']})")
            print(f"Signal Strength: {data['signal_strength']}")
            print(f"Alignment Required: {data['trend_alignment']['alignment_required']}")
            
            print("\nThresholds:")
            print(f"Bullish Threshold: {data['trend_alignment']['bullish_threshold']}")
            print(f"Bearish Threshold: {data['trend_alignment']['bearish_threshold']}")
            
            print("\nTrend Protection Analysis:")
            
            # Analyze current conditions
            score = data['scoring']['total_score']
            trend = data['trend_alignment']['higher_timeframe_trend']
            bullish_thresh = data['trend_alignment']['bullish_threshold']
            bearish_thresh = data['trend_alignment']['bearish_threshold']
            
            if score >= bullish_thresh:
                print(f"[OK] 5m score ({score}) >= bullish threshold ({bullish_thresh}) - CALL signal possible")
                if trend == 'Bearish':
                    print("[BLOCKED] Cannot buy CALL when 1H trend is Bearish")
                elif trend == 'Bullish':
                    print("[ALLOWED] 1H trend aligns (Bullish)")
                elif trend == 'Neutral':
                    print("[ALLOWED] 1H trend is Neutral (weak)")
            elif score <= bearish_thresh:
                print(f"[OK] 5m score ({score}) <= bearish threshold ({bearish_thresh}) - PUT signal possible")
                if trend == 'Bullish':
                    print("[BLOCKED] Cannot buy PUT when 1H trend is Bullish")
                elif trend == 'Bearish':
                    print("[ALLOWED] 1H trend aligns (Bearish)")
                elif trend == 'Neutral':
                    print("[ALLOWED] 1H trend is Neutral (weak)")
            else:
                print(f"[NEUTRAL] 5m score ({score}) between thresholds - No directional signal")
            
            # Check if neutral strategy is active
            if data['neutral_conditions']['neutral_signal_active']:
                print(f"[ACTIVE] NEUTRAL STRATEGY: {data['entry_type']}")
                print(f"   Reason: Trend is {trend} and 5m score ({score}) < 9")
            
            print(f"\n[STATUS] Current: {data['entry_type']} with {data['confidence']*100:.0f}% confidence")
            
        else:
            print(f"[ERROR] API Error: {data['message']}")
            
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")

if __name__ == "__main__":
    test_trend_alignment()