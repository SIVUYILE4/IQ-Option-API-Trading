#!/usr/bin/env python3
"""
Manual Trade Trigger - Test ML signals immediately
"""

import asyncio
import aiohttp
import json

async def test_ml_signal_and_force_trade():
    print("🎯 MANUAL ML TRADE TEST")
    print("=" * 40)
    
    base_url = "http://localhost:8001/api"
    
    async with aiohttp.ClientSession() as session:
        # 1. Check connection
        print("1. Checking connection...")
        async with session.get(f"{base_url}/connection-status") as response:
            if response.status == 200:
                data = await response.json()
                print(f"   ✅ Connected: ${data.get('balance', 0):.2f} balance")
            else:
                print(f"   ❌ Connection failed: {response.status}")
                return
        
        # 2. Get current ML signal
        print("\n2. Getting current ML signal...")
        async with session.get(f"{base_url}/strategy-signal/EURUSD?strategy=adaptive_ml") as response:
            if response.status == 200:
                signal = await response.json()
                print(f"   🤖 Signal: {signal['signal'].upper()}")
                print(f"   🎯 Confidence: {signal['confidence']*100:.1f}%")
                print(f"   🧠 ML Probability: {signal.get('ml_probability', 0)*100:.1f}%")
                print(f"   ⚡ Strategy: {signal.get('strategy_name', 'Unknown')}")
                
                if signal['signal'] != 'hold' and signal['confidence'] > 0.65:
                    print(f"   ✅ HIGH-CONFIDENCE SIGNAL DETECTED!")
                    
                    # 3. Force trade execution test
                    print("\n3. Testing trade execution...")
                    trade_data = {
                        "asset": "EURUSD",
                        "amount": 5.0,
                        "strategy": "adaptive_ml", 
                        "auto_trade": True
                    }
                    
                    async with session.post(f"{base_url}/execute-trade", json=trade_data) as trade_response:
                        result = await trade_response.json()
                        print(f"   Trade result: {result}")
                        
                        if result.get("success"):
                            print("   🎉 TRADE EXECUTED SUCCESSFULLY!")
                        else:
                            print(f"   ❌ Trade failed: {result.get('message')}")
                            
                            # Try with different parameters
                            print("\n4. Trying alternative parameters...")
                            for amount in [1, 2, 10]:
                                trade_data_alt = {
                                    "asset": "EURUSD",
                                    "amount": float(amount),
                                    "strategy": "adaptive_ml",
                                    "auto_trade": True
                                }
                                async with session.post(f"{base_url}/execute-trade", json=trade_data_alt) as alt_response:
                                    alt_result = await alt_response.json()
                                    print(f"   Amount ${amount}: {alt_result.get('success', False)} - {alt_result.get('message', 'No message')}")
                                    if alt_result.get("success"):
                                        print(f"   🎉 SUCCESS WITH ${amount}!")
                                        break
                else:
                    print(f"   ⏸️  Low confidence signal: {signal['confidence']*100:.1f}%")
            else:
                print(f"   ❌ Signal fetch failed: {response.status}")

if __name__ == "__main__":
    asyncio.run(test_ml_signal_and_force_trade())