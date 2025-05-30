#!/usr/bin/env python3
"""
Quick validation of optimized strategies
"""
import asyncio
import aiohttp
import json
import sys

async def test_optimized_strategy():
    base_url = "http://localhost:8001/api"
    assets = ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"]
    
    async with aiohttp.ClientSession() as session:
        print("🔬 TESTING OPTIMIZED STRATEGIES")
        print("=" * 50)
        
        for asset in assets:
            print(f"\n📊 {asset}:")
            
            # Test optimized strategy
            async with session.get(f"{base_url}/strategy-signal/{asset}?strategy=optimized") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  Optimized: {data['signal']:>4} | {data['confidence']*100:>5.1f}% | {data['strategy_name']}")
                
            # Test combined strategy
            async with session.get(f"{base_url}/strategy-signal/{asset}?strategy=combined") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  Combined:  {data['signal']:>4} | {data['confidence']*100:>5.1f}% | {data['strategy_name']}")
                    
            # Test best individual strategy for this asset
            best_strategies = {
                "EURUSD": "bollinger",
                "GBPUSD": "macd", 
                "USDJPY": "bollinger",
                "AUDCAD": "rsi"
            }
            
            best_strategy = best_strategies.get(asset, "bollinger")
            async with session.get(f"{base_url}/strategy-signal/{asset}?strategy={best_strategy}") as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"  Best ({best_strategy:>9}): {data['signal']:>4} | {data['confidence']*100:>5.1f}% | {data['strategy_name']}")

if __name__ == "__main__":
    asyncio.run(test_optimized_strategy())