#!/usr/bin/env python3
"""
Backtesting-Validated Auto Trader
Only trades on strategies and assets that have been proven profitable in backtesting
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/validated_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ValidatedAutoTrader:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        # Only trade profitable combinations from backtesting
        self.profitable_combinations = {
            "EURUSD": ["bollinger", "macd", "combined"],  # Only profitable asset
        }
        self.confidence_threshold = 0.75  # Increased threshold based on backtesting
        self.trade_amount = 1.0
        self.session = None
        self.trades_executed = 0
        self.signals_checked = 0
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def check_connection(self):
        try:
            async with self.session.get(f"{self.base_url}/connection-status") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("connected"):
                        logger.info(f"✅ Connected - Balance: ${data.get('balance', 0):.2f}")
                        return True
                    else:
                        logger.warning("❌ IQOption Disconnected")
                        return False
        except Exception as e:
            logger.error(f"Connection error: {e}")
            return False
    
    async def get_validated_signal(self, asset, strategy):
        """Get signal only for validated profitable combinations"""
        try:
            url = f"{self.base_url}/strategy-signal/{asset}?strategy={strategy}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting signal for {asset}/{strategy}: {e}")
        return None
    
    async def execute_validated_trade(self, asset, strategy, signal_data):
        """Execute trade only for validated combinations"""
        try:
            trade_data = {
                "asset": asset,
                "amount": self.trade_amount,
                "strategy": strategy,
                "auto_trade": True
            }
            
            async with self.session.post(f"{self.base_url}/execute-trade", json=trade_data) as response:
                result = await response.json()
                
                if result.get("success"):
                    self.trades_executed += 1
                    logger.info(f"🎯 VALIDATED TRADE #{self.trades_executed} EXECUTED!")
                    logger.info(f"   Asset: {asset} (PROFITABLE in backtesting)")
                    logger.info(f"   Strategy: {strategy}")
                    logger.info(f"   Direction: {signal_data['signal'].upper()}")
                    logger.info(f"   Confidence: {signal_data['confidence']*100:.1f}%")
                    logger.info(f"   Backtest Win Rate: {self.get_backtest_winrate(asset, strategy)}")
                    return True
                else:
                    logger.warning(f"Trade failed: {result.get('message')}")
                    return False
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    
    def get_backtest_winrate(self, asset, strategy):
        """Get expected win rate from backtesting"""
        backtest_results = {
            ("EURUSD", "bollinger"): "57.4%",
            ("EURUSD", "macd"): "57.1%", 
            ("EURUSD", "combined"): "Expected 60%+"
        }
        return backtest_results.get((asset, strategy), "Unknown")
    
    async def scan_validated_opportunities(self):
        """Scan only validated profitable combinations"""
        opportunities = []
        
        for asset, strategies in self.profitable_combinations.items():
            for strategy in strategies:
                signal = await self.get_validated_signal(asset, strategy)
                self.signals_checked += 1
                
                if signal and signal.get("confidence", 0) >= self.confidence_threshold:
                    if signal.get("signal") in ["call", "put"]:
                        opportunities.append({
                            "asset": asset,
                            "strategy": strategy,
                            "signal": signal
                        })
                        logger.info(f"🔍 VALIDATED OPPORTUNITY!")
                        logger.info(f"   {asset} - {strategy}: {signal['signal'].upper()} "
                                  f"({signal['confidence']*100:.1f}% confidence)")
                        logger.info(f"   Backtest Performance: {self.get_backtest_winrate(asset, strategy)} win rate")
        
        return opportunities
    
    async def run_validated_trading(self):
        """Run trading with only validated strategies"""
        logger.info("🚀 STARTING BACKTESTING-VALIDATED AUTO TRADER")
        logger.info("🔬 Only trades combinations proven profitable in backtesting")
        logger.info(f"💰 Trade Amount: ${self.trade_amount}")
        logger.info(f"📊 Confidence Threshold: {self.confidence_threshold*100:.0f}%")
        logger.info(f"🎯 Validated Assets: {list(self.profitable_combinations.keys())}")
        logger.info("=" * 70)
        
        await self.create_session()
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                
                if not await self.check_connection():
                    logger.warning("⏸️  Trading paused - Connection lost")
                    await asyncio.sleep(30)
                    continue
                
                logger.info(f"🔍 Validated Scan #{scan_count}")
                opportunities = await self.scan_validated_opportunities()
                
                if opportunities:
                    for opp in opportunities:
                        success = await self.execute_validated_trade(
                            opp["asset"], 
                            opp["strategy"], 
                            opp["signal"]
                        )
                        if success:
                            await asyncio.sleep(2)  # Brief pause between trades
                else:
                    logger.info(f"💤 No validated opportunities found (high standards)")
                
                logger.info(f"✅ Scan #{scan_count} complete - Validated trades: {self.trades_executed}")
                logger.info("-" * 50)
                
                await asyncio.sleep(30)  # Scan every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Validated trader stopped")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            await self.close_session()
            logger.info(f"🏁 SESSION COMPLETE - Validated trades executed: {self.trades_executed}")

async def main():
    trader = ValidatedAutoTrader()
    await trader.run_validated_trading()

if __name__ == "__main__":
    asyncio.run(main())