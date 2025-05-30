#!/usr/bin/env python3
"""
IQOption Auto Trading Monitor
Continuously monitors market signals and executes $1 trades when high-confidence opportunities arise
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
        logging.FileHandler('/app/trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoTrader:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.assets = ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"]
        self.strategies = ["combined", "rsi", "macd", "bollinger", "trend"]
        self.confidence_threshold = 0.70  # 70% minimum confidence
        self.trade_amount = 1.0  # $1 per trade as requested
        self.session = None
        self.trades_executed = 0
        self.total_signals_checked = 0
        
    async def create_session(self):
        """Create aiohttp session"""
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    async def check_connection(self):
        """Check IQOption API connection status"""
        try:
            async with self.session.get(f"{self.base_url}/connection-status") as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get("connected"):
                        logger.info(f"✅ IQOption Connected - Balance: ${data.get('balance', 0):.2f}")
                        return True
                    else:
                        logger.warning("❌ IQOption Disconnected")
                        return False
                else:
                    logger.error(f"Connection check failed: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error checking connection: {e}")
            return False
    
    async def get_strategy_signal(self, asset, strategy):
        """Get trading signal for specific asset and strategy"""
        try:
            url = f"{self.base_url}/strategy-signal/{asset}?strategy={strategy}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Failed to get signal for {asset}/{strategy}: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting signal for {asset}/{strategy}: {e}")
            return None
    
    async def execute_trade(self, asset, strategy, signal_data):
        """Execute a trade"""
        try:
            trade_data = {
                "asset": asset,
                "amount": self.trade_amount,
                "strategy": strategy,
                "auto_trade": True
            }
            
            async with self.session.post(f"{self.base_url}/execute-trade", 
                                       json=trade_data) as response:
                result = await response.json()
                
                if result.get("success"):
                    self.trades_executed += 1
                    logger.info(f"🎯 TRADE EXECUTED #{self.trades_executed}")
                    logger.info(f"   Asset: {asset}")
                    logger.info(f"   Strategy: {strategy}")
                    logger.info(f"   Direction: {signal_data['signal'].upper()}")
                    logger.info(f"   Confidence: {signal_data['confidence']*100:.1f}%")
                    logger.info(f"   Amount: ${self.trade_amount}")
                    return True
                else:
                    logger.warning(f"Trade execution failed: {result.get('message')}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    async def scan_for_opportunities(self):
        """Scan all assets and strategies for high-confidence opportunities"""
        opportunities = []
        
        for asset in self.assets:
            for strategy in self.strategies:
                signal = await self.get_strategy_signal(asset, strategy)
                self.total_signals_checked += 1
                
                if signal and signal.get("confidence", 0) >= self.confidence_threshold:
                    if signal.get("signal") in ["call", "put"]:
                        opportunities.append({
                            "asset": asset,
                            "strategy": strategy,
                            "signal": signal
                        })
                        logger.info(f"🔍 HIGH-CONFIDENCE OPPORTUNITY FOUND!")
                        logger.info(f"   {asset} - {strategy}: {signal['signal'].upper()} "
                                  f"({signal['confidence']*100:.1f}% confidence)")
        
        return opportunities
    
    async def get_performance_stats(self):
        """Get current performance statistics"""
        try:
            async with self.session.get(f"{self.base_url}/performance") as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
        return {}
    
    async def monitor_and_trade(self):
        """Main monitoring loop"""
        logger.info("🚀 Starting Auto Trading Monitor...")
        logger.info(f"💰 Trade Amount: ${self.trade_amount}")
        logger.info(f"📊 Confidence Threshold: {self.confidence_threshold*100:.0f}%")
        logger.info(f"🎯 Assets: {', '.join(self.assets)}")
        logger.info(f"⚡ Strategies: {', '.join(self.strategies)}")
        logger.info("=" * 60)
        
        await self.create_session()
        
        scan_count = 0
        last_performance_check = 0
        
        try:
            while True:
                scan_count += 1
                current_time = time.time()
                
                # Check connection status
                if not await self.check_connection():
                    logger.warning("⏸️  Trading paused - Connection lost")
                    await asyncio.sleep(30)  # Wait longer if disconnected
                    continue
                
                # Scan for opportunities
                logger.info(f"🔍 Scanning #{scan_count} for trading opportunities...")
                opportunities = await self.scan_for_opportunities()
                
                # Execute trades for high-confidence opportunities
                if opportunities:
                    for opp in opportunities:
                        success = await self.execute_trade(
                            opp["asset"], 
                            opp["strategy"], 
                            opp["signal"]
                        )
                        if success:
                            # Brief pause between trades
                            await asyncio.sleep(2)
                else:
                    logger.info(f"💤 No high-confidence opportunities found this scan")
                
                # Show performance stats every 10 minutes
                if current_time - last_performance_check > 600:  # 10 minutes
                    performance = await self.get_performance_stats()
                    if performance:
                        logger.info("📈 PERFORMANCE UPDATE:")
                        logger.info(f"   Total Trades: {performance.get('total_trades', 0)}")
                        logger.info(f"   Win Rate: {performance.get('win_rate', 0):.1f}%")
                        logger.info(f"   Total Profit: ${performance.get('total_profit', 0):.2f}")
                        logger.info(f"   Signals Checked: {self.total_signals_checked}")
                    last_performance_check = current_time
                
                # Status update
                logger.info(f"✅ Scan #{scan_count} complete - Trades executed: {self.trades_executed}")
                logger.info("-" * 40)
                
                # Wait before next scan (30 seconds)
                await asyncio.sleep(30)
                
        except KeyboardInterrupt:
            logger.info("🛑 Trading monitor stopped by user")
        except Exception as e:
            logger.error(f"❌ Unexpected error in monitor: {e}")
        finally:
            await self.close_session()
            
            # Final performance report
            performance = await self.get_performance_stats()
            if performance:
                logger.info("🏁 FINAL TRADING SESSION SUMMARY:")
                logger.info(f"   Trades Executed: {self.trades_executed}")
                logger.info(f"   Total Signals Scanned: {self.total_signals_checked}")
                logger.info(f"   Win Rate: {performance.get('win_rate', 0):.1f}%")
                logger.info(f"   Total Profit: ${performance.get('total_profit', 0):.2f}")

async def main():
    trader = AutoTrader()
    await trader.monitor_and_trade()

if __name__ == "__main__":
    asyncio.run(main())