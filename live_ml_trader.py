#!/usr/bin/env python3
"""
LIVE ML Auto Trader - FIXED VERSION
Now using working OTC assets for real trade execution
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/live_ml_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LiveMLTrader:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        # Use assets that work (OTC versions)
        self.working_assets = ["EURUSD", "GBPUSD", "USDJPY"]
        self.ml_strategies = ["adaptive_ml", "ml_enhanced"]
        self.confidence_threshold = 0.70
        self.trade_amount = 5.0  # Use successful amount
        self.session = None
        
        # Live trading stats
        self.trades_executed = 0
        self.successful_orders = []
        self.signals_detected = 0
        
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
                    return data
        except Exception as e:
            logger.error(f"Connection error: {e}")
        return {"connected": False, "balance": 0}
    
    async def get_ml_signal(self, asset, strategy):
        try:
            url = f"{self.base_url}/strategy-signal/{asset}?strategy={strategy}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
        return None
    
    async def execute_live_trade(self, asset, strategy, signal_data):
        """Execute live trade with working API"""
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
                    order_id = result.get("order_id")
                    trade_info = result.get("trade", {})
                    
                    self.successful_orders.append({
                        'order_id': order_id,
                        'timestamp': datetime.utcnow(),
                        'asset': asset,
                        'strategy': strategy,
                        'signal': signal_data['signal'],
                        'confidence': signal_data['confidence'],
                        'amount': self.trade_amount
                    })
                    
                    logger.info(f"🎉 LIVE TRADE #{self.trades_executed} EXECUTED!")
                    logger.info(f"   💰 Amount: \${self.trade_amount}")
                    logger.info(f"   🎯 Asset: {asset} (using OTC)")
                    logger.info(f"   🤖 Strategy: {strategy}")
                    logger.info(f"   📊 Signal: {signal_data['signal'].upper()}")
                    logger.info(f"   🎲 Confidence: {signal_data['confidence']*100:.1f}%")
                    logger.info(f"   📝 Order ID: {order_id}")
                    logger.info(f"   🆔 Trade ID: {trade_info.get('id', 'Unknown')}")
                    
                    return True
                else:
                    logger.warning(f"Trade failed: {result.get('message')}")
                    return False
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    
    async def scan_and_execute(self):
        """Scan for ML opportunities and execute live trades"""
        opportunities = 0
        
        for asset in self.working_assets:
            for strategy in self.ml_strategies:
                signal = await self.get_ml_signal(asset, strategy)
                
                if signal and signal.get("confidence", 0) >= self.confidence_threshold:
                    if signal.get("signal") in ["call", "put"]:
                        opportunities += 1
                        self.signals_detected += 1
                        
                        logger.info(f"🧠 ML OPPORTUNITY #{self.signals_detected}")
                        logger.info(f"   Asset: {asset}")
                        logger.info(f"   Strategy: {strategy}")
                        logger.info(f"   Signal: {signal['signal'].upper()}")
                        logger.info(f"   Confidence: {signal['confidence']*100:.1f}%")
                        logger.info(f"   ML Probability: {signal.get('ml_probability', 0)*100:.1f}%")
                        
                        # Execute live trade
                        success = await self.execute_live_trade(asset, strategy, signal)
                        
                        if success:
                            # Wait 60 seconds between trades for safety
                            logger.info(f"   ⏱️  Waiting 60 seconds before next trade...")
                            await asyncio.sleep(60)
                        else:
                            await asyncio.sleep(5)
        
        return opportunities
    
    async def run_live_trading(self):
        """Main live trading loop"""
        logger.info("🚀 STARTING LIVE ML AUTO TRADER")
        logger.info("✅ API FIXED - Now using working OTC assets")
        logger.info("🤖 Real trades with real money (practice account)")
        logger.info(f"💰 Trade Amount: \${self.trade_amount}")
        logger.info(f"📊 Confidence Threshold: {self.confidence_threshold*100:.0f}%")
        logger.info(f"🎯 Working Assets: {', '.join(self.working_assets)}")
        logger.info("=" * 70)
        
        await self.create_session()
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                
                # Check connection and balance
                connection = await self.check_connection()
                if not connection.get("connected"):
                    logger.warning("⏸️  IQOption API disconnected")
                    await asyncio.sleep(30)
                    continue
                
                current_balance = connection.get("balance", 0)
                logger.info(f"🧠 Live ML Scan #{scan_count}")
                logger.info(f"💰 Current Balance: \${current_balance:.2f}")
                
                # Scan and execute trades
                opportunities = await self.scan_and_execute()
                
                if opportunities == 0:
                    logger.info("🤖 No high-confidence ML opportunities this scan")
                
                # Show session stats
                total_invested = self.trades_executed * self.trade_amount
                logger.info(f"📊 Session: {self.signals_detected} signals, {self.trades_executed} trades, \${total_invested:.2f} invested")
                
                if self.successful_orders:
                    logger.info(f"📝 Recent Orders: {[o['order_id'] for o in self.successful_orders[-3:]]}")
                
                logger.info("-" * 50)
                
                await asyncio.sleep(30)  # Scan every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Live ML Trader stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
        finally:
            await self.close_session()
            
            # Final summary
            logger.info("🏁 LIVE TRADING SESSION COMPLETE!")
            logger.info(f"   🤖 ML Signals Generated: {self.signals_detected}")
            logger.info(f"   💰 Live Trades Executed: {self.trades_executed}")
            logger.info(f"   📈 Total Invested: \${self.trades_executed * self.trade_amount:.2f}")
            logger.info(f"   📝 Order IDs: {[o['order_id'] for o in self.successful_orders]}")

async def main():
    trader = LiveMLTrader()
    await trader.run_live_trading()

if __name__ == "__main__":
    asyncio.run(main())