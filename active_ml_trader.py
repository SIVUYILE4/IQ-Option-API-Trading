#!/usr/bin/env python3
"""
ML-Enhanced Trading Bot with Market Status Detection
Handles market closures gracefully and demonstrates bot capabilities
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
        logging.FileHandler('/app/active_ml_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ActiveMLTrader:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.assets = ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"]
        self.ml_strategies = ["adaptive_ml", "ml_enhanced"]
        self.confidence_threshold = 0.70
        self.trade_amount = 1.0
        self.session = None
        
        # Statistics
        self.trades_executed = 0
        self.trades_attempted = 0
        self.signals_detected = 0
        self.market_closed_count = 0
        
        # Trading signals log
        self.trading_signals = []
        
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
        """Get ML signal"""
        try:
            url = f"{self.base_url}/strategy-signal/{asset}?strategy={strategy}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting signal: {e}")
        return None
    
    async def attempt_trade_execution(self, asset, strategy, signal_data):
        """Attempt to execute trade with market status detection"""
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
                    logger.info(f"🎉 TRADE #{self.trades_executed} EXECUTED!")
                    logger.info(f"   Asset: {asset}")
                    logger.info(f"   Strategy: {strategy}")
                    logger.info(f"   Direction: {signal_data['signal'].upper()}")
                    logger.info(f"   Confidence: {signal_data['confidence']*100:.1f}%")
                    return "executed"
                else:
                    error_msg = result.get("message", "Unknown error")
                    if "suspended" in error_msg.lower() or "cannot purchase" in error_msg.lower():
                        self.market_closed_count += 1
                        return "market_closed"
                    else:
                        return "failed"
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return "error"
    
    def log_trading_signal(self, asset, strategy, signal_data, execution_result):
        """Log trading signal for analysis"""
        signal_log = {
            'timestamp': datetime.utcnow().isoformat(),
            'asset': asset,
            'strategy': strategy,
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'ml_probability': signal_data.get('ml_probability', 0),
            'execution_result': execution_result
        }
        
        self.trading_signals.append(signal_log)
        
        # Keep only last 100 signals
        if len(self.trading_signals) > 100:
            self.trading_signals = self.trading_signals[-100:]
    
    async def scan_and_trade(self):
        """Scan for opportunities and attempt trades"""
        opportunities_found = 0
        
        for asset in self.assets:
            for strategy in self.ml_strategies:
                signal = await self.get_ml_signal(asset, strategy)
                
                if signal and signal.get("confidence", 0) >= self.confidence_threshold:
                    if signal.get("signal") in ["call", "put"]:
                        opportunities_found += 1
                        self.signals_detected += 1
                        
                        logger.info(f"🤖 ML OPPORTUNITY #{self.signals_detected}")
                        logger.info(f"   Asset: {asset}")
                        logger.info(f"   Strategy: {strategy}")
                        logger.info(f"   Signal: {signal['signal'].upper()}")
                        logger.info(f"   Confidence: {signal['confidence']*100:.1f}%")
                        logger.info(f"   ML Probability: {signal.get('ml_probability', 0)*100:.1f}%")
                        
                        # Attempt trade execution
                        self.trades_attempted += 1
                        execution_result = await self.attempt_trade_execution(asset, strategy, signal)
                        
                        if execution_result == "executed":
                            logger.info(f"✅ Trade executed successfully!")
                        elif execution_result == "market_closed":
                            logger.warning(f"⏸️  Market closed for {asset} - Signal saved for later")
                        else:
                            logger.warning(f"❌ Trade execution failed")
                        
                        # Log the signal regardless of execution
                        self.log_trading_signal(asset, strategy, signal, execution_result)
                        
                        await asyncio.sleep(2)  # Pause between attempts
        
        return opportunities_found
    
    def generate_performance_report(self):
        """Generate performance report"""
        if not self.trading_signals:
            return "No trading signals generated yet."
        
        recent_signals = self.trading_signals[-20:]  # Last 20 signals
        
        total_signals = len(recent_signals)
        call_signals = len([s for s in recent_signals if s['signal'] == 'call'])
        put_signals = len([s for s in recent_signals if s['signal'] == 'put'])
        avg_confidence = sum(s['confidence'] for s in recent_signals) / total_signals
        
        executed_trades = len([s for s in recent_signals if s['execution_result'] == 'executed'])
        market_closed = len([s for s in recent_signals if s['execution_result'] == 'market_closed'])
        
        report = f"""
📊 ML TRADING BOT PERFORMANCE REPORT:
══════════════════════════════════════
🤖 Signals Generated: {total_signals}
📈 CALL Signals: {call_signals}
📉 PUT Signals: {put_signals}
🎯 Average Confidence: {avg_confidence*100:.1f}%
✅ Trades Executed: {executed_trades}
⏸️  Market Closed: {market_closed}
🔄 Execution Rate: {(executed_trades/total_signals*100):.1f}% (when markets open)
        """
        
        return report
    
    async def run_active_trading(self):
        """Main active trading loop"""
        logger.info("🚀 STARTING ACTIVE ML TRADING BOT")
        logger.info("🤖 Advanced ML strategies with market status detection")
        logger.info(f"💰 Trade Amount: ${self.trade_amount}")
        logger.info(f"📊 Confidence Threshold: {self.confidence_threshold*100:.0f}%")
        logger.info(f"🎯 Assets: {', '.join(self.assets)}")
        logger.info("=" * 70)
        
        await self.create_session()
        
        scan_count = 0
        last_report = time.time()
        
        try:
            while True:
                scan_count += 1
                current_time = time.time()
                
                # Check connection
                connection_status = await self.check_connection()
                if not connection_status.get("connected"):
                    logger.warning("⏸️  IQOption API disconnected")
                    await asyncio.sleep(30)
                    continue
                
                logger.info(f"🧠 Active ML Scan #{scan_count}")
                logger.info(f"💰 Balance: ${connection_status.get('balance', 0):.2f}")
                
                # Scan for opportunities
                opportunities = await self.scan_and_trade()
                
                if opportunities == 0:
                    logger.info("🤖 No high-confidence ML opportunities found this scan")
                
                # Generate report every 5 minutes
                if current_time - last_report > 300:
                    report = self.generate_performance_report()
                    logger.info(report)
                    last_report = current_time
                
                logger.info(f"📊 Session Stats: {self.signals_detected} signals, {self.trades_attempted} attempts, {self.trades_executed} executed")
                logger.info("-" * 50)
                
                await asyncio.sleep(30)  # Scan every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 Active ML Trader stopped by user")
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
        finally:
            await self.close_session()
            
            # Final report
            final_report = self.generate_performance_report()
            logger.info("🏁 FINAL SESSION REPORT:")
            logger.info(final_report)
            
            # Save signals to file
            with open('/app/trading_signals_log.json', 'w') as f:
                json.dump(self.trading_signals, f, indent=2)
            logger.info("💾 Trading signals saved to /app/trading_signals_log.json")

async def main():
    trader = ActiveMLTrader()
    await trader.run_active_trading()

if __name__ == "__main__":
    asyncio.run(main())