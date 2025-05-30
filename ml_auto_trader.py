#!/usr/bin/env python3
"""
ML-Enhanced Auto Trader
Uses machine learning models for high-accuracy trading signals
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
        logging.FileHandler('/app/ml_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MLEnhancedAutoTrader:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        # Focus on EUR/USD as it showed best performance in backtesting
        self.profitable_assets = ["EURUSD"]
        # Use ML-enhanced strategies
        self.ml_strategies = ["adaptive_ml", "ml_enhanced"]
        self.confidence_threshold = 0.70  # Lowered from 0.80 to get more trades
        self.trade_amount = 1.0
        self.session = None
        self.trades_executed = 0
        self.signals_checked = 0
        self.ml_predictions = []
        
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
    
    async def get_ml_signal(self, asset, strategy):
        """Get ML-enhanced signal"""
        try:
            url = f"{self.base_url}/strategy-signal/{asset}?strategy={strategy}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.json()
        except Exception as e:
            logger.error(f"Error getting ML signal for {asset}/{strategy}: {e}")
        return None
    
    async def execute_ml_trade(self, asset, strategy, signal_data):
        """Execute ML-enhanced trade"""
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
                    logger.info(f"🤖 ML TRADE #{self.trades_executed} EXECUTED!")
                    logger.info(f"   Asset: {asset}")
                    logger.info(f"   ML Strategy: {strategy}")
                    logger.info(f"   Direction: {signal_data['signal'].upper()}")
                    logger.info(f"   Confidence: {signal_data['confidence']*100:.1f}%")
                    logger.info(f"   ML Probability: {signal_data.get('ml_probability', 0)*100:.1f}%")
                    
                    # Log feature importance if available
                    if signal_data.get('feature_importance'):
                        logger.info(f"   Key Factors: {', '.join([f['feature'] for f in signal_data['feature_importance'][:3]])}")
                    
                    return True
                else:
                    logger.warning(f"ML Trade failed: {result.get('message')}")
                    return False
        except Exception as e:
            logger.error(f"ML Trade execution error: {e}")
            return False
    
    async def scan_ml_opportunities(self):
        """Scan for ML-enhanced opportunities"""
        opportunities = []
        
        for asset in self.profitable_assets:
            for strategy in self.ml_strategies:
                signal = await self.get_ml_signal(asset, strategy)
                self.signals_checked += 1
                
                if signal and signal.get("confidence", 0) >= self.confidence_threshold:
                    if signal.get("signal") in ["call", "put"]:
                        opportunities.append({
                            "asset": asset,
                            "strategy": strategy,
                            "signal": signal
                        })
                        
                        # Log detailed ML analysis
                        logger.info(f"🧠 ML OPPORTUNITY DETECTED!")
                        logger.info(f"   Asset: {asset}")
                        logger.info(f"   Strategy: {strategy}")
                        logger.info(f"   Signal: {signal['signal'].upper()}")
                        logger.info(f"   Confidence: {signal['confidence']*100:.1f}%")
                        logger.info(f"   ML Probability: {signal.get('ml_probability', 0)*100:.1f}%")
                        logger.info(f"   Strategy Type: {signal.get('strategy_name', 'Unknown')}")
                        
                        # Store ML prediction for analysis
                        self.ml_predictions.append({
                            'timestamp': datetime.utcnow(),
                            'asset': asset,
                            'strategy': strategy,
                            'signal': signal['signal'],
                            'confidence': signal['confidence'],
                            'ml_probability': signal.get('ml_probability', 0)
                        })
        
        return opportunities
    
    def analyze_ml_performance(self):
        """Analyze ML prediction patterns"""
        if len(self.ml_predictions) < 5:
            return
        
        recent_predictions = self.ml_predictions[-10:]
        
        call_signals = [p for p in recent_predictions if p['signal'] == 'call']
        put_signals = [p for p in recent_predictions if p['signal'] == 'put']
        
        avg_confidence = sum(p['confidence'] for p in recent_predictions) / len(recent_predictions)
        avg_ml_prob = sum(p['ml_probability'] for p in recent_predictions) / len(recent_predictions)
        
        logger.info(f"📊 ML PERFORMANCE ANALYSIS:")
        logger.info(f"   Recent Signals: {len(recent_predictions)}")
        logger.info(f"   CALL signals: {len(call_signals)}")
        logger.info(f"   PUT signals: {len(put_signals)}")
        logger.info(f"   Average Confidence: {avg_confidence*100:.1f}%")
        logger.info(f"   Average ML Probability: {avg_ml_prob*100:.1f}%")
    
    async def run_ml_trading(self):
        """Main ML-enhanced trading loop"""
        logger.info("🚀 STARTING ML-ENHANCED AUTO TRADER")
        logger.info("🤖 Using advanced machine learning models for signal generation")
        logger.info(f"💰 Trade Amount: ${self.trade_amount}")
        logger.info(f"📊 ML Confidence Threshold: {self.confidence_threshold*100:.0f}%")
        logger.info(f"🎯 Profitable Assets: {', '.join(self.profitable_assets)}")
        logger.info(f"🧠 ML Strategies: {', '.join(self.ml_strategies)}")
        logger.info("=" * 70)
        
        await self.create_session()
        
        scan_count = 0
        last_analysis = time.time()
        
        try:
            while True:
                scan_count += 1
                current_time = time.time()
                
                if not await self.check_connection():
                    logger.warning("⏸️  ML Trading paused - Connection lost")
                    await asyncio.sleep(30)
                    continue
                
                logger.info(f"🧠 ML Scan #{scan_count}")
                opportunities = await self.scan_ml_opportunities()
                
                if opportunities:
                    for opp in opportunities:
                        success = await self.execute_ml_trade(
                            opp["asset"], 
                            opp["strategy"], 
                            opp["signal"]
                        )
                        if success:
                            await asyncio.sleep(3)  # Brief pause between trades
                else:
                    logger.info(f"🤖 No high-confidence ML opportunities found")
                
                # Analyze ML performance every 10 minutes
                if current_time - last_analysis > 600:
                    self.analyze_ml_performance()
                    last_analysis = current_time
                
                logger.info(f"✅ ML Scan #{scan_count} complete - Trades: {self.trades_executed}")
                logger.info("-" * 50)
                
                await asyncio.sleep(30)  # Scan every 30 seconds
                
        except KeyboardInterrupt:
            logger.info("🛑 ML Trader stopped")
        except Exception as e:
            logger.error(f"❌ ML Trading error: {e}")
        finally:
            await self.close_session()
            
            # Final ML analysis
            if self.ml_predictions:
                logger.info("🏁 FINAL ML TRADING SESSION SUMMARY:")
                logger.info(f"   ML Trades Executed: {self.trades_executed}")
                logger.info(f"   ML Signals Analyzed: {self.signals_checked}")
                logger.info(f"   ML Predictions Generated: {len(self.ml_predictions)}")
                
                if self.trades_executed > 0:
                    execution_rate = (self.trades_executed / len(self.ml_predictions)) * 100
                    logger.info(f"   ML Signal Execution Rate: {execution_rate:.1f}%")

async def main():
    trader = MLEnhancedAutoTrader()
    await trader.run_ml_trading()

if __name__ == "__main__":
    asyncio.run(main())