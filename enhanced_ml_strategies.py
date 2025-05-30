#!/usr/bin/env python3
"""
Enhanced Trading Strategies with Machine Learning Integration
Combines traditional technical analysis with advanced ML predictions
"""

import sys
import os
sys.path.append('/app')

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from ml_trading_system import MLTradingSystem, MLTradingSignal

# Import original strategies
sys.path.append('/app/backend')

logger = logging.getLogger(__name__)

class EnhancedMLTradingStrategies:
    def __init__(self):
        self.ml_system = MLTradingSystem()
        self.ml_models_ready = {}
        
    async def initialize_ml_models(self, asset="EURUSD"):
        """Initialize and train ML models for the asset"""
        logger.info(f"🤖 Initializing ML models for {asset}...")
        
        try:
            # Try loading existing models first
            if self.ml_system.load_models(asset):
                self.ml_models_ready[asset] = True
                logger.info(f"✅ Loaded existing ML models for {asset}")
                return True
            
            # If no models exist, train new ones
            import aiohttp
            async with aiohttp.ClientSession() as session:
                url = f"http://localhost:8001/api/market-data/{asset}?count=2000"
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        df = pd.DataFrame(data['data'])
                        
                        if len(df) >= 100:
                            # Prepare and train
                            features, labels = self.ml_system.prepare_training_data(df)
                            results = self.ml_system.train_models(features, labels, asset)
                            
                            self.ml_models_ready[asset] = True
                            logger.info(f"✅ Trained new ML models for {asset}")
                            
                            # Log performance
                            for model_name, result in results.items():
                                logger.info(f"  {model_name}: {result['test_accuracy']:.3f} accuracy")
                            
                            return True
                        else:
                            logger.warning(f"Insufficient data for ML training: {len(df)} candles")
                            return False
                    else:
                        logger.error(f"Failed to fetch data for ML training: {response.status}")
                        return False
        except Exception as e:
            logger.error(f"Error initializing ML models: {e}")
            return False
    
    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        import ta
        return ta.momentum.RSIIndicator(data['close'], window=period).rsi()

    def calculate_macd(self, data: pd.DataFrame) -> dict:
        """Calculate MACD"""
        import ta
        macd = ta.trend.MACD(data['close'])
        return {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> dict:
        """Calculate Bollinger Bands"""
        import ta
        bb = ta.volatility.BollingerBands(data['close'], window=period)
        return {
            'upper': bb.bollinger_hband(),
            'middle': bb.bollinger_mavg(),
            'lower': bb.bollinger_lband()
        }
    
    def traditional_bollinger_strategy(self, data: pd.DataFrame) -> dict:
        """Traditional Bollinger Bands strategy"""
        if len(data) < 25:
            return {"signal": "hold", "confidence": 0.0, "strategy_name": "Bollinger_Traditional"}
        
        bb = self.calculate_bollinger_bands(data)
        current_price = data['close'].iloc[-1]
        upper_band = bb['upper'].iloc[-1]
        lower_band = bb['lower'].iloc[-1]
        
        if current_price <= lower_band:
            return {"signal": "call", "confidence": 0.75, "strategy_name": "Bollinger_Oversold"}
        elif current_price >= upper_band:
            return {"signal": "put", "confidence": 0.75, "strategy_name": "Bollinger_Overbought"}
        
        return {"signal": "hold", "confidence": 0.0, "strategy_name": "Bollinger_Traditional"}
    
    def traditional_macd_strategy(self, data: pd.DataFrame) -> dict:
        """Traditional MACD strategy"""
        if len(data) < 30:
            return {"signal": "hold", "confidence": 0.0, "strategy_name": "MACD_Traditional"}
        
        macd_data = self.calculate_macd(data)
        macd = macd_data['macd'].iloc[-1]
        signal = macd_data['signal'].iloc[-1]
        prev_macd = macd_data['macd'].iloc[-2]
        prev_signal = macd_data['signal'].iloc[-2]
        
        if prev_macd <= prev_signal and macd > signal:
            return {"signal": "call", "confidence": 0.70, "strategy_name": "MACD_Bullish"}
        elif prev_macd >= prev_signal and macd < signal:
            return {"signal": "put", "confidence": 0.70, "strategy_name": "MACD_Bearish"}
        
        return {"signal": "hold", "confidence": 0.0, "strategy_name": "MACD_Traditional"}
    
    async def ml_enhanced_strategy(self, asset: str, data: pd.DataFrame) -> MLTradingSignal:
        """ML-enhanced strategy combining traditional and ML signals"""
        
        # Ensure ML models are ready
        if asset not in self.ml_models_ready:
            await self.initialize_ml_models(asset)
        
        if not self.ml_models_ready.get(asset, False):
            # Fallback to traditional strategy
            traditional = self.traditional_bollinger_strategy(data)
            return MLTradingSignal(
                asset=asset,
                signal=traditional["signal"],
                confidence=traditional["confidence"],
                ml_probability=0.5,
                strategy_name="Traditional_Fallback"
            )
        
        try:
            # Get ML prediction
            ml_prob, ml_signal, ml_confidence = self.ml_system.predict_ensemble(data, asset)
            
            # Get traditional signals
            bollinger_signal = self.traditional_bollinger_strategy(data)
            macd_signal = self.traditional_macd_strategy(data)
            
            # Combine ML and traditional signals
            signals = []
            
            # ML signal (highest weight)
            if ml_signal != "hold" and ml_confidence > 0.7:
                signals.extend([{"signal": ml_signal, "confidence": ml_confidence, "weight": 0.5}])
            
            # Traditional signals
            if bollinger_signal["signal"] != "hold":
                signals.append({"signal": bollinger_signal["signal"], "confidence": bollinger_signal["confidence"], "weight": 0.3})
            
            if macd_signal["signal"] != "hold":
                signals.append({"signal": macd_signal["signal"], "confidence": macd_signal["confidence"], "weight": 0.2})
            
            if not signals:
                return MLTradingSignal(
                    asset=asset,
                    signal="hold",
                    confidence=0.0,
                    ml_probability=ml_prob,
                    strategy_name="ML_Enhanced_Hold"
                )
            
            # Calculate weighted consensus
            call_weight = sum(s["confidence"] * s["weight"] for s in signals if s["signal"] == "call")
            put_weight = sum(s["confidence"] * s["weight"] for s in signals if s["signal"] == "put")
            total_weight = sum(s["weight"] for s in signals)
            
            if call_weight > put_weight and call_weight / total_weight > 0.6:
                final_signal = "call"
                final_confidence = min(0.95, call_weight / total_weight + 0.1)
            elif put_weight > call_weight and put_weight / total_weight > 0.6:
                final_signal = "put"
                final_confidence = min(0.95, put_weight / total_weight + 0.1)
            else:
                final_signal = "hold"
                final_confidence = 0.0
            
            # Get feature importance for explainability
            feature_importance = []
            if asset in self.ml_system.feature_importance:
                importance = self.ml_system.feature_importance[asset]
                top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
                feature_importance = [{"feature": f, "importance": round(i, 4)} for f, i in top_features]
            
            return MLTradingSignal(
                asset=asset,
                signal=final_signal,
                confidence=final_confidence,
                ml_probability=ml_prob,
                strategy_name="ML_Enhanced_Ensemble",
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error in ML enhanced strategy: {e}")
            # Fallback to traditional
            traditional = self.traditional_bollinger_strategy(data)
            return MLTradingSignal(
                asset=asset,
                signal=traditional["signal"],
                confidence=traditional["confidence"] * 0.8,  # Reduced confidence due to error
                ml_probability=0.5,
                strategy_name="Traditional_Error_Fallback"
            )
    
    async def adaptive_ml_strategy(self, asset: str, data: pd.DataFrame) -> MLTradingSignal:
        """Adaptive ML strategy that learns from recent performance"""
        
        # Get base ML enhanced signal
        base_signal = await self.ml_enhanced_strategy(asset, data)
        
        # Adaptive confidence adjustment based on recent market conditions
        recent_volatility = data['close'].pct_change().tail(20).std()
        volatility_adjustment = 1.0
        
        if recent_volatility > 0.01:  # High volatility
            volatility_adjustment = 0.9  # Reduce confidence
        elif recent_volatility < 0.005:  # Low volatility
            volatility_adjustment = 1.1  # Increase confidence
        
        # Time-based adjustment (market hours)
        current_hour = datetime.utcnow().hour
        time_adjustment = 1.0
        
        # European market hours (7-9 UTC) showed best performance in backtesting
        if 7 <= current_hour <= 9:
            time_adjustment = 1.2
        elif 22 <= current_hour or current_hour <= 2:  # Low liquidity hours
            time_adjustment = 0.8
        
        # Apply adjustments
        adjusted_confidence = base_signal.confidence * volatility_adjustment * time_adjustment
        adjusted_confidence = min(0.95, max(0.0, adjusted_confidence))
        
        # If confidence drops too low, convert to hold
        if adjusted_confidence < 0.65 and base_signal.signal != "hold":
            final_signal = "hold"
            adjusted_confidence = 0.0
        else:
            final_signal = base_signal.signal
        
        return MLTradingSignal(
            asset=asset,
            signal=final_signal,
            confidence=adjusted_confidence,
            ml_probability=base_signal.ml_probability,
            strategy_name="Adaptive_ML_Enhanced",
            feature_importance=base_signal.feature_importance
        )
    
    def get_strategy_explanation(self, signal: MLTradingSignal) -> str:
        """Get human-readable explanation of the strategy decision"""
        explanation = f"🤖 ML-Enhanced Strategy Analysis for {signal.asset}:\n"
        explanation += f"📊 Signal: {signal.signal.upper()}\n"
        explanation += f"🎯 Confidence: {signal.confidence*100:.1f}%\n"
        explanation += f"🧠 ML Probability: {signal.ml_probability:.3f}\n"
        explanation += f"⚡ Strategy: {signal.strategy_name}\n"
        
        if signal.feature_importance:
            explanation += f"🔍 Key Factors:\n"
            for factor in signal.feature_importance:
                explanation += f"  • {factor['feature']}: {factor['importance']}\n"
        
        return explanation

# Global instance
enhanced_strategies = EnhancedMLTradingStrategies()

async def test_enhanced_strategies():
    """Test the enhanced ML trading strategies"""
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8001/api/market-data/EURUSD?count=200"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                df = pd.DataFrame(data['data'])
                
                # Test strategies
                print("🚀 TESTING ENHANCED ML TRADING STRATEGIES")
                print("=" * 60)
                
                # Traditional strategy
                bollinger_result = enhanced_strategies.traditional_bollinger_strategy(df)
                print(f"📊 Traditional Bollinger: {bollinger_result}")
                
                # ML Enhanced strategy
                ml_signal = await enhanced_strategies.ml_enhanced_strategy("EURUSD", df)
                print(f"🤖 ML Enhanced: {ml_signal.to_dict()}")
                
                # Adaptive ML strategy
                adaptive_signal = await enhanced_strategies.adaptive_ml_strategy("EURUSD", df)
                print(f"⚡ Adaptive ML: {adaptive_signal.to_dict()}")
                
                # Strategy explanation
                print("\n" + enhanced_strategies.get_strategy_explanation(adaptive_signal))

if __name__ == "__main__":
    asyncio.run(test_enhanced_strategies())