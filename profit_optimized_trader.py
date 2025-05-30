#!/usr/bin/env python3
"""
Enhanced Profitable Trading System - Practical Implementation
Focuses on realistic profitability improvements with proven techniques
"""

import asyncio
import aiohttp
import json
import time
import sys
sys.path.append('/app')

from datetime import datetime
import logging
import pandas as pd
import numpy as np
from enhanced_ml_strategies import EnhancedMLTradingStrategies
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProfitOptimizedTrader:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.enhanced_strategies = EnhancedMLTradingStrategies()
        self.profit_models = {}
        self.model_dir = Path("/app/profit_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Profit optimization parameters
        self.high_confidence_threshold = 0.80  # Higher threshold for better trades
        self.min_expected_profit = 0.5  # Minimum expected profit per trade
        self.max_daily_trades = 10  # Limit trades for better selection
        
        # Advanced models for profit optimization
        self.profit_models_config = {
            'profit_xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                eval_metric='logloss'
            ),
            'profit_rf': RandomForestClassifier(
                n_estimators=150,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'profit_gb': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        self.trade_history = []
        self.daily_trades = 0
        self.session_start = datetime.utcnow().date()
        
    async def initialize_profit_models(self, asset="EURUSD"):
        """Initialize profit-optimized models"""
        logger.info(f"🎯 Initializing profit-optimized models for {asset}...")
        
        try:
            # Try loading existing models
            if self.load_profit_models(asset):
                logger.info(f"✅ Loaded existing profit models for {asset}")
                return True
            
            # Train new profit models
            await self.train_profit_models(asset)
            return True
            
        except Exception as e:
            logger.error(f"Error initializing profit models: {e}")
            return False
    
    async def train_profit_models(self, asset):
        """Train models specifically for profit optimization"""
        logger.info(f"🚀 Training profit-optimized models for {asset}...")
        
        # Get historical data
        async with aiohttp.ClientSession() as session:
            url = f"{self.base_url}/market-data/{asset}?count=1000"
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error("Failed to get training data")
                    return
                
                data = await response.json()
                df = pd.DataFrame(data['data'])
        
        # Create enhanced features
        features = self.create_profit_features(df)
        labels = self.create_profit_labels(df)
        
        # Align data
        min_length = min(len(features), len(labels))
        features = features.iloc[:min_length]
        labels = labels[:min_length]
        
        # Remove NaN values
        valid_mask = ~(features.isnull().any(axis=1) | pd.isna(labels))
        features_clean = features[valid_mask]
        labels_clean = labels[valid_mask]
        
        if len(features_clean) < 200:
            logger.warning("Insufficient data for profit model training")
            return
        
        logger.info(f"📊 Training on {len(features_clean)} samples with {len(features_clean.columns)} features")
        
        # Feature scaling and selection
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        selector = SelectKBest(score_func=f_classif, k=min(50, features_clean.shape[1]))
        features_selected = selector.fit_transform(features_scaled, labels_clean)
        
        # Train profit models
        model_performance = {}
        
        for model_name, model in self.profit_models_config.items():
            logger.info(f"  🔧 Training {model_name}...")
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = []
            cv_profits = []
            
            for train_idx, val_idx in tscv.split(features_selected):
                X_train, X_val = features_selected[train_idx], features_selected[val_idx]
                y_train, y_val = labels_clean.iloc[train_idx], labels_clean.iloc[val_idx]
                
                # Train
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate profit score
                profit = self.calculate_profit_score(y_val, y_pred, y_proba)
                cv_profits.append(profit)
                
                accuracy = (y_val == y_pred).mean()
                cv_scores.append(accuracy)
            
            avg_profit = np.mean(cv_profits)
            avg_accuracy = np.mean(cv_scores)
            
            model_performance[model_name] = {
                'profit': avg_profit,
                'accuracy': avg_accuracy,
                'model': model
            }
            
            logger.info(f"    ✅ {model_name}: {avg_accuracy:.3f} accuracy, ${avg_profit:.2f} profit score")
        
        # Store models and preprocessing
        self.profit_models[asset] = {
            'models': model_performance,
            'scaler': scaler,
            'selector': selector,
            'feature_columns': features_clean.columns.tolist()
        }
        
        # Save models
        self.save_profit_models(asset)
        
        # Find best model
        best_model = max(model_performance.items(), key=lambda x: x[1]['profit'])
        logger.info(f"🏆 Best model: {best_model[0]} with ${best_model[1]['profit']:.2f} profit score")
    
    def create_profit_features(self, df):
        """Create features optimized for profitability"""
        features = pd.DataFrame()
        
        # Price action features
        features['returns'] = df['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['price_momentum'] = df['close'] / df['close'].shift(5) - 1
        
        # Technical indicators with profit focus
        for period in [10, 20, 50]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_ratio_{period}'] = df['close'] / sma
            features[f'sma_slope_{period}'] = sma.diff(5) / sma
        
        # RSI variations
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_oversold'] = (features['rsi'] < 30).astype(int)
        features['rsi_overbought'] = (features['rsi'] > 70).astype(int)
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        features['macd'] = ema12 - ema26
        features['macd_signal'] = features['macd'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        bb_middle = df['close'].rolling(bb_period).mean()
        bb_upper = bb_middle + (df['close'].rolling(bb_period).std() * bb_std)
        bb_lower = bb_middle - (df['close'].rolling(bb_period).std() * bb_std)
        features['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
        features['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
        
        # Support/Resistance
        features['support_distance'] = df['close'] / df['low'].rolling(20).min() - 1
        features['resistance_distance'] = df['high'].rolling(20).max() / df['close'] - 1
        
        # Time-based features
        if 'timestamp' in df.columns:
            dt = pd.to_datetime(df['timestamp'], unit='s')
            features['hour'] = dt.dt.hour
            features['is_market_hours'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(int)
        
        # Profit-focused combinations
        features['momentum_rsi'] = features['price_momentum'] * (features['rsi'] / 100)
        features['volatility_macd'] = features['volatility'] * abs(features['macd_histogram'])
        
        return features.fillna(method='ffill').fillna(0)
    
    def create_profit_labels(self, df, lookahead=1):
        """Create labels focused on profitable trades"""
        future_price = df['close'].shift(-lookahead)
        current_price = df['close']
        
        price_change_pct = (future_price - current_price) / current_price * 100
        
        # More lenient labeling for practical trading
        # 1 = UP (call), 0 = DOWN (put)
        labels = (price_change_pct > 0.01).astype(int)  # 0.01% threshold
        
        return labels[:-lookahead]
    
    def calculate_profit_score(self, y_true, y_pred, y_proba, trade_amount=5.0):
        """Calculate realistic profit score"""
        profit = 0
        trades = 0
        
        for i in range(len(y_true)):
            # Only count confident predictions
            if y_proba is not None:
                confidence = max(y_proba[i])
                if confidence < 0.65:  # Lower threshold for training
                    continue
            
            trades += 1
            
            # Binary options profit calculation
            if y_true.iloc[i] == y_pred[i]:
                profit += trade_amount * 0.8  # 80% payout
            else:
                profit -= trade_amount  # 100% loss
        
        return profit if trades > 0 else 0
    
    def save_profit_models(self, asset):
        """Save profit models"""
        model_file = self.model_dir / f"profit_models_{asset}.joblib"
        joblib.dump(self.profit_models[asset], model_file)
        logger.info(f"💾 Profit models saved for {asset}")
    
    def load_profit_models(self, asset):
        """Load profit models"""
        model_file = self.model_dir / f"profit_models_{asset}.joblib"
        
        try:
            if model_file.exists():
                self.profit_models[asset] = joblib.load(model_file)
                logger.info(f"📂 Profit models loaded for {asset}")
                return True
        except Exception as e:
            logger.error(f"Error loading profit models: {e}")
        return False
    
    async def get_profit_optimized_signal(self, asset, data):
        """Get profit-optimized trading signal"""
        try:
            if asset not in self.profit_models:
                # Fallback to enhanced ML strategy
                return await self.enhanced_strategies.adaptive_ml_strategy(asset, data)
            
            # Create features
            features = self.create_profit_features(data)
            
            if len(features) == 0:
                return await self.enhanced_strategies.adaptive_ml_strategy(asset, data)
            
            # Use latest row
            latest_features = features.iloc[-1:].fillna(0)
            
            # Get model components
            models = self.profit_models[asset]['models']
            scaler = self.profit_models[asset]['scaler']
            selector = self.profit_models[asset]['selector']
            
            # Prepare features
            try:
                # Ensure feature alignment
                expected_features = self.profit_models[asset]['feature_columns']
                aligned_features = latest_features.reindex(columns=expected_features, fill_value=0)
                
                scaled_features = scaler.transform(aligned_features)
                selected_features = selector.transform(scaled_features)
            except Exception as e:
                logger.warning(f"Feature processing error: {e}")
                return await self.enhanced_strategies.adaptive_ml_strategy(asset, data)
            
            # Ensemble prediction
            predictions = []
            confidences = []
            
            for model_name, model_info in models.items():
                model = model_info['model']
                
                pred = model.predict(selected_features)[0]
                
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(selected_features)[0]
                    confidence = max(proba)
                else:
                    confidence = 0.7
                
                predictions.append(pred)
                confidences.append(confidence)
            
            # Weighted ensemble
            avg_pred = np.mean(predictions)
            avg_confidence = np.mean(confidences)
            
            # Enhanced confidence calculation
            consensus = len(set(predictions))
            if consensus == 1:  # All models agree
                final_confidence = min(0.95, avg_confidence * 1.15)
            else:
                final_confidence = avg_confidence * 0.85
            
            # Signal generation with profit optimization
            if avg_pred > 0.5 and final_confidence > self.high_confidence_threshold:
                signal = "call"
            elif avg_pred < 0.5 and final_confidence > self.high_confidence_threshold:
                signal = "put"
            else:
                signal = "hold"
            
            # Create enhanced signal object
            from enhanced_ml_strategies import MLTradingSignal
            
            return MLTradingSignal(
                asset=asset,
                signal=signal,
                confidence=final_confidence,
                ml_probability=avg_pred,
                strategy_name="Profit_Optimized_Ensemble"
            )
            
        except Exception as e:
            logger.error(f"Error in profit-optimized signal: {e}")
            return await self.enhanced_strategies.adaptive_ml_strategy(asset, data)
    
    async def should_trade(self, signal):
        """Advanced trade filtering for maximum profitability"""
        # Reset daily counter if new day
        if datetime.utcnow().date() != self.session_start:
            self.daily_trades = 0
            self.session_start = datetime.utcnow().date()
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            logger.info(f"📊 Daily trade limit reached ({self.max_daily_trades})")
            return False
        
        # Check confidence threshold
        if signal.confidence < self.high_confidence_threshold:
            return False
        
        # Calculate expected profit
        expected_profit = self.calculate_expected_profit(signal)
        if expected_profit < self.min_expected_profit:
            return False
        
        # Time-based filtering (avoid low-liquidity hours)
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 20:  # Avoid night hours
            return False
        
        return True
    
    def calculate_expected_profit(self, signal, trade_amount=5.0):
        """Calculate expected profit for the signal"""
        if signal.signal == "hold":
            return 0.0
        
        # Binary options expected value
        win_prob = signal.confidence
        expected_return = (win_prob * 0.8) - ((1 - win_prob) * 1.0)
        return trade_amount * expected_return
    
    async def execute_profit_optimized_trade(self, asset, signal):
        """Execute trade with profit optimization"""
        if not await self.should_trade(signal):
            return False
        
        try:
            trade_data = {
                "asset": asset,
                "amount": 5.0,
                "strategy": "profit_optimized",
                "auto_trade": True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/execute-trade", json=trade_data) as response:
                    result = await response.json()
                    
                    if result.get("success"):
                        self.daily_trades += 1
                        expected_profit = self.calculate_expected_profit(signal)
                        
                        trade_record = {
                            'timestamp': datetime.utcnow(),
                            'asset': asset,
                            'signal': signal.signal,
                            'confidence': signal.confidence,
                            'expected_profit': expected_profit,
                            'order_id': result.get('order_id')
                        }
                        
                        self.trade_history.append(trade_record)
                        
                        logger.info(f"🎯 PROFIT-OPTIMIZED TRADE EXECUTED!")
                        logger.info(f"   💰 Expected Profit: ${expected_profit:.2f}")
                        logger.info(f"   📊 Confidence: {signal.confidence*100:.1f}%")
                        logger.info(f"   📝 Order ID: {result.get('order_id')}")
                        logger.info(f"   📈 Daily Trades: {self.daily_trades}/{self.max_daily_trades}")
                        
                        return True
                    else:
                        logger.warning(f"Trade execution failed: {result.get('message')}")
                        return False
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return False
    
    async def run_profit_optimized_trading(self):
        """Main profit-optimized trading loop"""
        logger.info("🚀 STARTING PROFIT-OPTIMIZED TRADING SYSTEM")
        logger.info("💰 Enhanced with profit-focused ML models and advanced filtering")
        logger.info(f"🎯 High Confidence Threshold: {self.high_confidence_threshold*100:.0f}%")
        logger.info(f"💵 Min Expected Profit: ${self.min_expected_profit:.2f}")
        logger.info(f"📊 Max Daily Trades: {self.max_daily_trades}")
        logger.info("=" * 70)
        
        # Initialize profit models
        await self.initialize_profit_models("EURUSD")
        
        scan_count = 0
        
        try:
            while True:
                scan_count += 1
                
                # Check connection
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/connection-status") as response:
                        if response.status != 200:
                            logger.warning("⏸️  API disconnected")
                            await asyncio.sleep(30)
                            continue
                        
                        connection = await response.json()
                        if not connection.get("connected"):
                            logger.warning("⏸️  IQOption disconnected")
                            await asyncio.sleep(30)
                            continue
                
                logger.info(f"🧠 Profit-Optimized Scan #{scan_count}")
                logger.info(f"💰 Balance: ${connection.get('balance', 0):.2f}")
                
                # Get market data
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/market-data/EURUSD?count=200") as response:
                        if response.status == 200:
                            data = await response.json()
                            df = pd.DataFrame(data['data'])
                            
                            # Get profit-optimized signal
                            signal = await self.get_profit_optimized_signal("EURUSD", df)
                            
                            if signal.signal != "hold":
                                logger.info(f"🎯 PROFIT SIGNAL DETECTED:")
                                logger.info(f"   Signal: {signal.signal.upper()}")
                                logger.info(f"   Confidence: {signal.confidence*100:.1f}%")
                                logger.info(f"   Expected Profit: ${self.calculate_expected_profit(signal):.2f}")
                                
                                # Execute trade
                                success = await self.execute_profit_optimized_trade("EURUSD", signal)
                                
                                if success:
                                    # Wait longer after successful trade
                                    await asyncio.sleep(120)  # 2 minutes
                                else:
                                    await asyncio.sleep(10)
                            else:
                                logger.info("🤖 No profit-optimized opportunities")
                
                # Show session summary
                total_expected_profit = sum(t['expected_profit'] for t in self.trade_history)
                logger.info(f"📊 Session: {len(self.trade_history)} trades, ${total_expected_profit:.2f} expected profit")
                
                logger.info("-" * 50)
                await asyncio.sleep(45)  # Longer intervals for quality over quantity
                
        except KeyboardInterrupt:
            logger.info("🛑 Profit-optimized trader stopped")
        except Exception as e:
            logger.error(f"❌ Trading error: {e}")
        finally:
            # Final summary
            logger.info("🏁 PROFIT-OPTIMIZED SESSION COMPLETE!")
            logger.info(f"   📊 Total Trades: {len(self.trade_history)}")
            if self.trade_history:
                total_expected = sum(t['expected_profit'] for t in self.trade_history)
                avg_confidence = sum(t['confidence'] for t in self.trade_history) / len(self.trade_history)
                logger.info(f"   💰 Total Expected Profit: ${total_expected:.2f}")
                logger.info(f"   🎯 Average Confidence: {avg_confidence*100:.1f}%")

async def main():
    """Test profit-optimized system"""
    trader = ProfitOptimizedTrader()
    await trader.train_profit_models("EURUSD")

if __name__ == "__main__":
    asyncio.run(main())