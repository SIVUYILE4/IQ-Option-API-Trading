#!/usr/bin/env python3
"""
Advanced ML-Powered Trading System
Uses machine learning models to enhance trading signal accuracy and profitability
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import joblib
import os
from pathlib import Path
import asyncio
import aiohttp

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Technical Analysis
import ta
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_dir = Path("/app/ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'random_forest': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 0.25
            },
            'xgboost': {
                'model': xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'weight': 0.3
            },
            'gradient_boost': {
                'model': GradientBoostingClassifier(
                    n_estimators=150,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ),
                'weight': 0.25
            },
            'neural_network': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(100, 50),
                    max_iter=1000,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.1
                ),
                'weight': 0.2
            }
        }
        
        self.feature_importance = {}
        self.model_performance = {}
        
    def create_advanced_features(self, df):
        """Create comprehensive feature set for ML models"""
        logger.info("🔧 Creating advanced feature set...")
        
        # Ensure we have enough data
        if len(df) < 50:
            raise ValueError("Insufficient data for feature creation")
        
        features_df = df.copy()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['price_range'] = (features_df['high'] - features_df['low']) / features_df['close']
        features_df['body_size'] = abs(features_df['close'] - features_df['open']) / features_df['close']
        features_df['upper_shadow'] = (features_df['high'] - np.maximum(features_df['open'], features_df['close'])) / features_df['close']
        features_df['lower_shadow'] = (np.minimum(features_df['open'], features_df['close']) - features_df['low']) / features_df['close']
        
        # Technical Indicators
        # Moving Averages
        for period in [5, 10, 20, 50]:
            features_df[f'sma_{period}'] = ta.trend.SMAIndicator(features_df['close'], window=period).sma_indicator()
            features_df[f'ema_{period}'] = ta.trend.EMAIndicator(features_df['close'], window=period).ema_indicator()
            features_df[f'close_sma_{period}_ratio'] = features_df['close'] / features_df[f'sma_{period}']
            features_df[f'close_ema_{period}_ratio'] = features_df['close'] / features_df[f'ema_{period}']
        
        # RSI variants
        for period in [7, 14, 21]:
            features_df[f'rsi_{period}'] = ta.momentum.RSIIndicator(features_df['close'], window=period).rsi()
        
        # MACD
        macd = ta.trend.MACD(features_df['close'])
        features_df['macd'] = macd.macd()
        features_df['macd_signal'] = macd.macd_signal()
        features_df['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        for period in [20, 30]:
            bb = ta.volatility.BollingerBands(features_df['close'], window=period)
            features_df[f'bb_upper_{period}'] = bb.bollinger_hband()
            features_df[f'bb_lower_{period}'] = bb.bollinger_lband()
            features_df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            features_df[f'bb_width_{period}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            features_df[f'bb_position_{period}'] = (features_df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(features_df['high'], features_df['low'], features_df['close'])
        features_df['stoch_k'] = stoch.stoch()
        features_df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        features_df['williams_r'] = ta.momentum.WilliamsRIndicator(features_df['high'], features_df['low'], features_df['close']).williams_r()
        
        # Average True Range
        features_df['atr'] = ta.volatility.AverageTrueRange(features_df['high'], features_df['low'], features_df['close']).average_true_range()
        features_df['atr_ratio'] = features_df['atr'] / features_df['close']
        
        # Volume indicators (simplified)
        if 'volume' in features_df.columns and features_df['volume'].sum() > 0:
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
        
        # Price momentum features
        for period in [5, 10, 20]:
            features_df[f'momentum_{period}'] = features_df['close'] / features_df['close'].shift(period) - 1
            features_df[f'roc_{period}'] = ta.momentum.ROCIndicator(features_df['close'], window=period).roc()
        
        # Volatility features
        for period in [10, 20, 30]:
            features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
            features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(50).mean()
        
        # Support and Resistance levels
        features_df['support_level'] = features_df['low'].rolling(20).min()
        features_df['resistance_level'] = features_df['high'].rolling(20).max()
        features_df['support_distance'] = (features_df['close'] - features_df['support_level']) / features_df['close']
        features_df['resistance_distance'] = (features_df['resistance_level'] - features_df['close']) / features_df['close']
        
        # Market microstructure features
        features_df['bid_ask_spread'] = features_df['price_range']  # Approximation
        features_df['price_acceleration'] = features_df['returns'].diff()
        
        # Time-based features
        if 'timestamp' in features_df.columns:
            features_df['hour'] = pd.to_datetime(features_df['timestamp'], unit='s').dt.hour
            features_df['day_of_week'] = pd.to_datetime(features_df['timestamp'], unit='s').dt.dayofweek
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
        
        # Statistical features
        for period in [10, 20]:
            features_df[f'skew_{period}'] = features_df['returns'].rolling(period).skew()
            features_df[f'kurt_{period}'] = features_df['returns'].rolling(period).kurt()
        
        # Fractal features (simplified)
        features_df['fractal_high'] = ((features_df['high'] > features_df['high'].shift(1)) & 
                                      (features_df['high'] > features_df['high'].shift(-1))).astype(int)
        features_df['fractal_low'] = ((features_df['low'] < features_df['low'].shift(1)) & 
                                     (features_df['low'] < features_df['low'].shift(-1))).astype(int)
        
        # Market regime features
        features_df['trend_strength'] = abs(features_df['close'] - features_df['sma_20']) / features_df['sma_20']
        features_df['market_volatility_regime'] = (features_df['volatility_20'] > features_df['volatility_20'].rolling(50).mean()).astype(int)
        
        # Drop original OHLCV columns and non-feature columns
        feature_columns = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        return features_df[feature_columns].fillna(method='ffill').fillna(method='bfill')
    
    def create_labels(self, df, lookahead=1, threshold=0.0001):
        """Create labels for binary classification (next candle direction)"""
        # For binary options: predict if price will be higher or lower after lookahead candles
        future_price = df['close'].shift(-lookahead)
        current_price = df['close']
        
        price_change = (future_price - current_price) / current_price
        
        # Labels: 1 for price increase (CALL), 0 for price decrease (PUT)
        labels = (price_change > threshold).astype(int)
        
        return labels[:-lookahead]  # Remove last lookahead rows (no future data)
    
    def prepare_training_data(self, df):
        """Prepare features and labels for training"""
        logger.info("📊 Preparing training data...")
        
        # Create features
        features = self.create_advanced_features(df)
        
        # Create labels (predict next candle direction)
        labels = self.create_labels(df, lookahead=1, threshold=0.0001)
        
        # Align features and labels
        min_length = min(len(features), len(labels))
        features = features.iloc[:min_length]
        labels = labels[:min_length]
        
        # Remove rows with NaN values
        valid_indices = ~features.isnull().any(axis=1)
        features = features[valid_indices]
        labels = labels[valid_indices]
        
        logger.info(f"📈 Training data prepared: {len(features)} samples, {len(features.columns)} features")
        
        return features, labels
    
    def train_models(self, features, labels, asset="EURUSD"):
        """Train all ML models"""
        logger.info(f"🎯 Training ML models for {asset}...")
        
        # Split data with time series consideration
        split_index = int(len(features) * 0.8)
        X_train, X_test = features.iloc[:split_index], features.iloc[split_index:]
        y_train, y_test = labels[:split_index], labels[split_index:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(50, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        # Store scaler and selector
        self.scalers[asset] = scaler
        self.feature_selectors[asset] = selector
        
        model_results = {}
        
        # Train each model
        for model_name, config in self.model_configs.items():
            logger.info(f"  🔧 Training {model_name}...")
            
            model = config['model']
            
            # Train model
            model.fit(X_train_selected, y_train)
            
            # Evaluate
            train_score = model.score(X_train_selected, y_train)
            test_score = model.score(X_test_selected, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
            
            # Predictions for detailed analysis
            y_pred = model.predict(X_test_selected)
            
            model_results[model_name] = {
                'model': model,
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'actual': y_test
            }
            
            # Store model
            self.models[f"{asset}_{model_name}"] = model
            
            logger.info(f"    ✅ {model_name}: Train={train_score:.3f}, Test={test_score:.3f}, CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Feature importance analysis
        if hasattr(self.models[f"{asset}_random_forest"], 'feature_importances_'):
            feature_names = X_train.columns[selector.get_support()]
            importance = self.models[f"{asset}_random_forest"].feature_importances_
            self.feature_importance[asset] = dict(zip(feature_names, importance))
        
        # Store performance metrics
        self.model_performance[asset] = model_results
        
        # Save models
        self.save_models(asset)
        
        return model_results
    
    def save_models(self, asset):
        """Save trained models to disk"""
        model_file = self.model_dir / f"models_{asset}.joblib"
        scaler_file = self.model_dir / f"scaler_{asset}.joblib"
        selector_file = self.model_dir / f"selector_{asset}.joblib"
        
        # Save models
        models_to_save = {k: v for k, v in self.models.items() if k.startswith(asset)}
        joblib.dump(models_to_save, model_file)
        
        # Save scaler and selector
        if asset in self.scalers:
            joblib.dump(self.scalers[asset], scaler_file)
        if asset in self.feature_selectors:
            joblib.dump(self.feature_selectors[asset], selector_file)
        
        logger.info(f"💾 Models saved for {asset}")
    
    def load_models(self, asset):
        """Load trained models from disk"""
        model_file = self.model_dir / f"models_{asset}.joblib"
        scaler_file = self.model_dir / f"scaler_{asset}.joblib"
        selector_file = self.model_dir / f"selector_{asset}.joblib"
        
        try:
            if model_file.exists():
                loaded_models = joblib.load(model_file)
                self.models.update(loaded_models)
            
            if scaler_file.exists():
                self.scalers[asset] = joblib.load(scaler_file)
            
            if selector_file.exists():
                self.feature_selectors[asset] = joblib.load(selector_file)
            
            logger.info(f"📂 Models loaded for {asset}")
            return True
        except Exception as e:
            logger.error(f"Error loading models for {asset}: {e}")
            return False
    
    def predict_ensemble(self, features, asset="EURUSD"):
        """Make ensemble prediction using all models"""
        if asset not in self.scalers or asset not in self.feature_selectors:
            logger.warning(f"Models not ready for {asset}")
            return 0.5, "hold", 0.0
        
        try:
            # Prepare features
            features_df = self.create_advanced_features(features)
            
            if len(features_df) == 0:
                return 0.5, "hold", 0.0
            
            # Use latest row for prediction
            latest_features = features_df.iloc[-1:].fillna(method='ffill').fillna(0)
            
            # Scale and select features
            scaled_features = self.scalers[asset].transform(latest_features)
            selected_features = self.feature_selectors[asset].transform(scaled_features)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            
            for model_name, config in self.model_configs.items():
                model_key = f"{asset}_{model_name}"
                if model_key in self.models:
                    model = self.models[model_key]
                    
                    # Get prediction and probability
                    pred = model.predict(selected_features)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(selected_features)[0]
                        confidence = max(prob)
                    else:
                        confidence = 0.7  # Default confidence for models without probability
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = confidence
            
            if not predictions:
                return 0.5, "hold", 0.0
            
            # Weighted ensemble prediction
            weighted_sum = 0
            total_weight = 0
            weighted_confidence = 0
            
            for model_name, pred in predictions.items():
                weight = self.model_configs[model_name]['weight']
                confidence = probabilities[model_name]
                
                weighted_sum += pred * weight * confidence
                weighted_confidence += confidence * weight
                total_weight += weight
            
            # Final ensemble prediction
            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else 0.5
            ensemble_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
            
            # Convert to trading signal
            if ensemble_pred > 0.6 and ensemble_confidence > 0.7:
                signal = "call"
            elif ensemble_pred < 0.4 and ensemble_confidence > 0.7:
                signal = "put"
            else:
                signal = "hold"
            
            return ensemble_pred, signal, ensemble_confidence
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return 0.5, "hold", 0.0
    
    def get_feature_importance_report(self, asset="EURUSD"):
        """Get feature importance analysis"""
        if asset not in self.feature_importance:
            return "No feature importance data available"
        
        importance = self.feature_importance[asset]
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        report = f"\n🔍 TOP FEATURES FOR {asset}:\n"
        report += "=" * 50 + "\n"
        
        for i, (feature, importance_val) in enumerate(sorted_features[:15], 1):
            report += f"{i:2d}. {feature:<25}: {importance_val:.4f}\n"
        
        return report
    
    async def retrain_models(self, asset="EURUSD"):
        """Retrain models with latest data"""
        logger.info(f"🔄 Retraining models for {asset}...")
        
        # Get fresh data
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:8001/api/market-data/{asset}?count=2000"
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data['data'])
                    
                    # Prepare training data
                    features, labels = self.prepare_training_data(df)
                    
                    # Train models
                    results = self.train_models(features, labels, asset)
                    
                    logger.info(f"✅ Models retrained for {asset}")
                    return results
                else:
                    logger.error(f"Failed to get data for retraining: {response.status}")
                    return None

# Enhanced Trading Signal with ML
class MLTradingSignal:
    def __init__(self, asset, signal, confidence, ml_probability, strategy_name, feature_importance=None):
        self.asset = asset
        self.signal = signal
        self.confidence = confidence
        self.ml_probability = ml_probability
        self.strategy_name = strategy_name
        self.feature_importance = feature_importance or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        return {
            'asset': self.asset,
            'signal': self.signal,
            'confidence': self.confidence,
            'ml_probability': self.ml_probability,
            'strategy_name': self.strategy_name,
            'feature_importance': self.feature_importance[:5],  # Top 5 features
            'timestamp': self.timestamp.isoformat()
        }

async def main():
    """Test the ML trading system"""
    ml_system = MLTradingSystem()
    
    # Get data for training
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8001/api/market-data/EURUSD?count=1500"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                df = pd.DataFrame(data['data'])
                
                # Prepare and train
                features, labels = ml_system.prepare_training_data(df)
                results = ml_system.train_models(features, labels)
                
                # Show results
                print("🤖 ML TRADING SYSTEM TRAINING COMPLETE!")
                print("=" * 60)
                
                for model_name, result in results.items():
                    print(f"{model_name:<20}: {result['test_accuracy']:.3f} accuracy")
                
                # Feature importance
                print(ml_system.get_feature_importance_report())
                
                # Test prediction
                recent_data = df.tail(100)
                pred, signal, conf = ml_system.predict_ensemble(recent_data)
                print(f"\n🎯 CURRENT PREDICTION:")
                print(f"Signal: {signal}, Probability: {pred:.3f}, Confidence: {conf:.3f}")

if __name__ == "__main__":
    asyncio.run(main())