#!/usr/bin/env python3
"""
Advanced ML Trading System - Version 2.0
Enhanced with reinforcement learning, advanced ensembles, and profit optimization
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
import json

# Advanced ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Advanced techniques
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# Technical Analysis
import ta
from scipy import stats
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedMLTradingSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.model_dir = Path("/app/advanced_ml_models")
        self.model_dir.mkdir(exist_ok=True)
        
        # Advanced model configurations with hyperparameter optimization
        self.advanced_models = {
            'xgboost_optimized': {
                'model': xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    gamma=0.1,
                    reg_alpha=0.1,
                    reg_lambda=1,
                    random_state=42,
                    eval_metric='logloss'
                ),
                'weight': 0.25
            },
            'lightgbm': {
                'model': lgb.LGBMClassifier(
                    n_estimators=250,
                    max_depth=7,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbosity=-1
                ),
                'weight': 0.20
            },
            'random_forest_optimized': {
                'model': RandomForestClassifier(
                    n_estimators=250,
                    max_depth=12,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='log2',
                    bootstrap=True,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 0.15
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                'weight': 0.15
            },
            'neural_network_deep': {
                'model': MLPClassifier(
                    hidden_layer_sizes=(150, 100, 50),
                    max_iter=2000,
                    learning_rate_init=0.001,
                    alpha=0.01,
                    random_state=42,
                    early_stopping=True,
                    validation_fraction=0.15
                ),
                'weight': 0.15
            },
            'svm_optimized': {
                'model': SVC(
                    C=1.0,
                    kernel='rbf',
                    gamma='scale',
                    probability=True,
                    random_state=42
                ),
                'weight': 0.10
            }
        }
        
        # Profit-focused metrics
        self.profit_history = []
        self.feature_importance = {}
        self.model_performance = {}
        self.adaptive_weights = {}
        
    def create_advanced_features(self, df):
        """Enhanced feature engineering for maximum profitability"""
        logger.info("🔧 Creating advanced profit-focused features...")
        
        if len(df) < 100:
            raise ValueError("Insufficient data for advanced feature creation")
        
        features_df = df.copy()
        
        # === PRICE ACTION FEATURES ===
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['price_range'] = (features_df['high'] - features_df['low']) / features_df['close']
        
        # Advanced price patterns
        features_df['body_size'] = abs(features_df['close'] - features_df['open']) / features_df['close']
        features_df['upper_shadow'] = (features_df['high'] - np.maximum(features_df['open'], features_df['close'])) / features_df['close']
        features_df['lower_shadow'] = (np.minimum(features_df['open'], features_df['close']) - features_df['low']) / features_df['close']
        
        # Doji patterns and candlestick analysis
        features_df['is_doji'] = (features_df['body_size'] < 0.001).astype(int)
        features_df['is_hammer'] = ((features_df['lower_shadow'] > 2 * features_df['body_size']) & 
                                   (features_df['upper_shadow'] < features_df['body_size'])).astype(int)
        features_df['is_shooting_star'] = ((features_df['upper_shadow'] > 2 * features_df['body_size']) & 
                                          (features_df['lower_shadow'] < features_df['body_size'])).astype(int)
        
        # === MOMENTUM INDICATORS ===
        for period in [5, 10, 14, 20, 30]:
            features_df[f'rsi_{period}'] = ta.momentum.RSIIndicator(features_df['close'], window=period).rsi()
            features_df[f'momentum_{period}'] = features_df['close'] / features_df['close'].shift(period) - 1
            features_df[f'roc_{period}'] = ta.momentum.ROCIndicator(features_df['close'], window=period).roc()
        
        # Stochastic indicators
        stoch = ta.momentum.StochasticOscillator(features_df['high'], features_df['low'], features_df['close'])
        features_df['stoch_k'] = stoch.stoch()
        features_df['stoch_d'] = stoch.stoch_signal()
        features_df['stoch_diff'] = features_df['stoch_k'] - features_df['stoch_d']
        
        # Williams %R
        features_df['williams_r'] = ta.momentum.WilliamsRIndicator(features_df['high'], features_df['low'], features_df['close']).williams_r()
        
        # === TREND INDICATORS ===
        for period in [5, 10, 20, 50, 100]:
            features_df[f'sma_{period}'] = ta.trend.SMAIndicator(features_df['close'], window=period).sma_indicator()
            features_df[f'ema_{period}'] = ta.trend.EMAIndicator(features_df['close'], window=period).ema_indicator()
            features_df[f'close_sma_{period}_ratio'] = features_df['close'] / features_df[f'sma_{period}']
            features_df[f'close_ema_{period}_ratio'] = features_df['close'] / features_df[f'ema_{period}']
        
        # MACD variants
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 21, 5)]:
            macd = ta.trend.MACD(features_df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
            features_df[f'macd_{fast}_{slow}'] = macd.macd()
            features_df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
            features_df[f'macd_diff_{fast}_{slow}'] = macd.macd_diff()
        
        # Parabolic SAR
        features_df['sar'] = ta.trend.PSARIndicator(features_df['high'], features_df['low'], features_df['close']).psar()
        features_df['sar_trend'] = (features_df['close'] > features_df['sar']).astype(int)
        
        # === VOLATILITY INDICATORS ===
        for period in [10, 20, 30]:
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(features_df['close'], window=period)
            features_df[f'bb_upper_{period}'] = bb.bollinger_hband()
            features_df[f'bb_lower_{period}'] = bb.bollinger_lband()
            features_df[f'bb_middle_{period}'] = bb.bollinger_mavg()
            features_df[f'bb_width_{period}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            features_df[f'bb_position_{period}'] = (features_df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            
            # ATR and volatility
            features_df[f'atr_{period}'] = ta.volatility.AverageTrueRange(features_df['high'], features_df['low'], features_df['close'], window=period).average_true_range()
            features_df[f'volatility_{period}'] = features_df['returns'].rolling(period).std()
            features_df[f'volatility_ratio_{period}'] = features_df[f'volatility_{period}'] / features_df[f'volatility_{period}'].rolling(50).mean()
        
        # Keltner Channels
        keltner = ta.volatility.KeltnerChannel(features_df['high'], features_df['low'], features_df['close'])
        features_df['keltner_upper'] = keltner.keltner_channel_hband()
        features_df['keltner_lower'] = keltner.keltner_channel_lband()
        features_df['keltner_position'] = (features_df['close'] - keltner.keltner_channel_lband()) / (keltner.keltner_channel_hband() - keltner.keltner_channel_lband())
        
        # === VOLUME INDICATORS ===
        if 'volume' in features_df.columns and features_df['volume'].sum() > 0:
            # Volume-based indicators
            features_df['volume_sma'] = features_df['volume'].rolling(20).mean()
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma']
            
            # On-Balance Volume
            features_df['obv'] = ta.volume.OnBalanceVolumeIndicator(features_df['close'], features_df['volume']).on_balance_volume()
            features_df['obv_sma'] = features_df['obv'].rolling(20).mean()
            
            # Volume Price Trend
            features_df['vpt'] = ta.volume.VolumePriceTrendIndicator(features_df['close'], features_df['volume']).volume_price_trend()
        
        # === SUPPORT/RESISTANCE ===
        for window in [10, 20, 50]:
            features_df[f'support_{window}'] = features_df['low'].rolling(window).min()
            features_df[f'resistance_{window}'] = features_df['high'].rolling(window).max()
            features_df[f'support_distance_{window}'] = (features_df['close'] - features_df[f'support_{window}']) / features_df['close']
            features_df[f'resistance_distance_{window}'] = (features_df[f'resistance_{window}'] - features_df['close']) / features_df['close']
        
        # === MARKET MICROSTRUCTURE ===
        # Price efficiency measures
        features_df['price_efficiency_10'] = abs(features_df['close'] - features_df['close'].shift(10)) / features_df['close'].rolling(10).std()
        features_df['price_acceleration'] = features_df['returns'].diff()
        features_df['price_jerk'] = features_df['price_acceleration'].diff()
        
        # Fractal dimension (Hurst exponent approximation)
        for window in [20, 50]:
            rolling_returns = features_df['returns'].rolling(window)
            features_df[f'hurst_{window}'] = rolling_returns.apply(self.calculate_hurst_exponent, raw=True)
        
        # === STATISTICAL FEATURES ===
        for period in [10, 20, 50]:
            features_df[f'skew_{period}'] = features_df['returns'].rolling(period).skew()
            features_df[f'kurt_{period}'] = features_df['returns'].rolling(period).kurt()
            features_df[f'var_{period}'] = features_df['returns'].rolling(period).var()
            
            # Z-score of price
            features_df[f'zscore_{period}'] = (features_df['close'] - features_df['close'].rolling(period).mean()) / features_df['close'].rolling(period).std()
        
        # === TIME-BASED FEATURES ===
        if 'timestamp' in features_df.columns:
            dt = pd.to_datetime(features_df['timestamp'], unit='s')
            features_df['hour'] = dt.dt.hour
            features_df['day_of_week'] = dt.dt.dayofweek
            features_df['day_of_month'] = dt.dt.day
            features_df['month'] = dt.dt.month
            
            # Cyclical encoding
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['dow_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['dow_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        # === REGIME DETECTION ===
        # Market regime indicators
        features_df['trend_strength'] = abs(features_df['close'] - features_df['sma_20']) / features_df['sma_20']
        features_df['volatility_regime'] = (features_df['volatility_20'] > features_df['volatility_20'].rolling(100).mean()).astype(int)
        
        # Bull/Bear market detection
        features_df['bull_market'] = (features_df['close'] > features_df['sma_50']).astype(int)
        features_df['market_momentum'] = features_df['ema_10'] / features_df['ema_50'] - 1
        
        # === INTERACTION FEATURES ===
        # RSI-Bollinger interaction
        features_df['rsi_bb_interaction'] = features_df['rsi_14'] * features_df['bb_position_20']
        
        # MACD-RSI interaction
        features_df['macd_rsi_interaction'] = features_df['macd_12_26'] * features_df['rsi_14'] / 100
        
        # Volume-Price interaction
        if 'volume' in features_df.columns:
            features_df['volume_price_interaction'] = features_df['volume_ratio'] * features_df['returns']
        
        # Drop original OHLCV columns
        feature_columns = [col for col in features_df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        result_df = features_df[feature_columns].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        logger.info(f"✅ Created {len(result_df.columns)} advanced features")
        return result_df
    
    def calculate_hurst_exponent(self, returns):
        """Calculate Hurst exponent for fractal analysis"""
        try:
            if len(returns) < 10:
                return 0.5
            
            lags = range(2, min(20, len(returns)//2))
            variancetau = [np.var(np.subtract(returns[lag:], returns[:-lag])) for lag in lags]
            
            if len(variancetau) < 3:
                return 0.5
                
            # Linear regression on log-log plot
            m = np.polyfit(np.log(lags), np.log(variancetau), 1)
            hurst = m[0] / 2.0
            
            # Clamp to reasonable range
            return max(0.1, min(0.9, hurst))
        except:
            return 0.5
    
    def create_profit_focused_labels(self, df, lookahead=1, profit_threshold=0.0002):
        """Create labels focused on profitable trades"""
        # Enhanced labeling for binary options profitability
        future_price = df['close'].shift(-lookahead)
        current_price = df['close']
        
        price_change = (future_price - current_price) / current_price
        
        # Create more nuanced labels
        # 1 = Strong bullish (high probability call)
        # 0 = Strong bearish (high probability put)
        # Remove marginal cases that are hard to predict
        
        strong_up = price_change > profit_threshold * 2
        strong_down = price_change < -profit_threshold * 2
        
        # Only keep high-confidence cases
        labels = np.where(strong_up, 1, np.where(strong_down, 0, -1))
        
        # Remove uncertain cases (-1)
        valid_mask = labels != -1
        
        return labels[:-lookahead], valid_mask[:-lookahead]
    
    def train_advanced_models(self, df, asset="EURUSD"):
        """Train advanced ML ensemble with hyperparameter optimization"""
        logger.info(f"🚀 Training advanced ML ensemble for {asset}...")
        
        # Create features and labels together
        features = self.create_advanced_features(df)
        labels, valid_mask = self.create_profit_focused_labels(df, 1, 0.0002)
        
        # Align features and labels
        min_length = min(len(features), len(labels))
        features = features.iloc[:min_length]
        labels = labels[:min_length]
        valid_mask = valid_mask[:min_length]
        
        # Apply valid mask
        features_clean = features[valid_mask]
        labels_clean = labels[valid_mask]
        
        if len(features_clean) < 100:
            logger.warning("Insufficient clean data for training")
            return None
        
        # Time series split to prevent lookahead bias
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Feature scaling
        scaler = RobustScaler()
        features_scaled = scaler.fit_transform(features_clean)
        
        # Advanced feature selection
        selector = SelectKBest(score_func=f_classif, k=min(80, features_clean.shape[1]))
        features_selected = selector.fit_transform(features_scaled, labels_clean)
        
        # Store preprocessing objects
        self.scalers[asset] = scaler
        self.feature_selectors[asset] = selector
        
        # Train ensemble of advanced models
        ensemble_results = {}
        
        for model_name, config in self.advanced_models.items():
            logger.info(f"  🔧 Training {model_name}...")
            
            model = config['model']
            
            # Cross-validation for model evaluation
            cv_scores = []
            cv_profits = []
            
            for train_idx, val_idx in tscv.split(features_selected):
                X_train, X_val = features_selected[train_idx], features_selected[val_idx]
                y_train, y_val = labels_clean[train_idx], labels_clean[val_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, 'predict_proba') else None
                
                # Calculate accuracy
                accuracy = accuracy_score(y_val, y_pred)
                cv_scores.append(accuracy)
                
                # Calculate profit simulation (binary options)
                profit = self.simulate_trading_profit(y_val, y_pred, y_proba)
                cv_profits.append(profit)
            
            avg_accuracy = np.mean(cv_scores)
            avg_profit = np.mean(cv_profits)
            
            ensemble_results[model_name] = {
                'model': model,
                'accuracy': avg_accuracy,
                'profit': avg_profit,
                'scores': cv_scores,
                'weight': config['weight']
            }
            
            logger.info(f"    ✅ {model_name}: {avg_accuracy:.3f} accuracy, ${avg_profit:.2f} profit")
            
            # Store model
            self.models[f"{asset}_{model_name}"] = model
        
        # Optimize ensemble weights based on profitability
        self.optimize_ensemble_weights(ensemble_results, asset)
        
        # Store performance
        self.model_performance[asset] = ensemble_results
        
        # Save models
        self.save_advanced_models(asset)
        
        return ensemble_results
    
    def simulate_trading_profit(self, y_true, y_pred, y_proba, trade_amount=5.0):
        """Simulate trading profit for model evaluation"""
        profit = 0
        confidence_threshold = 0.7
        
        for i in range(len(y_true)):
            # Only trade if confident
            if y_proba is not None:
                confidence = max(y_proba[i])
                if confidence < confidence_threshold:
                    continue
            
            # Binary options: 80% payout for win, 100% loss for loss
            if y_true[i] == y_pred[i]:
                profit += trade_amount * 0.8  # Win
            else:
                profit -= trade_amount  # Loss
        
        return profit
    
    def optimize_ensemble_weights(self, ensemble_results, asset):
        """Optimize ensemble weights for maximum profitability"""
        logger.info(f"🎯 Optimizing ensemble weights for {asset}...")
        
        # Extract profit scores
        profits = [result['profit'] for result in ensemble_results.values()]
        accuracies = [result['accuracy'] for result in ensemble_results.values()]
        
        # Weight optimization objective: maximize profit while maintaining accuracy
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            
            # Weighted profit and accuracy
            weighted_profit = np.sum([w * p for w, p in zip(weights, profits)])
            weighted_accuracy = np.sum([w * a for w, a in zip(weights, accuracies)])
            
            # Objective: maximize profit with accuracy constraint
            return -(weighted_profit + weighted_accuracy * 100)  # Negative for minimization
        
        # Initial weights
        initial_weights = np.array([0.2] * len(ensemble_results))
        
        # Constraints: weights sum to 1, all positive
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0.05, 0.5) for _ in range(len(ensemble_results))]
        
        # Optimize
        try:
            result = minimize(objective, initial_weights, method='SLSQP', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                optimized_weights = result.x / np.sum(result.x)
                
                # Update model weights
                for i, (model_name, model_result) in enumerate(ensemble_results.items()):
                    self.advanced_models[model_name]['weight'] = optimized_weights[i]
                    logger.info(f"  📊 {model_name}: weight = {optimized_weights[i]:.3f}")
                
                self.adaptive_weights[asset] = dict(zip(ensemble_results.keys(), optimized_weights))
            else:
                logger.warning("Weight optimization failed, using default weights")
        except Exception as e:
            logger.error(f"Weight optimization error: {e}")
    
    def predict_advanced_ensemble(self, features, asset="EURUSD"):
        """Advanced ensemble prediction with optimized weights"""
        if asset not in self.scalers or asset not in self.feature_selectors:
            logger.warning(f"Advanced models not ready for {asset}")
            return 0.5, "hold", 0.0, {}
        
        try:
            # Prepare features
            features_df = self.create_advanced_features(features)
            
            if len(features_df) == 0:
                return 0.5, "hold", 0.0, {}
            
            # Use latest row
            latest_features = features_df.iloc[-1:].fillna(0)
            
            # Scale and select features
            scaled_features = self.scalers[asset].transform(latest_features)
            selected_features = self.feature_selectors[asset].transform(scaled_features)
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            model_contributions = {}
            
            for model_name, config in self.advanced_models.items():
                model_key = f"{asset}_{model_name}"
                
                if model_key in self.models:
                    model = self.models[model_key]
                    
                    # Get prediction and probability
                    pred = model.predict(selected_features)[0]
                    
                    if hasattr(model, 'predict_proba'):
                        prob = model.predict_proba(selected_features)[0]
                        confidence = max(prob)
                        
                        # Store detailed prediction info
                        model_contributions[model_name] = {
                            'prediction': pred,
                            'confidence': confidence,
                            'probabilities': prob.tolist()
                        }
                    else:
                        confidence = 0.7  # Default for models without probability
                        model_contributions[model_name] = {
                            'prediction': pred,
                            'confidence': confidence,
                            'probabilities': [0.5, 0.5]
                        }
                    
                    predictions[model_name] = pred
                    probabilities[model_name] = confidence
            
            if not predictions:
                return 0.5, "hold", 0.0, {}
            
            # Weighted ensemble with optimized weights
            weighted_sum = 0
            total_weight = 0
            weighted_confidence = 0
            
            for model_name, pred in predictions.items():
                weight = self.advanced_models[model_name]['weight']
                confidence = probabilities[model_name]
                
                # Enhanced weighting: give more weight to confident models
                dynamic_weight = weight * (1 + confidence)
                
                weighted_sum += pred * dynamic_weight
                weighted_confidence += confidence * dynamic_weight
                total_weight += dynamic_weight
            
            # Final ensemble prediction
            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else 0.5
            ensemble_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
            
            # Enhanced signal generation with profit optimization
            confidence_threshold = 0.75  # Increased threshold for profitability
            
            if ensemble_pred > 0.6 and ensemble_confidence > confidence_threshold:
                signal = "call"
                final_confidence = min(0.98, ensemble_confidence * 1.1)
            elif ensemble_pred < 0.4 and ensemble_confidence > confidence_threshold:
                signal = "put"
                final_confidence = min(0.98, ensemble_confidence * 1.1)
            else:
                signal = "hold"
                final_confidence = 0.0
            
            return ensemble_pred, signal, final_confidence, model_contributions
            
        except Exception as e:
            logger.error(f"Error in advanced ensemble prediction: {e}")
            return 0.5, "hold", 0.0, {}
    
    def save_advanced_models(self, asset):
        """Save advanced models"""
        model_file = self.model_dir / f"advanced_models_{asset}.joblib"
        scaler_file = self.model_dir / f"advanced_scaler_{asset}.joblib"
        selector_file = self.model_dir / f"advanced_selector_{asset}.joblib"
        weights_file = self.model_dir / f"adaptive_weights_{asset}.joblib"
        
        # Save models
        models_to_save = {k: v for k, v in self.models.items() if k.startswith(asset)}
        joblib.dump(models_to_save, model_file)
        
        # Save preprocessing
        if asset in self.scalers:
            joblib.dump(self.scalers[asset], scaler_file)
        if asset in self.feature_selectors:
            joblib.dump(self.feature_selectors[asset], selector_file)
        if asset in self.adaptive_weights:
            joblib.dump(self.adaptive_weights[asset], weights_file)
        
        logger.info(f"💾 Advanced models saved for {asset}")
    
    def load_advanced_models(self, asset):
        """Load advanced models"""
        model_file = self.model_dir / f"advanced_models_{asset}.joblib"
        scaler_file = self.model_dir / f"advanced_scaler_{asset}.joblib"
        selector_file = self.model_dir / f"advanced_selector_{asset}.joblib"
        weights_file = self.model_dir / f"adaptive_weights_{asset}.joblib"
        
        try:
            if model_file.exists():
                loaded_models = joblib.load(model_file)
                self.models.update(loaded_models)
            
            if scaler_file.exists():
                self.scalers[asset] = joblib.load(scaler_file)
            
            if selector_file.exists():
                self.feature_selectors[asset] = joblib.load(selector_file)
            
            if weights_file.exists():
                self.adaptive_weights[asset] = joblib.load(weights_file)
                # Update model weights
                for model_name, weight in self.adaptive_weights[asset].items():
                    if model_name in self.advanced_models:
                        self.advanced_models[model_name]['weight'] = weight
            
            logger.info(f"📂 Advanced models loaded for {asset}")
            return True
        except Exception as e:
            logger.error(f"Error loading advanced models for {asset}: {e}")
            return False

# Enhanced Trading Signal with detailed ML analysis
class AdvancedMLTradingSignal:
    def __init__(self, asset, signal, confidence, ml_probability, strategy_name, 
                 model_contributions=None, feature_importance=None):
        self.asset = asset
        self.signal = signal
        self.confidence = confidence
        self.ml_probability = ml_probability
        self.strategy_name = strategy_name
        self.model_contributions = model_contributions or {}
        self.feature_importance = feature_importance or []
        self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        return {
            'asset': self.asset,
            'signal': self.signal,
            'confidence': self.confidence,
            'ml_probability': self.ml_probability,
            'strategy_name': self.strategy_name,
            'model_contributions': self.model_contributions,
            'feature_importance': self.feature_importance[:5],
            'timestamp': self.timestamp.isoformat(),
            'expected_profit': self.calculate_expected_profit()
        }
    
    def calculate_expected_profit(self, trade_amount=5.0):
        """Calculate expected profit based on confidence"""
        if self.signal == "hold":
            return 0.0
        
        # Binary options: 80% payout for win, 100% loss for loss
        win_probability = self.confidence
        expected_return = (win_probability * 0.8) - ((1 - win_probability) * 1.0)
        return trade_amount * expected_return

async def main():
    """Test the advanced ML system"""
    advanced_ml = AdvancedMLTradingSystem()
    
    # Get data for training
    async with aiohttp.ClientSession() as session:
        url = "http://localhost:8001/api/market-data/EURUSD?count=2000"
        async with session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                df = pd.DataFrame(data['data'])
                
                # Create advanced features
                features = advanced_ml.create_advanced_features(df)
                
                # Train advanced models with the dataframe
                results = advanced_ml.train_advanced_models(df)
                
                if results:
                    print("🚀 ADVANCED ML TRAINING COMPLETE!")
                    print("=" * 60)
                    
                    total_profit = sum(r['profit'] for r in results.values())
                    avg_accuracy = sum(r['accuracy'] for r in results.values()) / len(results)
                    
                    print(f"💰 Total Expected Profit: ${total_profit:.2f}")
                    print(f"🎯 Average Accuracy: {avg_accuracy:.3f}")
                    
                    for model_name, result in results.items():
                        print(f"  {model_name:<20}: {result['accuracy']:.3f} accuracy, ${result['profit']:>6.2f} profit")
                    
                    # Test advanced prediction
                    recent_data = df.tail(200)
                    pred, signal, conf, contributions = advanced_ml.predict_advanced_ensemble(recent_data)
                    
                    print(f"\n🎯 ADVANCED PREDICTION:")
                    print(f"Signal: {signal}, Probability: {pred:.3f}, Confidence: {conf:.3f}")
                    print(f"Model Contributions: {len(contributions)} models")
                    
                    # Calculate expected profit
                    expected_profit = AdvancedMLTradingSignal("EURUSD", signal, conf, pred, "Advanced_ML").calculate_expected_profit()
                    print(f"Expected Profit: ${expected_profit:.2f}")

if __name__ == "__main__":
    asyncio.run(main())