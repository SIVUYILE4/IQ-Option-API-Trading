#!/usr/bin/env python3
"""
ML-Enhanced Trading System Backtesting
Comprehensive backtesting for ML-enhanced strategies
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
sys.path.append('/app')

from enhanced_ml_strategies import EnhancedMLTradingStrategies
from ml_trading_system import MLTradingSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLBacktester:
    def __init__(self):
        self.enhanced_strategies = EnhancedMLTradingStrategies()
        self.results = {}
        
    async def comprehensive_ml_backtest(self, asset="EURUSD", test_samples=1000):
        """Comprehensive backtest of ML-enhanced strategies"""
        logger.info(f"🔬 Starting comprehensive ML backtest for {asset}")
        
        # Get historical data
        async with aiohttp.ClientSession() as session:
            url = f"http://localhost:8001/api/market-data/{asset}?count={test_samples + 200}"
            async with session.get(url) as response:
                if response.status != 200:
                    logger.error(f"Failed to get data: {response.status}")
                    return None
                
                data = await response.json()
                df = pd.DataFrame(data['data'])
        
        logger.info(f"📊 Testing on {len(df)} historical candles")
        
        # Initialize ML models
        await self.enhanced_strategies.initialize_ml_models(asset)
        
        # Backtest parameters
        strategies_to_test = [
            "traditional_bollinger",
            "traditional_macd", 
            "ml_enhanced",
            "adaptive_ml"
        ]
        
        results = {}
        
        for strategy_name in strategies_to_test:
            logger.info(f"🧪 Testing {strategy_name} strategy...")
            trades = await self.backtest_strategy(strategy_name, asset, df)
            performance = self.analyze_performance(trades, strategy_name)
            results[strategy_name] = performance
            
            logger.info(f"  ✅ {strategy_name}: {performance['win_rate']:.1f}% win rate, ${performance['total_profit']:.2f} profit")
        
        # Generate comparison report
        self.generate_ml_comparison_report(results, asset)
        
        return results
    
    async def backtest_strategy(self, strategy_name, asset, df, initial_balance=1000):
        """Backtest a specific strategy"""
        trades = []
        balance = initial_balance
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        peak_balance = balance
        
        # Test on sliding windows
        for i in range(150, len(df) - 1):  # Need sufficient history for indicators
            try:
                # Get data window
                window_data = df.iloc[:i+1].copy()
                
                # Get strategy signal
                if strategy_name == "traditional_bollinger":
                    signal_result = self.enhanced_strategies.traditional_bollinger_strategy(window_data)
                    signal = signal_result["signal"]
                    confidence = signal_result["confidence"]
                    
                elif strategy_name == "traditional_macd":
                    signal_result = self.enhanced_strategies.traditional_macd_strategy(window_data)
                    signal = signal_result["signal"]
                    confidence = signal_result["confidence"]
                    
                elif strategy_name == "ml_enhanced":
                    ml_signal = await self.enhanced_strategies.ml_enhanced_strategy(asset, window_data)
                    signal = ml_signal.signal
                    confidence = ml_signal.confidence
                    
                elif strategy_name == "adaptive_ml":
                    adaptive_signal = await self.enhanced_strategies.adaptive_ml_strategy(asset, window_data)
                    signal = adaptive_signal.signal
                    confidence = adaptive_signal.confidence
                    
                else:
                    continue
                
                # Only trade with sufficient confidence
                min_confidence = 0.7 if "ml" in strategy_name else 0.65
                if signal in ["call", "put"] and confidence >= min_confidence:
                    
                    entry_price = df.iloc[i]['close']
                    exit_price = df.iloc[i + 1]['close']
                    trade_amount = 1.0
                    
                    # Determine win/loss
                    if signal == "call":
                        win = exit_price > entry_price
                    else:  # put
                        win = exit_price < entry_price
                    
                    # Calculate profit (binary options: +80% for win, -100% for loss)
                    profit = trade_amount * 0.8 if win else -trade_amount
                    balance += profit
                    
                    # Track streaks
                    if win:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    # Track peak for drawdown
                    if balance > peak_balance:
                        peak_balance = balance
                    
                    trade = {
                        'timestamp': df.iloc[i]['timestamp'],
                        'signal': signal,
                        'confidence': confidence,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'win': win,
                        'profit': profit,
                        'balance': balance,
                        'drawdown': (peak_balance - balance) / peak_balance * 100
                    }
                    trades.append(trade)
                    
            except Exception as e:
                logger.error(f"Error in backtest at index {i}: {e}")
                continue
        
        return trades
    
    def analyze_performance(self, trades, strategy_name):
        """Analyze trading performance"""
        if not trades:
            return {
                'strategy': strategy_name,
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'avg_confidence': 0,
                'profit_factor': 0
            }
        
        df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        wins = len(df[df['win'] == True])
        win_rate = (wins / total_trades) * 100
        total_profit = df['profit'].sum()
        roi = (total_profit / 1000) * 100  # Based on $1000 starting balance
        max_drawdown = df['drawdown'].max()
        avg_confidence = df['confidence'].mean()
        
        # Profit factor
        winning_profits = df[df['win'] == True]['profit'].sum()
        losing_profits = abs(df[df['win'] == False]['profit'].sum())
        profit_factor = winning_profits / losing_profits if losing_profits > 0 else float('inf')
        
        # Sharpe ratio (simplified)
        returns = df['profit'].tolist()
        if len(returns) > 1:
            returns_mean = np.mean(returns)
            returns_std = np.std(returns)
            sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Consecutive streaks
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade['win']:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
        
        return {
            'strategy': strategy_name,
            'total_trades': total_trades,
            'winning_trades': wins,
            'losing_trades': total_trades - wins,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'avg_confidence': avg_confidence,
            'profit_factor': profit_factor,
            'final_balance': 1000 + total_profit
        }
    
    def generate_ml_comparison_report(self, results, asset):
        """Generate comprehensive comparison report"""
        print("\n" + "=" * 80)
        print("🤖 ML-ENHANCED TRADING STRATEGIES BACKTESTING RESULTS")
        print(f"📊 Asset: {asset}")
        print(f"📅 Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print("=" * 80)
        
        # Summary table
        print(f"\n📈 PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Profit':<10} {'ROI':<8} {'Drawdown':<10} {'Sharpe':<8}")
        print("-" * 80)
        
        for strategy_name, performance in results.items():
            print(f"{strategy_name:<20} {performance['total_trades']:<8} "
                  f"{performance['win_rate']:>7.1f}% {performance['total_profit']:>8.2f}$ "
                  f"{performance['roi']:>6.1f}% {performance['max_drawdown']:>8.1f}% "
                  f"{performance['sharpe_ratio']:>6.2f}")
        
        # Best performers
        print(f"\n🏆 BEST PERFORMERS:")
        
        # Best by win rate
        best_win_rate = max(results.values(), key=lambda x: x['win_rate'])
        print(f"  🎯 Highest Win Rate: {best_win_rate['strategy']} ({best_win_rate['win_rate']:.1f}%)")
        
        # Best by profit
        best_profit = max(results.values(), key=lambda x: x['total_profit'])
        print(f"  💰 Highest Profit: {best_profit['strategy']} (${best_profit['total_profit']:.2f})")
        
        # Best by Sharpe ratio
        best_sharpe = max(results.values(), key=lambda x: x['sharpe_ratio'])
        print(f"  📊 Best Risk-Adjusted: {best_sharpe['strategy']} (Sharpe: {best_sharpe['sharpe_ratio']:.2f})")
        
        # ML vs Traditional comparison
        print(f"\n🧠 ML vs TRADITIONAL ANALYSIS:")
        
        traditional_strategies = [k for k in results.keys() if 'traditional' in k]
        ml_strategies = [k for k in results.keys() if 'ml' in k]
        
        if traditional_strategies and ml_strategies:
            trad_avg_win_rate = np.mean([results[s]['win_rate'] for s in traditional_strategies])
            ml_avg_win_rate = np.mean([results[s]['win_rate'] for s in ml_strategies])
            
            trad_avg_profit = np.mean([results[s]['total_profit'] for s in traditional_strategies])
            ml_avg_profit = np.mean([results[s]['total_profit'] for s in ml_strategies])
            
            print(f"  Traditional Average Win Rate: {trad_avg_win_rate:.1f}%")
            print(f"  ML Enhanced Average Win Rate: {ml_avg_win_rate:.1f}%")
            print(f"  🚀 ML Improvement: {ml_avg_win_rate - trad_avg_win_rate:+.1f} percentage points")
            
            print(f"  Traditional Average Profit: ${trad_avg_profit:.2f}")
            print(f"  ML Enhanced Average Profit: ${ml_avg_profit:.2f}")
            print(f"  💰 ML Profit Improvement: ${ml_avg_profit - trad_avg_profit:+.2f}")
        
        # Risk analysis
        print(f"\n⚠️ RISK ANALYSIS:")
        for strategy_name, performance in results.items():
            risk_level = "LOW" if performance['max_consecutive_losses'] <= 5 else "MEDIUM" if performance['max_consecutive_losses'] <= 8 else "HIGH"
            print(f"  {strategy_name:<20}: Max {performance['max_consecutive_losses']} losses, {performance['max_drawdown']:.1f}% drawdown ({risk_level} risk)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        
        # Find the overall best strategy
        best_overall = max(results.values(), key=lambda x: x['win_rate'] * x['profit_factor'] / (x['max_drawdown'] + 1))
        print(f"  🥇 RECOMMENDED STRATEGY: {best_overall['strategy']}")
        print(f"     Win Rate: {best_overall['win_rate']:.1f}%")
        print(f"     Expected Profit: ${best_overall['total_profit']:.2f}")
        print(f"     Risk Level: {best_overall['max_drawdown']:.1f}% max drawdown")
        
        # Confidence threshold recommendations
        high_conf_strategies = [s for s in results.values() if s['avg_confidence'] > 0.8]
        if high_conf_strategies:
            print(f"  🎯 High Confidence Strategies: {len(high_conf_strategies)} strategies with >80% avg confidence")
        
        print("\n" + "=" * 80)

async def main():
    """Run comprehensive ML backtesting"""
    backtester = MLBacktester()
    
    print("🚀 STARTING COMPREHENSIVE ML BACKTESTING...")
    print("🔬 Testing ML-enhanced strategies against traditional approaches")
    print("📊 This may take a few minutes to complete...")
    
    results = await backtester.comprehensive_ml_backtest("EURUSD", 1500)
    
    if results:
        print("✅ ML Backtesting completed successfully!")
        
        # Save results
        import json
        with open('/app/ml_backtest_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            json_results = {}
            for strategy, performance in results.items():
                json_results[strategy] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in performance.items()}
            json.dump(json_results, f, indent=2)
        
        print("📄 Results saved to: /app/ml_backtest_results.json")
    else:
        print("❌ ML Backtesting failed!")

if __name__ == "__main__":
    asyncio.run(main())