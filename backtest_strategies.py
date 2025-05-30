#!/usr/bin/env python3
"""
IQOption Strategy Backtesting System
Tests all trading strategies on historical data to validate accuracy claims
"""

import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os

# Add backend path for imports
sys.path.append('/app/backend')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StrategyBacktester:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.assets = ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"]
        self.strategies = ["combined", "rsi", "macd", "bollinger", "trend"]
        self.trade_amount = 1.0
        self.session = None
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def get_historical_data(self, asset, candles=500):
        """Get historical market data for backtesting"""
        try:
            url = f"{self.base_url}/market-data/{asset}?count={candles}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"]
                else:
                    logger.error(f"Failed to get data for {asset}: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error getting historical data for {asset}: {e}")
            return []
    
    def simulate_trade_outcome(self, direction, entry_price, exit_price):
        """Simulate trade outcome (simplified binary options simulation)"""
        if direction == "call":
            return 1 if exit_price > entry_price else -1
        elif direction == "put":
            return 1 if exit_price < entry_price else -1
        return 0
    
    async def backtest_strategy(self, asset, strategy, historical_data):
        """Backtest a specific strategy on historical data"""
        logger.info(f"Backtesting {strategy} strategy on {asset} ({len(historical_data)} candles)")
        
        if len(historical_data) < 100:
            logger.warning(f"Insufficient data for {asset}: {len(historical_data)} candles")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(historical_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        trades = []
        
        # Test strategy on sliding windows
        for i in range(100, len(df) - 1):  # Need 100 candles for indicators, leave 1 for exit
            # Get data window for analysis
            window_data = df.iloc[:i+1].copy()
            
            # Get strategy signal using the API (simulating real conditions)
            try:
                # Create a temporary CSV for this window (simplified approach)
                signal_data = {
                    'asset': asset,
                    'signal': 'hold',
                    'confidence': 0.0,
                    'strategy_name': strategy
                }
                
                # For backtesting, we'll implement the strategies locally
                signal_data = self.calculate_strategy_signal(window_data, strategy)
                
                if signal_data['signal'] in ['call', 'put'] and signal_data['confidence'] >= 0.7:
                    entry_price = df.iloc[i]['close']
                    exit_price = df.iloc[i + 1]['close']  # Next candle close (1 minute later)
                    
                    outcome = self.simulate_trade_outcome(
                        signal_data['signal'], 
                        entry_price, 
                        exit_price
                    )
                    
                    profit = self.trade_amount * 0.8 if outcome > 0 else -self.trade_amount  # 80% payout typical
                    
                    trade = {
                        'timestamp': df.iloc[i]['timestamp'],
                        'asset': asset,
                        'strategy': strategy,
                        'signal': signal_data['signal'],
                        'confidence': signal_data['confidence'],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'outcome': outcome,
                        'profit': profit
                    }
                    trades.append(trade)
                    
            except Exception as e:
                logger.error(f"Error processing window {i}: {e}")
                continue
        
        return trades
    
    def calculate_strategy_signal(self, df, strategy):
        """Calculate strategy signals locally for backtesting"""
        try:
            # Import strategy calculation functions
            from server import TradingStrategies
            strategies_obj = TradingStrategies()
            
            # Prepare data in the format expected by strategies
            data = df[['open', 'high', 'low', 'close', 'volume']].copy()
            data.columns = ['open', 'high', 'low', 'close', 'volume']
            
            if strategy == "rsi":
                signal = strategies_obj.rsi_strategy(data)
            elif strategy == "macd":
                signal = strategies_obj.macd_strategy(data)
            elif strategy == "bollinger":
                signal = strategies_obj.bollinger_strategy(data)
            elif strategy == "trend":
                signal = strategies_obj.trend_following_strategy(data)
            elif strategy == "combined":
                signal = strategies_obj.combined_strategy(data)
            else:
                signal = strategies_obj.combined_strategy(data)
            
            return {
                'signal': signal.signal,
                'confidence': signal.confidence,
                'strategy_name': signal.strategy_name
            }
            
        except Exception as e:
            logger.error(f"Error calculating {strategy} signal: {e}")
            return {'signal': 'hold', 'confidence': 0.0, 'strategy_name': strategy}
    
    def analyze_results(self, trades, asset, strategy):
        """Analyze backtest results"""
        if not trades:
            return {
                'asset': asset,
                'strategy': strategy,
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_confidence': 0,
                'profitable': False
            }
        
        df = pd.DataFrame(trades)
        
        total_trades = len(trades)
        winning_trades = len(df[df['outcome'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_profit = df['profit'].sum()
        avg_confidence = df['confidence'].mean()
        
        return {
            'asset': asset,
            'strategy': strategy,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_confidence': avg_confidence,
            'max_profit': df['profit'].max(),
            'max_loss': df['profit'].min(),
            'profitable': total_profit > 0,
            'confidence_range': f"{df['confidence'].min():.2f} - {df['confidence'].max():.2f}"
        }
    
    async def run_comprehensive_backtest(self):
        """Run backtesting on all assets and strategies"""
        logger.info("🔬 STARTING COMPREHENSIVE STRATEGY BACKTESTING")
        logger.info("=" * 60)
        
        await self.create_session()
        
        all_results = []
        
        try:
            for asset in self.assets:
                logger.info(f"\n📊 Testing asset: {asset}")
                
                # Get historical data
                historical_data = await self.get_historical_data(asset, 500)
                
                if not historical_data:
                    logger.warning(f"No data available for {asset}")
                    continue
                
                asset_results = []
                
                for strategy in self.strategies:
                    # Backtest this strategy
                    trades = await self.backtest_strategy(asset, strategy, historical_data)
                    
                    # Analyze results
                    analysis = self.analyze_results(trades, asset, strategy)
                    asset_results.append(analysis)
                    all_results.append(analysis)
                    
                    # Print strategy results
                    logger.info(f"  {strategy:>10}: {analysis['total_trades']:>3} trades, "
                              f"{analysis['win_rate']:>5.1f}% win rate, "
                              f"${analysis['total_profit']:>6.2f} profit")
                
                # Find best strategy for this asset
                if asset_results:
                    best = max(asset_results, key=lambda x: x['win_rate'])
                    logger.info(f"  🏆 Best for {asset}: {best['strategy']} "
                              f"({best['win_rate']:.1f}% win rate)")
        
        finally:
            await self.close_session()
        
        # Generate comprehensive report
        self.generate_final_report(all_results)
        
        return all_results
    
    def generate_final_report(self, all_results):
        """Generate comprehensive backtesting report"""
        logger.info("\n" + "=" * 80)
        logger.info("🎯 COMPREHENSIVE BACKTESTING RESULTS")
        logger.info("=" * 80)
        
        if not all_results:
            logger.error("❌ No backtest results available")
            return
        
        df = pd.DataFrame(all_results)
        
        # Overall statistics
        total_trades = df['total_trades'].sum()
        total_profit = df['total_profit'].sum()
        overall_win_rate = df[df['total_trades'] > 0]['win_rate'].mean()
        
        logger.info(f"\n📈 OVERALL PERFORMANCE:")
        logger.info(f"  Total Trades Simulated: {total_trades}")
        logger.info(f"  Average Win Rate: {overall_win_rate:.1f}%")
        logger.info(f"  Total Profit: ${total_profit:.2f}")
        logger.info(f"  Profitable Strategies: {len(df[df['profitable']])} of {len(df)}")
        
        # Best strategies by win rate
        logger.info(f"\n🏆 TOP STRATEGIES BY WIN RATE:")
        top_strategies = df[df['total_trades'] >= 5].sort_values('win_rate', ascending=False).head(10)
        for _, row in top_strategies.iterrows():
            logger.info(f"  {row['asset']:<8} {row['strategy']:<12} "
                       f"{row['win_rate']:>5.1f}% ({row['total_trades']} trades, "
                       f"${row['total_profit']:>6.2f})")
        
        # Strategy comparison
        logger.info(f"\n📊 STRATEGY COMPARISON (Average across all assets):")
        strategy_summary = df.groupby('strategy').agg({
            'total_trades': 'sum',
            'win_rate': 'mean',
            'total_profit': 'sum',
            'avg_confidence': 'mean'
        }).sort_values('win_rate', ascending=False)
        
        for strategy, row in strategy_summary.iterrows():
            logger.info(f"  {strategy:<12}: {row['win_rate']:>5.1f}% win rate, "
                       f"{row['total_trades']:>3.0f} trades, "
                       f"${row['total_profit']:>6.2f} profit, "
                       f"{row['avg_confidence']:>4.1f} avg confidence")
        
        # Asset performance
        logger.info(f"\n💱 ASSET PERFORMANCE:")
        asset_summary = df.groupby('asset').agg({
            'total_trades': 'sum',
            'win_rate': 'mean',
            'total_profit': 'sum'
        }).sort_values('win_rate', ascending=False)
        
        for asset, row in asset_summary.iterrows():
            logger.info(f"  {asset}: {row['win_rate']:>5.1f}% win rate, "
                       f"{row['total_trades']:>3.0f} trades, "
                       f"${row['total_profit']:>6.2f} profit")
        
        # Recommendations
        logger.info(f"\n💡 RECOMMENDATIONS:")
        best_overall = df.loc[df['win_rate'].idxmax()]
        if best_overall['win_rate'] >= 60:
            logger.info(f"  ✅ VALIDATED: {best_overall['strategy']} on {best_overall['asset']} "
                       f"shows {best_overall['win_rate']:.1f}% win rate")
        else:
            logger.info(f"  ⚠️  WARNING: Highest win rate is only {best_overall['win_rate']:.1f}%")
            logger.info(f"  🔧 RECOMMENDATION: Consider adjusting confidence thresholds or strategies")
        
        # Confidence analysis
        high_confidence_results = df[df['avg_confidence'] >= 0.8]
        if not high_confidence_results.empty:
            logger.info(f"\n🎯 HIGH CONFIDENCE STRATEGIES (≥80% avg confidence):")
            for _, row in high_confidence_results.iterrows():
                logger.info(f"  {row['asset']} {row['strategy']}: "
                           f"{row['win_rate']:.1f}% win rate, "
                           f"{row['avg_confidence']:.1f} confidence")
        
        logger.info("\n" + "=" * 80)

async def main():
    backtester = StrategyBacktester()
    results = await backtester.run_comprehensive_backtest()
    return results

if __name__ == "__main__":
    asyncio.run(main())