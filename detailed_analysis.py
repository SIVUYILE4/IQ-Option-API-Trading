#!/usr/bin/env python3
"""
Detailed analysis of the best-performing strategy-pair from backtesting
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys

sys.path.append('/app/backend')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class BestStrategyAnalyzer:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.session = None
        
    async def create_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def get_historical_data(self, asset, candles=1000):
        """Get more historical data for detailed analysis"""
        try:
            url = f"{self.base_url}/market-data/{asset}?count={candles}"
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["data"]
        except Exception as e:
            logger.error(f"Error getting data for {asset}: {e}")
        return []
    
    def simulate_detailed_trades(self, df, strategy_name):
        """Simulate trades with detailed metrics"""
        try:
            # Import strategy functions
            from server import TradingStrategies
            strategies_obj = TradingStrategies()
            
            trades = []
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            drawdown_periods = []
            running_balance = 1000  # Starting with $1000 for percentage analysis
            peak_balance = running_balance
            
            # Test on sliding windows
            for i in range(100, len(df) - 1):
                window_data = df.iloc[:i+1].copy()
                data = window_data[['open', 'high', 'low', 'close', 'volume']].copy()
                
                # Get strategy signal
                if strategy_name == "bollinger":
                    signal = strategies_obj.bollinger_strategy(data)
                elif strategy_name == "macd":
                    signal = strategies_obj.macd_strategy(data)
                elif strategy_name == "combined":
                    signal = strategies_obj.combined_strategy(data)
                else:
                    continue
                
                if signal.signal in ['call', 'put'] and signal.confidence >= 0.7:
                    entry_price = df.iloc[i]['close']
                    exit_price = df.iloc[i + 1]['close']
                    
                    # Determine outcome
                    if strategy_name == "bollinger" and signal.signal == "call":
                        win = exit_price > entry_price
                    elif strategy_name == "bollinger" and signal.signal == "put":
                        win = exit_price < entry_price
                    elif strategy_name == "macd" and signal.signal == "call":
                        win = exit_price > entry_price
                    elif strategy_name == "macd" and signal.signal == "put":
                        win = exit_price < entry_price
                    else:
                        win = (exit_price > entry_price) if signal.signal == "call" else (exit_price < entry_price)
                    
                    # Calculate profit (binary options: +80% for win, -100% for loss)
                    profit = 1.0 * 0.8 if win else -1.0
                    running_balance += profit
                    
                    # Track peak for drawdown calculation
                    if running_balance > peak_balance:
                        peak_balance = running_balance
                    
                    # Track consecutive wins/losses
                    if win:
                        consecutive_wins += 1
                        consecutive_losses = 0
                        max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                    else:
                        consecutive_losses += 1
                        consecutive_wins = 0
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                    
                    # Calculate drawdown percentage
                    drawdown = ((peak_balance - running_balance) / peak_balance) * 100
                    
                    trade = {
                        'timestamp': df.iloc[i]['timestamp'],
                        'signal': signal.signal,
                        'confidence': signal.confidence,
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'win': win,
                        'profit': profit,
                        'running_balance': running_balance,
                        'drawdown': drawdown,
                        'consecutive_wins': consecutive_wins,
                        'consecutive_losses': consecutive_losses
                    }
                    trades.append(trade)
            
            return trades, max_consecutive_wins, max_consecutive_losses
            
        except Exception as e:
            logger.error(f"Error in detailed simulation: {e}")
            return [], 0, 0
    
    def analyze_best_strategy_detailed(self, trades, strategy_name, asset):
        """Provide comprehensive analysis of the best strategy"""
        if not trades:
            logger.error(f"No trades found for {strategy_name} on {asset}")
            return
        
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        wins = len(df[df['win'] == True])
        losses = total_trades - wins
        win_rate = (wins / total_trades) * 100
        total_profit = df['profit'].sum()
        
        # Advanced metrics
        avg_win = df[df['win'] == True]['profit'].mean() if wins > 0 else 0
        avg_loss = df[df['win'] == False]['profit'].mean() if losses > 0 else 0
        profit_factor = abs(avg_win * wins / avg_loss / losses) if losses > 0 and avg_loss != 0 else float('inf')
        
        # Risk metrics
        max_drawdown = df['drawdown'].max()
        final_balance = df['running_balance'].iloc[-1]
        roi = ((final_balance - 1000) / 1000) * 100
        
        # Streak analysis
        max_wins = df['consecutive_wins'].max()
        max_losses = df['consecutive_losses'].max()
        
        # Confidence analysis
        avg_confidence = df['confidence'].mean()
        high_conf_trades = df[df['confidence'] >= 0.8]
        high_conf_win_rate = (len(high_conf_trades[high_conf_trades['win']]) / len(high_conf_trades)) * 100 if len(high_conf_trades) > 0 else 0
        
        # Time-based analysis
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['timestamp'].dt.hour
        hourly_performance = df.groupby('hour')['win'].mean() * 100
        
        # Print comprehensive results
        print("=" * 80)
        print(f"🎯 DETAILED ANALYSIS: {strategy_name.upper()} STRATEGY ON {asset}")
        print("=" * 80)
        
        print(f"\n📊 BASIC PERFORMANCE:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Wins: {wins}")
        print(f"  Losses: {losses}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Total Profit: ${total_profit:.2f}")
        print(f"  ROI: {roi:.2f}%")
        
        print(f"\n💰 PROFITABILITY ANALYSIS:")
        print(f"  Average Win: ${avg_win:.2f}")
        print(f"  Average Loss: ${avg_loss:.2f}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"  Final Balance: ${final_balance:.2f} (started with $1000)")
        
        print(f"\n⚠️ RISK ANALYSIS:")
        print(f"  Maximum Drawdown: {max_drawdown:.2f}%")
        print(f"  Max Consecutive Wins: {max_wins}")
        print(f"  Max Consecutive Losses: {max_losses}")
        
        print(f"\n🎯 CONFIDENCE ANALYSIS:")
        print(f"  Average Confidence: {avg_confidence:.3f} ({avg_confidence*100:.1f}%)")
        print(f"  High Confidence Trades (≥80%): {len(high_conf_trades)}")
        print(f"  High Confidence Win Rate: {high_conf_win_rate:.2f}%")
        print(f"  Confidence Range: {df['confidence'].min():.3f} - {df['confidence'].max():.3f}")
        
        print(f"\n⏰ HOURLY PERFORMANCE (Best Hours UTC):")
        best_hours = hourly_performance.sort_values(ascending=False).head(5)
        for hour, win_rate in best_hours.items():
            print(f"  {hour:02d}:00 UTC: {win_rate:.1f}% win rate ({len(df[df['hour'] == hour])} trades)")
        
        print(f"\n📈 TRADE DISTRIBUTION:")
        confidence_bins = pd.cut(df['confidence'], bins=[0.7, 0.75, 0.8, 0.85, 0.9, 1.0], labels=['70-75%', '75-80%', '80-85%', '85-90%', '90%+'])
        for bin_label, group in df.groupby(confidence_bins):
            if len(group) > 0:
                bin_win_rate = (len(group[group['win']]) / len(group)) * 100
                print(f"  {bin_label} confidence: {len(group)} trades, {bin_win_rate:.1f}% win rate")
        
        print(f"\n🏆 RECOMMENDATION:")
        if win_rate >= 55 and total_profit > 0:
            print(f"  ✅ PROFITABLE STRATEGY - Recommended for live trading")
            print(f"  💡 Best confidence threshold: ≥{df[df['win']]['confidence'].quantile(0.3):.2f}")
            if len(best_hours) > 0:
                print(f"  ⏰ Best trading hours: {best_hours.index[0]:02d}:00-{best_hours.index[1]:02d}:00 UTC")
        else:
            print(f"  ⚠️  MARGINAL STRATEGY - Use with caution or higher confidence threshold")
        
        return {
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_trades': total_trades,
            'roi': roi,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor
        }
    
    async def analyze_best_performers(self):
        """Analyze the top-performing strategy combinations"""
        await self.create_session()
        
        try:
            # Analyze top performers: EUR/USD with Bollinger and MACD
            logger.info("🔍 Analyzing best-performing strategies from backtesting...")
            
            # Get historical data for EUR/USD
            historical_data = await self.get_historical_data("EURUSD", 1000)
            
            if not historical_data:
                logger.error("No historical data available")
                return
            
            df = pd.DataFrame(historical_data)
            
            print("🔬 COMPREHENSIVE ANALYSIS OF BEST STRATEGY-PAIRS")
            print("Based on backtesting results, analyzing top performers:")
            print("1. EUR/USD + Bollinger Bands (57.4% win rate)")
            print("2. EUR/USD + MACD (57.1% win rate)")
            print("3. EUR/USD + Combined Strategy (optimized)")
            
            # Analyze Bollinger strategy
            bollinger_trades, bb_max_wins, bb_max_losses = self.simulate_detailed_trades(df, "bollinger")
            bb_results = self.analyze_best_strategy_detailed(bollinger_trades, "Bollinger Bands", "EURUSD")
            
            # Analyze MACD strategy  
            macd_trades, macd_max_wins, macd_max_losses = self.simulate_detailed_trades(df, "macd")
            macd_results = self.analyze_best_strategy_detailed(macd_trades, "MACD", "EURUSD")
            
            # Analyze Combined strategy
            combined_trades, comb_max_wins, comb_max_losses = self.simulate_detailed_trades(df, "combined")
            combined_results = self.analyze_best_strategy_detailed(combined_trades, "Combined (Optimized)", "EURUSD")
            
            # Comparison summary
            print("\n" + "=" * 80)
            print("🏆 STRATEGY COMPARISON SUMMARY")
            print("=" * 80)
            
            strategies = [
                ("Bollinger Bands", bb_results),
                ("MACD", macd_results), 
                ("Combined", combined_results)
            ]
            
            print(f"{'Strategy':<20} {'Win Rate':<10} {'Profit':<10} {'Trades':<8} {'ROI':<8} {'Drawdown':<10}")
            print("-" * 80)
            
            for name, results in strategies:
                if results:
                    print(f"{name:<20} {results['win_rate']:>7.1f}% {results['total_profit']:>7.2f}$ {results['total_trades']:>6} {results['roi']:>6.1f}% {results['max_drawdown']:>8.1f}%")
            
            # Final recommendation
            best_strategy = max(strategies, key=lambda x: x[1]['win_rate'] if x[1] else 0)
            print(f"\n🥇 BEST PERFORMER: {best_strategy[0]}")
            print(f"   This strategy should be prioritized for live trading")
            
        finally:
            await self.close_session()

async def main():
    analyzer = BestStrategyAnalyzer()
    await analyzer.analyze_best_performers()

if __name__ == "__main__":
    asyncio.run(main())