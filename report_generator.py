#!/usr/bin/env python3
"""
Automated Trading System Reports
Generates daily, weekly, and monthly performance reports
"""

import sqlite3
import pandas as pd
import json
from datetime import datetime, timedelta
import logging
import os
import asyncio
import aiohttp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingReportGenerator:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.db_path = "/app/trading_monitor.db"
        self.reports_dir = "/app/reports"
        
        # Create reports directory
        os.makedirs(self.reports_dir, exist_ok=True)
    
    async def get_current_data(self):
        """Get current trading data from API"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get performance
                async with session.get(f"{self.base_url}/performance") as response:
                    performance = await response.json() if response.status == 200 else {}
                
                # Get recent trades
                async with session.get(f"{self.base_url}/trades?limit=1000") as response:
                    trades_data = await response.json() if response.status == 200 else {}
                    trades = trades_data.get('trades', [])
                
                # Get connection status
                async with session.get(f"{self.base_url}/connection-status") as response:
                    connection = await response.json() if response.status == 200 else {}
                
                return performance, trades, connection
        except Exception as e:
            logger.error(f"Error getting current data: {e}")
            return {}, [], {}
    
    def get_historical_data(self, days=30):
        """Get historical data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get performance history
            performance_df = pd.read_sql_query('''
                SELECT * FROM trading_performance 
                WHERE date >= date('now', '-{} days')
                ORDER BY date
            '''.format(days), conn)
            
            # Get system health history
            health_df = pd.read_sql_query('''
                SELECT * FROM system_health 
                WHERE date(timestamp) >= date('now', '-{} days')
                ORDER BY timestamp
            '''.format(days), conn)
            
            # Get alerts history
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts 
                WHERE date(timestamp) >= date('now', '-{} days')
                ORDER BY timestamp DESC
            '''.format(days), conn)
            
            conn.close()
            
            return performance_df, health_df, alerts_df
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def calculate_performance_metrics(self, trades, performance):
        """Calculate detailed performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit_per_trade': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        df = pd.DataFrame(trades)
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = len(df[df.get('profit', 0) > 0]) if 'profit' in df.columns else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_profit = df['profit'].sum() if 'profit' in df.columns else 0
        
        # Advanced metrics
        profits = df['profit'].tolist() if 'profit' in df.columns else [0] * len(trades)
        
        # Consecutive wins/losses
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for profit in profits:
            if profit > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Profit factor
        winning_profits = [p for p in profits if p > 0]
        losing_profits = [abs(p) for p in profits if p < 0]
        
        total_wins = sum(winning_profits) if winning_profits else 0
        total_losses = sum(losing_profits) if losing_profits else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Sharpe ratio (simplified)
        if len(profits) > 1:
            returns_mean = sum(profits) / len(profits)
            returns_std = (sum([(p - returns_mean) ** 2 for p in profits]) / (len(profits) - 1)) ** 0.5
            sharpe_ratio = returns_mean / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        running_total = 0
        peak = 0
        max_drawdown = 0
        
        for profit in profits:
            running_total += profit
            if running_total > peak:
                peak = running_total
            
            drawdown = peak - running_total
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': total_profit / total_trades if total_trades > 0 else 0,
            'max_consecutive_wins': max_consecutive_wins,
            'max_consecutive_losses': max_consecutive_losses,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    def generate_daily_report(self, date=None):
        """Generate daily performance report"""
        if date is None:
            date = datetime.utcnow().date()
        
        logger.info(f"📊 Generating daily report for {date}")
        
        # Get data
        performance_df, health_df, alerts_df = self.get_historical_data(days=1)
        
        report = {
            'report_type': 'daily',
            'date': date.isoformat(),
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'trades_today': 0,
                'profit_today': 0,
                'win_rate_today': 0,
                'system_uptime': '0%',
                'alerts_count': len(alerts_df)
            },
            'details': {
                'performance': performance_df.to_dict('records') if not performance_df.empty else [],
                'system_health': health_df.to_dict('records') if not health_df.empty else [],
                'alerts': alerts_df.to_dict('records') if not alerts_df.empty else []
            }
        }
        
        # Save report
        filename = f"{self.reports_dir}/daily_report_{date.isoformat()}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Daily report saved: {filename}")
        return report
    
    def generate_weekly_report(self):
        """Generate weekly performance report"""
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=7)
        
        logger.info(f"📊 Generating weekly report: {start_date} to {end_date}")
        
        # Get data
        performance_df, health_df, alerts_df = self.get_historical_data(days=7)
        
        # Calculate weekly metrics
        weekly_metrics = {
            'total_trades': performance_df['total_trades'].sum() if not performance_df.empty else 0,
            'total_profit': performance_df['total_profit'].sum() if not performance_df.empty else 0,
            'avg_win_rate': performance_df['win_rate'].mean() if not performance_df.empty else 0,
            'best_day': performance_df.loc[performance_df['total_profit'].idxmax()].to_dict() if not performance_df.empty else {},
            'worst_day': performance_df.loc[performance_df['total_profit'].idxmin()].to_dict() if not performance_df.empty else {}
        }
        
        report = {
            'report_type': 'weekly',
            'period': f"{start_date.isoformat()} to {end_date.isoformat()}",
            'generated_at': datetime.utcnow().isoformat(),
            'summary': weekly_metrics,
            'daily_breakdown': performance_df.to_dict('records') if not performance_df.empty else [],
            'alerts_summary': {
                'total_alerts': len(alerts_df),
                'high_severity': len(alerts_df[alerts_df['severity'] == 'high']) if not alerts_df.empty else 0,
                'medium_severity': len(alerts_df[alerts_df['severity'] == 'medium']) if not alerts_df.empty else 0
            }
        }
        
        # Save report
        filename = f"{self.reports_dir}/weekly_report_{end_date.isoformat()}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Weekly report saved: {filename}")
        return report
    
    async def generate_comprehensive_report(self):
        """Generate comprehensive current status report"""
        logger.info("📊 Generating comprehensive status report...")
        
        # Get current data
        performance, trades, connection = await self.get_current_data()
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(trades, performance)
        
        # Get historical data
        performance_df, health_df, alerts_df = self.get_historical_data(days=30)
        
        report = {
            'report_type': 'comprehensive',
            'generated_at': datetime.utcnow().isoformat(),
            'system_status': {
                'api_connected': connection.get('connected', False),
                'account_balance': connection.get('balance', 0),
                'trading_mode': connection.get('mode', 'unknown')
            },
            'current_performance': metrics,
            'historical_summary': {
                'days_tracked': len(performance_df),
                'total_historical_trades': performance_df['total_trades'].sum() if not performance_df.empty else 0,
                'avg_daily_profit': performance_df['total_profit'].mean() if not performance_df.empty else 0,
                'best_performing_strategy': 'Bollinger_EURUSD'  # From backtesting
            },
            'recent_trades': trades[-10:] if trades else [],
            'alerts_summary': {
                'active_alerts': len(alerts_df[alerts_df['resolved'] == 0]) if not alerts_df.empty else 0,
                'resolved_alerts': len(alerts_df[alerts_df['resolved'] == 1]) if not alerts_df.empty else 0
            },
            'recommendations': self.generate_recommendations(metrics, performance_df)
        }
        
        # Save report
        filename = f"{self.reports_dir}/comprehensive_report_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Also save as latest
        with open(f"{self.reports_dir}/latest_comprehensive_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Comprehensive report saved: {filename}")
        return report
    
    def generate_recommendations(self, metrics, performance_df):
        """Generate trading recommendations based on performance"""
        recommendations = []
        
        # Win rate recommendations
        if metrics['win_rate'] < 50:
            recommendations.append({
                'type': 'performance',
                'severity': 'high',
                'message': f"Win rate is {metrics['win_rate']:.1f}%. Consider increasing confidence threshold or reviewing strategy parameters."
            })
        elif metrics['win_rate'] > 60:
            recommendations.append({
                'type': 'performance',
                'severity': 'positive',
                'message': f"Excellent win rate of {metrics['win_rate']:.1f}%. Current strategy is performing well."
            })
        
        # Consecutive losses
        if metrics['max_consecutive_losses'] >= 5:
            recommendations.append({
                'type': 'risk',
                'severity': 'medium',
                'message': f"Maximum consecutive losses: {metrics['max_consecutive_losses']}. Consider implementing daily loss limits."
            })
        
        # Profit factor
        if metrics['profit_factor'] < 1.0:
            recommendations.append({
                'type': 'strategy',
                'severity': 'high',
                'message': f"Profit factor is {metrics['profit_factor']:.2f} (unprofitable). Strategy adjustment needed."
            })
        
        # Trading frequency
        if metrics['total_trades'] < 10:
            recommendations.append({
                'type': 'activity',
                'severity': 'low',
                'message': f"Only {metrics['total_trades']} trades executed. Consider lowering confidence threshold if market conditions are stable."
            })
        
        return recommendations
    
    def create_html_report(self, report_data):
        """Create HTML version of report for easy viewing"""
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>IQOption Trading System Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; background-color: #e8f4f8; border-radius: 5px; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .alert-high {{ background-color: #ffebee; color: #c62828; }}
                .alert-medium {{ background-color: #fff3e0; color: #ef6c00; }}
                .alert-positive {{ background-color: #e8f5e8; color: #2e7d32; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 IQOption Trading System Report</h1>
                <p>Generated: {report_data.get('generated_at', 'Unknown')}</p>
                <p>Report Type: {report_data.get('report_type', 'Unknown').title()}</p>
            </div>
            
            <h2>📊 Performance Summary</h2>
            <div>
                <div class="metric">
                    <strong>Total Trades:</strong> {report_data.get('current_performance', {}).get('total_trades', 0)}
                </div>
                <div class="metric">
                    <strong>Win Rate:</strong> {report_data.get('current_performance', {}).get('win_rate', 0):.1f}%
                </div>
                <div class="metric">
                    <strong>Total Profit:</strong> ${report_data.get('current_performance', {}).get('total_profit', 0):.2f}
                </div>
                <div class="metric">
                    <strong>Profit Factor:</strong> {report_data.get('current_performance', {}).get('profit_factor', 0):.2f}
                </div>
            </div>
            
            <h2>⚠️ Recommendations</h2>
            {self._format_recommendations_html(report_data.get('recommendations', []))}
            
            <h2>📈 Recent Trades</h2>
            {self._format_trades_html(report_data.get('recent_trades', []))}
        </body>
        </html>
        """
        
        return html_template
    
    def _format_recommendations_html(self, recommendations):
        if not recommendations:
            return "<p>No specific recommendations at this time.</p>"
        
        html = ""
        for rec in recommendations:
            css_class = f"alert-{rec.get('severity', 'medium')}"
            html += f'<div class="alert {css_class}">{rec.get("message", "")}</div>'
        
        return html
    
    def _format_trades_html(self, trades):
        if not trades:
            return "<p>No recent trades to display.</p>"
        
        html = """
        <table>
            <tr><th>Asset</th><th>Direction</th><th>Amount</th><th>Profit</th><th>Status</th></tr>
        """
        
        for trade in trades[-10:]:  # Last 10 trades
            profit = trade.get('profit', 0)
            profit_color = 'green' if profit > 0 else 'red'
            html += f"""
            <tr>
                <td>{trade.get('asset', 'N/A')}</td>
                <td>{trade.get('direction', 'N/A').upper()}</td>
                <td>${trade.get('amount', 0):.2f}</td>
                <td style="color: {profit_color}">${profit:.2f}</td>
                <td>{trade.get('status', 'N/A')}</td>
            </tr>
            """
        
        html += "</table>"
        return html

async def main():
    """Generate reports"""
    generator = TradingReportGenerator()
    
    # Generate all reports
    logger.info("🚀 Starting automated report generation...")
    
    # Daily report
    daily_report = generator.generate_daily_report()
    
    # Weekly report (if it's Sunday)
    if datetime.utcnow().weekday() == 6:  # Sunday
        weekly_report = generator.generate_weekly_report()
    
    # Comprehensive report
    comprehensive_report = await generator.generate_comprehensive_report()
    
    # Create HTML version
    html_content = generator.create_html_report(comprehensive_report)
    with open(f"{generator.reports_dir}/latest_report.html", 'w') as f:
        f.write(html_content)
    
    logger.info("✅ All reports generated successfully!")
    logger.info(f"📁 Reports saved in: {generator.reports_dir}")

if __name__ == "__main__":
    asyncio.run(main())