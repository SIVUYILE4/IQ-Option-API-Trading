#!/usr/bin/env python3
"""
IQOption Trading System - Production Monitoring Dashboard
Real-time monitoring, alerting, and performance tracking
"""

import asyncio
import aiohttp
import json
import time
from datetime import datetime, timedelta
import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingSystemMonitor:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.frontend_url = "http://localhost:3000"
        self.session = None
        
        # Monitoring thresholds
        self.max_consecutive_losses = 5
        self.max_daily_loss = 10.0  # $10 daily loss limit
        self.min_win_rate_threshold = 45.0  # Alert if win rate drops below 45%
        self.connection_timeout = 30  # seconds
        
        # Performance tracking
        self.daily_stats = {}
        self.alerts_sent = set()
        
        # Initialize database for historical tracking
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for historical tracking"""
        try:
            conn = sqlite3.connect('/app/trading_monitor.db')
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_health (
                    timestamp TEXT PRIMARY KEY,
                    api_connected BOOLEAN,
                    balance REAL,
                    frontend_status TEXT,
                    backend_status TEXT,
                    validator_running BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trading_performance (
                    date TEXT PRIMARY KEY,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    win_rate REAL,
                    daily_profit REAL,
                    total_profit REAL,
                    avg_confidence REAL,
                    best_strategy TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    timestamp TEXT PRIMARY KEY,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    resolved BOOLEAN
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("📊 Database initialized successfully")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    async def create_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self):
        if self.session:
            await self.session.close()
    
    async def check_api_health(self):
        """Check backend API health"""
        try:
            async with self.session.get(f"{self.base_url}/connection-status") as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'status': 'healthy',
                        'connected': data.get('connected', False),
                        'balance': data.get('balance', 0),
                        'mode': data.get('mode', 'unknown')
                    }
                else:
                    return {'status': 'unhealthy', 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def check_frontend_health(self):
        """Check frontend health"""
        try:
            async with self.session.get(self.frontend_url) as response:
                if response.status == 200:
                    return {'status': 'healthy'}
                else:
                    return {'status': 'unhealthy', 'error': f'HTTP {response.status}'}
        except Exception as e:
            return {'status': 'error', 'error': str(e)}
    
    async def get_trading_performance(self):
        """Get current trading performance metrics"""
        try:
            async with self.session.get(f"{self.base_url}/performance") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {}
        except Exception as e:
            logger.error(f"Error getting performance: {e}")
            return {}
    
    async def get_recent_trades(self):
        """Get recent trades for analysis"""
        try:
            async with self.session.get(f"{self.base_url}/trades?limit=100") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get('trades', [])
                else:
                    return []
        except Exception as e:
            logger.error(f"Error getting trades: {e}")
            return []
    
    def check_validator_process(self):
        """Check if validated trader is running"""
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', 'validated_trader.py'], 
                                  capture_output=True, text=True)
            return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.error(f"Error checking validator process: {e}")
            return False
    
    def analyze_consecutive_losses(self, trades):
        """Analyze for consecutive losses"""
        if len(trades) < 2:
            return 0
        
        consecutive_losses = 0
        max_consecutive = 0
        
        for trade in sorted(trades, key=lambda x: x.get('timestamp', '')):
            if trade.get('result') == 'loss' or trade.get('profit', 0) < 0:
                consecutive_losses += 1
                max_consecutive = max(max_consecutive, consecutive_losses)
            else:
                consecutive_losses = 0
        
        return max_consecutive
    
    def store_system_health(self, health_data):
        """Store system health metrics in database"""
        try:
            conn = sqlite3.connect('/app/trading_monitor.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO system_health 
                (timestamp, api_connected, balance, frontend_status, backend_status, validator_running)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                health_data.get('api_connected', False),
                health_data.get('balance', 0),
                health_data.get('frontend_status', 'unknown'),
                health_data.get('backend_status', 'unknown'),
                health_data.get('validator_running', False)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing health data: {e}")
    
    def store_performance_data(self, performance):
        """Store daily performance data"""
        try:
            conn = sqlite3.connect('/app/trading_monitor.db')
            cursor = conn.cursor()
            
            today = datetime.utcnow().date().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trading_performance 
                (date, total_trades, winning_trades, losing_trades, win_rate, 
                 daily_profit, total_profit, avg_confidence, best_strategy)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today,
                performance.get('total_trades', 0),
                performance.get('winning_trades', 0),
                performance.get('losing_trades', 0),
                performance.get('win_rate', 0),
                0,  # Calculate daily profit separately
                performance.get('total_profit', 0),
                performance.get('average_confidence', 0),
                'Bollinger_EURUSD'  # Best validated strategy
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error storing performance data: {e}")
    
    def create_alert(self, alert_type, message, severity='medium'):
        """Create and store alert"""
        try:
            alert_key = f"{alert_type}_{datetime.utcnow().date()}"
            
            if alert_key not in self.alerts_sent:
                conn = sqlite3.connect('/app/trading_monitor.db')
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO alerts (timestamp, alert_type, message, severity, resolved)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    datetime.utcnow().isoformat(),
                    alert_type,
                    message,
                    severity,
                    False
                ))
                
                conn.commit()
                conn.close()
                
                self.alerts_sent.add(alert_key)
                logger.warning(f"🚨 ALERT [{severity.upper()}]: {message}")
                
                return True
        except Exception as e:
            logger.error(f"Error creating alert: {e}")
        return False
    
    def generate_status_report(self, health_data, performance, trades):
        """Generate comprehensive status report"""
        report = []
        report.append("=" * 80)
        report.append("🤖 IQOPTION TRADING SYSTEM - STATUS REPORT")
        report.append(f"📅 {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        report.append("=" * 80)
        
        # System Health
        report.append("\n🏥 SYSTEM HEALTH:")
        report.append(f"  API Connection: {'✅ Connected' if health_data.get('api_connected') else '❌ Disconnected'}")
        report.append(f"  Account Balance: ${health_data.get('balance', 0):.2f}")
        report.append(f"  Frontend Status: {health_data.get('frontend_status', 'unknown')}")
        report.append(f"  Backend Status: {health_data.get('backend_status', 'unknown')}")
        report.append(f"  Validator Running: {'✅ Active' if health_data.get('validator_running') else '❌ Stopped'}")
        
        # Trading Performance
        report.append("\n📊 TRADING PERFORMANCE:")
        report.append(f"  Total Trades: {performance.get('total_trades', 0)}")
        report.append(f"  Win Rate: {performance.get('win_rate', 0):.1f}%")
        report.append(f"  Total Profit: ${performance.get('total_profit', 0):.2f}")
        report.append(f"  Avg Confidence: {performance.get('average_confidence', 0)*100:.1f}%")
        
        # Recent Activity
        recent_trades = trades[-5:] if trades else []
        report.append(f"\n📈 RECENT ACTIVITY ({len(recent_trades)} latest trades):")
        for trade in recent_trades:
            profit = trade.get('profit', 0)
            status = '✅ WIN' if profit > 0 else '❌ LOSS'
            report.append(f"  {trade.get('asset', 'N/A')} {trade.get('direction', 'N/A')} - {status} (${profit:.2f})")
        
        # Alerts
        try:
            conn = sqlite3.connect('/app/trading_monitor.db')
            cursor = conn.cursor()
            cursor.execute('''
                SELECT alert_type, message, severity 
                FROM alerts 
                WHERE date(timestamp) = date('now') AND resolved = 0
                ORDER BY timestamp DESC LIMIT 5
            ''')
            active_alerts = cursor.fetchall()
            conn.close()
            
            if active_alerts:
                report.append(f"\n🚨 ACTIVE ALERTS ({len(active_alerts)}):")
                for alert_type, message, severity in active_alerts:
                    icon = "🔴" if severity == "high" else "🟡" if severity == "medium" else "🟢"
                    report.append(f"  {icon} {alert_type}: {message}")
            else:
                report.append("\n✅ NO ACTIVE ALERTS")
        except Exception as e:
            report.append(f"\n❌ Error getting alerts: {e}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)
    
    async def comprehensive_system_check(self):
        """Perform comprehensive system health check"""
        logger.info("🔍 Performing comprehensive system check...")
        
        # Check all system components
        api_health = await self.check_api_health()
        frontend_health = await self.check_frontend_health()
        validator_running = self.check_validator_process()
        performance = await self.get_trading_performance()
        trades = await self.get_recent_trades()
        
        # Compile health data
        health_data = {
            'api_connected': api_health.get('status') == 'healthy' and api_health.get('connected', False),
            'balance': api_health.get('balance', 0),
            'frontend_status': frontend_health.get('status', 'unknown'),
            'backend_status': api_health.get('status', 'unknown'),
            'validator_running': validator_running
        }
        
        # Store metrics
        self.store_system_health(health_data)
        self.store_performance_data(performance)
        
        # Check for alerts
        await self.check_alerts(health_data, performance, trades)
        
        # Generate and log status report
        status_report = self.generate_status_report(health_data, performance, trades)
        logger.info(status_report)
        
        # Save report to file
        with open('/app/latest_status_report.txt', 'w') as f:
            f.write(status_report)
        
        return health_data, performance, trades
    
    async def check_alerts(self, health_data, performance, trades):
        """Check for alert conditions"""
        # API Connection Alert
        if not health_data.get('api_connected'):
            self.create_alert('api_disconnected', 
                            'IQOption API connection lost', 'high')
        
        # Validator Process Alert
        if not health_data.get('validator_running'):
            self.create_alert('validator_stopped', 
                            'Validated trader process not running', 'high')
        
        # Performance Alerts
        win_rate = performance.get('win_rate', 0)
        if win_rate < self.min_win_rate_threshold and performance.get('total_trades', 0) >= 10:
            self.create_alert('low_win_rate', 
                            f'Win rate dropped to {win_rate:.1f}% (threshold: {self.min_win_rate_threshold}%)', 
                            'medium')
        
        # Consecutive losses alert
        if trades:
            consecutive_losses = self.analyze_consecutive_losses(trades)
            if consecutive_losses >= self.max_consecutive_losses:
                self.create_alert('consecutive_losses', 
                                f'{consecutive_losses} consecutive losses detected', 'high')
        
        # Balance Alert
        balance = health_data.get('balance', 0)
        if balance < 30:  # Alert if balance drops below $30
            self.create_alert('low_balance', 
                            f'Account balance low: ${balance:.2f}', 'medium')
    
    async def run_monitoring(self):
        """Main monitoring loop"""
        logger.info("🚀 STARTING TRADING SYSTEM MONITOR")
        logger.info("🔍 Comprehensive monitoring and alerting system")
        logger.info("📊 Real-time performance tracking")
        logger.info("🚨 Automated alert system")
        logger.info("=" * 70)
        
        await self.create_session()
        
        check_count = 0
        
        try:
            while True:
                check_count += 1
                logger.info(f"🔍 System Check #{check_count}")
                
                # Perform comprehensive check
                health_data, performance, trades = await self.comprehensive_system_check()
                
                # Brief summary
                api_status = "✅" if health_data.get('api_connected') else "❌"
                validator_status = "✅" if health_data.get('validator_running') else "❌"
                balance = health_data.get('balance', 0)
                total_trades = performance.get('total_trades', 0)
                win_rate = performance.get('win_rate', 0)
                
                logger.info(f"📊 SUMMARY: API {api_status} | Validator {validator_status} | "
                          f"Balance: ${balance:.2f} | Trades: {total_trades} | "
                          f"Win Rate: {win_rate:.1f}%")
                
                logger.info("-" * 50)
                
                # Wait 60 seconds between checks
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("🛑 Monitoring stopped by user")
        except Exception as e:
            logger.error(f"❌ Monitoring error: {e}")
        finally:
            await self.close_session()
            logger.info("🏁 Monitoring session ended")

async def main():
    monitor = TradingSystemMonitor()
    await monitor.run_monitoring()

if __name__ == "__main__":
    asyncio.run(main())