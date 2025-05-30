#!/usr/bin/env python3
"""
Real-time Trading Dashboard API
Provides endpoints for monitoring dashboard
"""

from fastapi import FastAPI, WebSocket
import asyncio
import json
import sqlite3
from datetime import datetime, timedelta
import aiohttp
import logging

dashboard_app = FastAPI(title="Trading System Dashboard")

class DashboardAPI:
    def __init__(self):
        self.base_url = "http://localhost:8001/api"
        self.active_connections = []
    
    async def get_system_metrics(self):
        """Get comprehensive system metrics"""
        try:
            # Get from database
            conn = sqlite3.connect('/app/trading_monitor.db')
            cursor = conn.cursor()
            
            # Latest system health
            cursor.execute('''
                SELECT * FROM system_health 
                ORDER BY timestamp DESC LIMIT 1
            ''')
            latest_health = cursor.fetchone()
            
            # Trading performance for last 7 days
            cursor.execute('''
                SELECT * FROM trading_performance 
                WHERE date >= date('now', '-7 days')
                ORDER BY date DESC
            ''')
            performance_history = cursor.fetchall()
            
            # Active alerts
            cursor.execute('''
                SELECT * FROM alerts 
                WHERE date(timestamp) >= date('now', '-1 day') AND resolved = 0
                ORDER BY timestamp DESC
            ''')
            active_alerts = cursor.fetchall()
            
            conn.close()
            
            return {
                'system_health': latest_health,
                'performance_history': performance_history,
                'active_alerts': active_alerts,
                'last_updated': datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logging.error(f"Error getting metrics: {e}")
            return {'error': str(e)}
    
    async def get_live_trades(self):
        """Get live trading data"""
        try:
            async with aiohttp.ClientSession() as session:
                # Get recent trades
                async with session.get(f"{self.base_url}/trades?limit=20") as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('trades', [])
                
                # Get performance
                async with session.get(f"{self.base_url}/performance") as response:
                    if response.status == 200:
                        return await response.json()
                        
        except Exception as e:
            logging.error(f"Error getting live trades: {e}")
            return []

dashboard_api = DashboardAPI()

@dashboard_app.get("/dashboard/metrics")
async def get_dashboard_metrics():
    """Get dashboard metrics"""
    return await dashboard_api.get_system_metrics()

@dashboard_app.get("/dashboard/trades")
async def get_dashboard_trades():
    """Get recent trades for dashboard"""
    return await dashboard_api.get_live_trades()

@dashboard_app.websocket("/dashboard/ws")
async def dashboard_websocket(websocket: WebSocket):
    """WebSocket for real-time dashboard updates"""
    await websocket.accept()
    dashboard_api.active_connections.append(websocket)
    
    try:
        while True:
            # Send updates every 10 seconds
            metrics = await dashboard_api.get_system_metrics()
            trades = await dashboard_api.get_live_trades()
            
            update = {
                'type': 'dashboard_update',
                'metrics': metrics,
                'trades': trades,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await websocket.send_text(json.dumps(update))
            await asyncio.sleep(10)
            
    except Exception as e:
        logging.error(f"Dashboard WebSocket error: {e}")
    finally:
        if websocket in dashboard_api.active_connections:
            dashboard_api.active_connections.remove(websocket)