#!/usr/bin/env python3
"""
IQOption Trading System - Deployment Manager
Handles deployment, startup, shutdown, and service management
"""

import subprocess
import time
import os
import signal
import sys
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TradingSystemDeployment:
    def __init__(self):
        self.services = {
            'backend': {'port': 8001, 'process': None},
            'frontend': {'port': 3000, 'process': None},
            'mongodb': {'port': 27017, 'process': None},
            'validator': {'script': '/app/validated_trader.py', 'process': None},
            'monitor': {'script': '/app/system_monitor.py', 'process': None}
        }
        
        self.log_files = {
            'backend': '/var/log/supervisor/backend.out.log',
            'frontend': '/var/log/supervisor/frontend.out.log',
            'validator': '/app/validated_trading.log',
            'monitor': '/app/monitoring.log'
        }
        
    def check_service_health(self, service_name):
        """Check if a service is healthy"""
        try:
            if service_name in ['backend', 'frontend', 'mongodb']:
                # Check supervisor services
                result = subprocess.run(
                    ['sudo', 'supervisorctl', 'status', service_name],
                    capture_output=True, text=True
                )
                return 'RUNNING' in result.stdout
            else:
                # Check custom processes
                result = subprocess.run(
                    ['pgrep', '-f', self.services[service_name]['script']],
                    capture_output=True, text=True
                )
                return len(result.stdout.strip()) > 0
        except Exception as e:
            logger.error(f"Error checking {service_name}: {e}")
            return False
    
    def start_service(self, service_name):
        """Start a specific service"""
        try:
            if service_name in ['backend', 'frontend', 'mongodb']:
                logger.info(f"🚀 Starting {service_name} via supervisor...")
                result = subprocess.run(
                    ['sudo', 'supervisorctl', 'start', service_name],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.info(f"✅ {service_name} started successfully")
                    return True
                else:
                    logger.error(f"❌ Failed to start {service_name}: {result.stderr}")
                    return False
            
            elif service_name == 'validator':
                logger.info("🚀 Starting validated trader...")
                subprocess.Popen([
                    'python', '/app/validated_trader.py'
                ], stdout=open('/app/validated_trading.log', 'a'),
                   stderr=subprocess.STDOUT)
                time.sleep(2)
                return self.check_service_health('validator')
            
            elif service_name == 'monitor':
                logger.info("🚀 Starting system monitor...")
                subprocess.Popen([
                    'python', '/app/system_monitor.py'
                ], stdout=open('/app/monitoring.log', 'a'),
                   stderr=subprocess.STDOUT)
                time.sleep(2)
                return self.check_service_health('monitor')
                
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            return False
    
    def stop_service(self, service_name):
        """Stop a specific service"""
        try:
            if service_name in ['backend', 'frontend', 'mongodb']:
                logger.info(f"🛑 Stopping {service_name}...")
                result = subprocess.run(
                    ['sudo', 'supervisorctl', 'stop', service_name],
                    capture_output=True, text=True
                )
                return result.returncode == 0
            
            else:
                logger.info(f"🛑 Stopping {service_name}...")
                subprocess.run(['pkill', '-f', self.services[service_name]['script']])
                return True
                
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
            return False
    
    def restart_service(self, service_name):
        """Restart a specific service"""
        logger.info(f"🔄 Restarting {service_name}...")
        self.stop_service(service_name)
        time.sleep(2)
        return self.start_service(service_name)
    
    def deploy_full_system(self):
        """Deploy the complete trading system"""
        logger.info("🚀 DEPLOYING IQOPTION TRADING SYSTEM")
        logger.info("=" * 60)
        
        deployment_success = True
        
        # Start core services in order
        startup_order = ['mongodb', 'backend', 'frontend', 'validator', 'monitor']
        
        for service in startup_order:
            logger.info(f"📦 Deploying {service}...")
            
            if self.start_service(service):
                logger.info(f"✅ {service} deployed successfully")
                time.sleep(3)  # Wait between services
            else:
                logger.error(f"❌ Failed to deploy {service}")
                deployment_success = False
                break
        
        if deployment_success:
            logger.info("🎉 FULL SYSTEM DEPLOYMENT SUCCESSFUL!")
            self.show_system_status()
            self.show_access_urls()
        else:
            logger.error("💥 DEPLOYMENT FAILED - Check logs for details")
        
        return deployment_success
    
    def show_system_status(self):
        """Show current system status"""
        logger.info("\n📊 SYSTEM STATUS CHECK:")
        logger.info("-" * 40)
        
        all_healthy = True
        
        for service_name in self.services.keys():
            is_healthy = self.check_service_health(service_name)
            status = "✅ RUNNING" if is_healthy else "❌ STOPPED"
            logger.info(f"  {service_name:<12}: {status}")
            
            if not is_healthy:
                all_healthy = False
        
        if all_healthy:
            logger.info("\n🎯 ALL SYSTEMS OPERATIONAL!")
        else:
            logger.warning("\n⚠️  SOME SERVICES ARE DOWN!")
    
    def show_access_urls(self):
        """Show access URLs and important information"""
        logger.info("\n🌐 ACCESS INFORMATION:")
        logger.info("-" * 40)
        logger.info("  Frontend Dashboard: http://localhost:3000")
        logger.info("  Backend API: http://localhost:8001/api")
        logger.info("  API Docs: http://localhost:8001/docs")
        logger.info("  System Logs: /app/monitoring.log")
        logger.info("  Trading Logs: /app/validated_trading.log")
        logger.info("  Status Reports: /app/latest_status_report.txt")
    
    def show_monitoring_commands(self):
        """Show useful monitoring commands"""
        logger.info("\n🔍 MONITORING COMMANDS:")
        logger.info("-" * 40)
        logger.info("  System Status: python /app/deployment_manager.py --status")
        logger.info("  Live Trading Log: tail -f /app/validated_trading.log")
        logger.info("  Live Monitor Log: tail -f /app/monitoring.log")
        logger.info("  Latest Report: cat /app/latest_status_report.txt")
        logger.info("  Restart All: python /app/deployment_manager.py --restart")
    
    def emergency_shutdown(self):
        """Emergency shutdown of all services"""
        logger.warning("🚨 EMERGENCY SHUTDOWN INITIATED")
        
        for service_name in reversed(list(self.services.keys())):
            logger.info(f"🛑 Emergency stop: {service_name}")
            self.stop_service(service_name)
        
        logger.info("🛑 EMERGENCY SHUTDOWN COMPLETE")
    
    def create_startup_script(self):
        """Create a startup script for easy deployment"""
        startup_script = '''#!/bin/bash
# IQOption Trading System Startup Script

echo "🚀 Starting IQOption Trading System..."

# Change to app directory
cd /app

# Start the deployment
python deployment_manager.py --deploy

echo "✅ Startup complete! Check logs for status."
'''
        
        with open('/app/start_trading_system.sh', 'w') as f:
            f.write(startup_script)
        
        os.chmod('/app/start_trading_system.sh', 0o755)
        logger.info("📜 Startup script created: /app/start_trading_system.sh")
    
    def tail_logs(self, service_name, lines=50):
        """Show recent logs for a service"""
        if service_name in self.log_files:
            log_file = self.log_files[service_name]
            if os.path.exists(log_file):
                try:
                    result = subprocess.run(
                        ['tail', '-n', str(lines), log_file],
                        capture_output=True, text=True
                    )
                    logger.info(f"\n📋 RECENT {service_name.upper()} LOGS:")
                    logger.info("-" * 50)
                    print(result.stdout)
                except Exception as e:
                    logger.error(f"Error reading logs: {e}")
            else:
                logger.warning(f"Log file not found: {log_file}")
        else:
            logger.error(f"Unknown service: {service_name}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='IQOption Trading System Deployment Manager')
    parser.add_argument('--deploy', action='store_true', help='Deploy full system')
    parser.add_argument('--status', action='store_true', help='Show system status')
    parser.add_argument('--restart', action='store_true', help='Restart all services')
    parser.add_argument('--stop', action='store_true', help='Stop all services')
    parser.add_argument('--start', type=str, help='Start specific service')
    parser.add_argument('--logs', type=str, help='Show logs for service')
    parser.add_argument('--emergency-stop', action='store_true', help='Emergency shutdown')
    
    args = parser.parse_args()
    
    deployment = TradingSystemDeployment()
    
    if args.deploy:
        deployment.deploy_full_system()
        deployment.create_startup_script()
        deployment.show_monitoring_commands()
    
    elif args.status:
        deployment.show_system_status()
        deployment.show_access_urls()
    
    elif args.restart:
        logger.info("🔄 RESTARTING ALL SERVICES...")
        for service in deployment.services.keys():
            deployment.restart_service(service)
        deployment.show_system_status()
    
    elif args.stop:
        logger.info("🛑 STOPPING ALL SERVICES...")
        for service in reversed(list(deployment.services.keys())):
            deployment.stop_service(service)
    
    elif args.start:
        deployment.start_service(args.start)
    
    elif args.logs:
        deployment.tail_logs(args.logs)
    
    elif args.emergency_stop:
        deployment.emergency_shutdown()
    
    else:
        logger.info("🤖 IQOption Trading System Deployment Manager")
        logger.info("Use --help for available commands")
        deployment.show_system_status()

if __name__ == "__main__":
    main()