#!/bin/bash
# IQOption Trading System Startup Script

echo "🚀 Starting IQOption Trading System..."

# Change to app directory
cd /app

# Start the deployment
python deployment_manager.py --deploy

echo "✅ Startup complete! Check logs for status."
