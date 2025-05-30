from fastapi import FastAPI, APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
import asyncio
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid
from iqoptionapi.stable_api import IQ_Option
import ta
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sys

# Import ML-enhanced strategies
sys.path.append('/app')
from enhanced_ml_strategies import EnhancedMLTradingStrategies, enhanced_strategies

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(title="IQOption Trading Strategy System")
api_router = APIRouter(prefix="/api")

# Global IQOption API instance
iq_api = None
connection_manager = None

# Models
class Trade(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    asset: str
    amount: float
    direction: str  # "call" or "put"
    duration: int
    strategy: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    result: Optional[str] = None
    profit: Optional[float] = None
    status: str = "pending"

class TradeRequest(BaseModel):
    asset: str
    amount: float
    strategy: str = "auto"
    auto_trade: bool = False

class MarketData(BaseModel):
    asset: str
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float

class StrategySignal(BaseModel):
    asset: str
    signal: str  # "call", "put", "hold"
    confidence: float
    strategy_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict):
        if self.active_connections:
            for connection in self.active_connections:
                try:
                    await connection.send_text(json.dumps(message))
                except:
                    await self.disconnect(connection)

# IQOption Service
class IQOptionService:
    def __init__(self):
        self.api = None
        self.connected = False
        
    async def connect(self):
        """Connect to IQOption API"""
        if not self.connected:
            try:
                self.api = IQ_Option(
                    os.environ.get('IQOPTION_EMAIL'),
                    os.environ.get('IQOPTION_PASSWORD')
                )
                check, reason = self.api.connect()
                if check:
                    # Switch to practice account for safety
                    self.api.change_balance("PRACTICE")
                    self.connected = True
                    logging.info("Successfully connected to IQOption API")
                    return True
                else:
                    logging.error(f"Failed to connect to IQOption: {reason}")
                    return False
            except Exception as e:
                logging.error(f"Error connecting to IQOption: {e}")
                return False
        return True

    async def get_balance(self):
        """Get account balance"""
        if await self.connect():
            try:
                balance = self.api.get_balance()
                return balance
            except Exception as e:
                logging.error(f"Error getting balance: {e}")
                return 0
        return 0

    async def get_candles(self, asset: str, duration: int = 60, count: int = 100):
        """Get historical candle data"""
        if await self.connect():
            try:
                end_time = time.time()
                candles = self.api.get_candles(asset, duration, count, end_time)
                
                # Convert dict format to list format for easier processing
                if candles and isinstance(candles[0], dict):
                    formatted_candles = []
                    for candle in candles:
                        formatted_candles.append([
                            candle['from'],  # timestamp
                            candle['open'],  # open
                            candle['max'],   # high
                            candle['min'],   # low
                            candle['close'], # close
                            candle['volume'] # volume
                        ])
                    return formatted_candles
                return candles
            except Exception as e:
                logging.error(f"Error getting candles for {asset}: {e}")
                return []
        return []

    async def execute_trade(self, asset: str, amount: float, direction: str, duration: int = 1):
        """Execute a trade"""
        if await self.connect():
            try:
                # Try multiple trading methods
                
                # Method 1: Traditional binary options
                check, order_id = self.api.buy(amount, asset, direction, duration)
                if check:
                    return {"success": True, "order_id": order_id, "method": "binary"}
                
                # Method 2: Digital options if binary fails
                try:
                    digital_result = self.api.buy_digital_spot(asset, amount, direction, duration)
                    if digital_result and digital_result[0]:
                        return {"success": True, "order_id": digital_result[1], "method": "digital"}
                except Exception as digital_error:
                    logging.info(f"Digital options not available: {digital_error}")
                
                # Method 3: Try with different durations
                for alt_duration in [2, 5, 15]:
                    try:
                        check, alt_order_id = self.api.buy(amount, asset, direction, alt_duration)
                        if check:
                            return {"success": True, "order_id": alt_order_id, "method": f"binary_{alt_duration}m"}
                    except:
                        continue
                
                return {"success": False, "error": str(order_id)}
            except Exception as e:
                logging.error(f"Error executing trade: {e}")
                return {"success": False, "error": str(e)}
        return {"success": False, "error": "Not connected"}

    async def get_available_assets(self):
        """Get available trading assets"""
        if await self.connect():
            try:
                # Get all open times to find available assets
                all_assets = self.api.get_all_open_time()
                available_assets = []
                
                current_time = time.time()
                for asset, times in all_assets.items():
                    if times and any(t['open'] for t in times.values()):
                        available_assets.append(asset)
                
                return available_assets[:10]  # Return top 10 for demo
            except Exception as e:
                logging.error(f"Error getting assets: {e}")
                return ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"]
        return ["EURUSD", "GBPUSD", "USDJPY", "AUDCAD"]

# Trading Strategies
class TradingStrategies:
    def __init__(self):
        self.ml_model = None
        self.trained = False

    def calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        return ta.momentum.RSIIndicator(data['close'], window=period).rsi()

    def calculate_macd(self, data: pd.DataFrame) -> Dict:
        """Calculate MACD"""
        macd = ta.trend.MACD(data['close'])
        return {
            'macd': macd.macd(),
            'signal': macd.macd_signal(),
            'histogram': macd.macd_diff()
        }

    def calculate_bollinger_bands(self, data: pd.DataFrame, period: int = 20) -> Dict:
        """Calculate Bollinger Bands"""
        bb = ta.volatility.BollingerBands(data['close'], window=period)
        return {
            'upper': bb.bollinger_hband(),
            'middle': bb.bollinger_mavg(),
            'lower': bb.bollinger_lband()
        }

    def calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        
        resistance = highs.iloc[-1]
        support = lows.iloc[-1]
        
        return {'support': support, 'resistance': resistance}

    def rsi_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """RSI-based strategy with high accuracy"""
        if len(data) < 30:
            return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="RSI")
        
        rsi = self.calculate_rsi(data)
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # High confidence RSI signals
        if current_rsi < 20 and prev_rsi > current_rsi:  # Strong oversold
            confidence = min(0.85, (20 - current_rsi) / 20 + 0.6)
            return StrategySignal(asset="", signal="call", confidence=confidence, strategy_name="RSI_Oversold")
        elif current_rsi > 80 and prev_rsi < current_rsi:  # Strong overbought
            confidence = min(0.85, (current_rsi - 80) / 20 + 0.6)
            return StrategySignal(asset="", signal="put", confidence=confidence, strategy_name="RSI_Overbought")
        
        return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="RSI")

    def macd_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """MACD-based strategy"""
        if len(data) < 30:
            return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="MACD")
        
        macd_data = self.calculate_macd(data)
        macd = macd_data['macd'].iloc[-1]
        signal = macd_data['signal'].iloc[-1]
        prev_macd = macd_data['macd'].iloc[-2]
        prev_signal = macd_data['signal'].iloc[-2]
        
        # MACD crossover signals
        if prev_macd <= prev_signal and macd > signal:  # Bullish crossover
            confidence = 0.75
            return StrategySignal(asset="", signal="call", confidence=confidence, strategy_name="MACD_Bullish")
        elif prev_macd >= prev_signal and macd < signal:  # Bearish crossover
            confidence = 0.75
            return StrategySignal(asset="", signal="put", confidence=confidence, strategy_name="MACD_Bearish")
        
        return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="MACD")

    def bollinger_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Bollinger Bands strategy"""
        if len(data) < 25:
            return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Bollinger")
        
        bb = self.calculate_bollinger_bands(data)
        current_price = data['close'].iloc[-1]
        upper_band = bb['upper'].iloc[-1]
        lower_band = bb['lower'].iloc[-1]
        middle_band = bb['middle'].iloc[-1]
        
        # High confidence Bollinger signals
        if current_price <= lower_band:  # Price at lower band - oversold
            confidence = 0.80
            return StrategySignal(asset="", signal="call", confidence=confidence, strategy_name="Bollinger_Oversold")
        elif current_price >= upper_band:  # Price at upper band - overbought
            confidence = 0.80
            return StrategySignal(asset="", signal="put", confidence=confidence, strategy_name="Bollinger_Overbought")
        
        return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Bollinger")

    def trend_following_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Advanced trend following with multiple confirmations"""
        if len(data) < 50:
            return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Trend")
        
        # Multiple moving averages
        ema_20 = ta.trend.EMAIndicator(data['close'], window=20).ema_indicator()
        ema_50 = ta.trend.EMAIndicator(data['close'], window=50).ema_indicator()
        sma_100 = ta.trend.SMAIndicator(data['close'], window=100).sma_indicator()
        
        current_price = data['close'].iloc[-1]
        ema20_current = ema_20.iloc[-1]
        ema50_current = ema_50.iloc[-1]
        sma100_current = sma_100.iloc[-1]
        
        # Strong uptrend confirmation
        if (current_price > ema20_current > ema50_current > sma100_current and
            ema20_current > ema_20.iloc[-2]):  # All MAs aligned and EMA20 rising
            confidence = 0.82
            return StrategySignal(asset="", signal="call", confidence=confidence, strategy_name="Strong_Uptrend")
        
        # Strong downtrend confirmation
        if (current_price < ema20_current < ema50_current < sma100_current and
            ema20_current < ema_20.iloc[-2]):  # All MAs aligned and EMA20 falling
            confidence = 0.82
            return StrategySignal(asset="", signal="put", confidence=confidence, strategy_name="Strong_Downtrend")
        
        return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Trend")

    def combined_strategy(self, data: pd.DataFrame) -> StrategySignal:
        """Combine multiple strategies for higher accuracy - OPTIMIZED VERSION"""
        if len(data) < 50:
            return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Combined")
        
        # Get signals from profitable strategies only (based on backtesting)
        bollinger_signal = self.bollinger_strategy(data)
        macd_signal = self.macd_strategy(data)
        
        # Only use trend if it has reasonable confidence
        trend_signal = self.trend_following_strategy(data)
        
        # Weighted approach based on backtesting performance
        signals = []
        
        # Bollinger Bands (best performer): weight 0.4
        if bollinger_signal.confidence > 0.5:
            signals.extend([bollinger_signal] * 4)
        
        # MACD (second best): weight 0.3  
        if macd_signal.confidence > 0.5:
            signals.extend([macd_signal] * 3)
        
        # Trend (if confident): weight 0.3
        if trend_signal.confidence > 0.6:
            signals.extend([trend_signal] * 3)
        
        if not signals:
            return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Combined_Optimized")
        
        # Count weighted votes
        call_votes = sum(1 for s in signals if s.signal == "call")
        put_votes = sum(1 for s in signals if s.signal == "put")
        total_votes = len(signals)
        
        # Calculate confidence based on consensus and individual confidences
        if call_votes > put_votes and call_votes >= total_votes * 0.6:  # 60% consensus required
            avg_confidence = sum(s.confidence for s in signals if s.signal == "call") / call_votes
            final_confidence = min(0.9, avg_confidence * (call_votes / total_votes))
            return StrategySignal(asset="", signal="call", confidence=final_confidence, strategy_name="Combined_Optimized")
        elif put_votes > call_votes and put_votes >= total_votes * 0.6:
            avg_confidence = sum(s.confidence for s in signals if s.signal == "put") / put_votes
            final_confidence = min(0.9, avg_confidence * (put_votes / total_votes))
            return StrategySignal(asset="", signal="put", confidence=final_confidence, strategy_name="Combined_Optimized")
        
        return StrategySignal(asset="", signal="hold", confidence=0.0, strategy_name="Combined_Optimized")

    def backtested_optimized_strategy(self, asset: str, data: pd.DataFrame) -> StrategySignal:
        """Use only the most profitable strategy for each asset based on backtesting"""
        try:
            # Based on backtesting results, use best strategy for each asset
            if asset == "EURUSD":
                # Best: Bollinger (57.4% win rate)
                signal = self.bollinger_strategy(data)
                if signal.confidence < 0.8:  # Increase threshold for proven strategy
                    # Fallback to MACD (57.1% win rate)
                    macd_signal = self.macd_strategy(data)
                    if macd_signal.confidence > signal.confidence:
                        signal = macd_signal
            elif asset == "GBPUSD":
                # Best: MACD (50.0% win rate, though barely profitable)
                signal = self.macd_strategy(data)
                if signal.confidence < 0.8:  # Very high threshold for this asset
                    signal = StrategySignal(asset=asset, signal="hold", confidence=0.0, strategy_name="Optimized_Conservative")
            elif asset == "USDJPY":
                # Best: Bollinger (50.0% win rate, though unprofitable)
                signal = self.bollinger_strategy(data)
                if signal.confidence < 0.8:  # Very high threshold
                    signal = StrategySignal(asset=asset, signal="hold", confidence=0.0, strategy_name="Optimized_Conservative")
            elif asset == "AUDCAD":
                # Best: RSI (50.0% win rate) but very few trades
                signal = self.rsi_strategy(data)
                if signal.confidence < 0.8:  # Very high threshold
                    signal = StrategySignal(asset=asset, signal="hold", confidence=0.0, strategy_name="Optimized_Conservative")
            else:
                # Default to most conservative approach
                signal = StrategySignal(asset=asset, signal="hold", confidence=0.0, strategy_name="Optimized_Unknown_Asset")
            
            signal.asset = asset
            return signal
        except Exception as e:
            logging.error(f"Error in optimized strategy for {asset}: {e}")
            return StrategySignal(asset=asset, signal="hold", confidence=0.0, strategy_name="Optimized_Error")

    def get_strategy_signal(self, asset: str, data: pd.DataFrame, strategy: str = "combined") -> StrategySignal:
        """Get trading signal based on selected strategy"""
        try:
            if strategy == "rsi":
                signal = self.rsi_strategy(data)
            elif strategy == "macd":
                signal = self.macd_strategy(data)
            elif strategy == "bollinger":
                signal = self.bollinger_strategy(data)
            elif strategy == "trend":
                signal = self.trend_following_strategy(data)
            elif strategy == "combined":
                signal = self.combined_strategy(data)
            elif strategy == "optimized":
                signal = self.backtested_optimized_strategy(asset, data)
            else:
                signal = self.combined_strategy(data)  # Default to combined
            
            signal.asset = asset
            return signal
        except Exception as e:
            logging.error(f"Error in strategy {strategy}: {e}")
            return StrategySignal(asset=asset, signal="hold", confidence=0.0, strategy_name=strategy)

# Global instances
iq_service = IQOptionService()
strategies = TradingStrategies()
connection_manager = ConnectionManager()

# API Routes
@api_router.get("/")
async def root():
    return {"message": "IQOption Trading Strategy System API"}

@api_router.get("/connection-status")
async def get_connection_status():
    """Check IQOption API connection status"""
    connected = await iq_service.connect()
    balance = await iq_service.get_balance() if connected else 0
    return {
        "connected": connected,
        "balance": balance,
        "mode": "PRACTICE"
    }

@api_router.get("/assets")
async def get_assets():
    """Get available trading assets"""
    assets = await iq_service.get_available_assets()
    return {"assets": assets}

@api_router.get("/market-data/{asset}")
async def get_market_data(asset: str, count: int = 100):
    """Get historical market data for an asset"""
    candles = await iq_service.get_candles(asset, 60, count)
    
    if candles:
        df = pd.DataFrame(candles)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        return {
            "asset": asset,
            "data": df.to_dict('records'),
            "count": len(candles)
        }
    return {"asset": asset, "data": [], "count": 0}

@api_router.get("/strategy-signal/{asset}")
async def get_strategy_signal(asset: str, strategy: str = "ml_enhanced"):
    """Get trading signal for an asset using ML-enhanced strategies"""
    candles = await iq_service.get_candles(asset, 60, 200)
    
    if candles:
        df = pd.DataFrame(candles)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        try:
            if strategy == "ml_enhanced":
                signal = await enhanced_strategies.ml_enhanced_strategy(asset, df)
                return {
                    "asset": signal.asset,
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "ml_probability": signal.ml_probability,
                    "strategy_name": signal.strategy_name,
                    "feature_importance": signal.feature_importance,
                    "timestamp": signal.timestamp.isoformat()
                }
            elif strategy == "adaptive_ml":
                signal = await enhanced_strategies.adaptive_ml_strategy(asset, df)
                return {
                    "asset": signal.asset,
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "ml_probability": signal.ml_probability,
                    "strategy_name": signal.strategy_name,
                    "feature_importance": signal.feature_importance,
                    "timestamp": signal.timestamp.isoformat()
                }
            else:
                # Fallback to traditional strategies
                signal = strategies.get_strategy_signal(asset, df, strategy)
                return signal.dict()
        except Exception as e:
            logging.error(f"Error in ML strategy for {asset}: {e}")
            # Fallback to traditional
            signal = strategies.get_strategy_signal(asset, df, "bollinger")
            return signal.dict()
    
    return {"asset": asset, "signal": "hold", "confidence": 0.0, "strategy_name": strategy}

@api_router.post("/execute-trade")
async def execute_trade(trade_request: TradeRequest):
    """Execute a trade based on strategy signal"""
    try:
        # Get current signal
        candles = await iq_service.get_candles(trade_request.asset, 60, 200)
        
        if not candles:
            raise HTTPException(status_code=400, detail="Unable to get market data")
        
        df = pd.DataFrame(candles)
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Get strategy signal
        if trade_request.strategy in ["adaptive_ml", "ml_enhanced"]:
            try:
                if trade_request.strategy == "adaptive_ml":
                    signal = await enhanced_strategies.adaptive_ml_strategy(trade_request.asset, df)
                else:
                    signal = await enhanced_strategies.ml_enhanced_strategy(trade_request.asset, df)
                
                signal_dict = {
                    "signal": signal.signal,
                    "confidence": signal.confidence,
                    "strategy_name": signal.strategy_name
                }
            except Exception as e:
                logger.error(f"ML strategy error: {e}")
                # Fallback to traditional
                signal = strategies.get_strategy_signal(trade_request.asset, df, "bollinger")
                signal_dict = signal.dict()
        else:
            signal = strategies.get_strategy_signal(trade_request.asset, df, trade_request.strategy)
            signal_dict = signal.dict()
        
        if signal_dict["signal"] == "hold" or signal_dict["confidence"] < 0.65:
            return {
                "success": False,
                "message": f"No high-confidence signal. Signal: {signal_dict['signal']}, Confidence: {signal_dict['confidence']:.2f}"
            }
        
        # Execute trade if auto_trade is enabled
        if trade_request.auto_trade:
            result = await iq_service.execute_trade(
                trade_request.asset,
                trade_request.amount,
                signal_dict["signal"],
                1  # 1 minute duration
            )
            
            if result["success"]:
                # Store trade in database
                trade = Trade(
                    asset=trade_request.asset,
                    amount=trade_request.amount,
                    direction=signal_dict["signal"],
                    duration=1,
                    strategy=signal_dict["strategy_name"],
                    confidence=signal_dict["confidence"],
                    status="executed"
                )
                
                await db.trades.insert_one(trade.dict())
                
                # Broadcast to WebSocket clients
                await connection_manager.broadcast({
                    "type": "trade_executed",
                    "trade": trade.dict()
                })
                
                return {
                    "success": True,
                    "trade": trade.dict(),
                    "order_id": result.get("order_id")
                }
            else:
                return {
                    "success": False,
                    "message": f"Trade execution failed: {result.get('error')}"
                }
        else:
            return {
                "success": True,
                "signal": signal_dict,
                "message": "Signal generated. Set auto_trade=true to execute."
            }
            
    except Exception as e:
        logging.error(f"Error in execute_trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    trades = await db.trades.find().sort("timestamp", -1).limit(limit).to_list(limit)
    return {"trades": trades}

@api_router.get("/performance")
async def get_performance():
    """Get trading performance statistics"""
    trades = await db.trades.find({"status": {"$in": ["executed", "completed"]}}).to_list(1000)
    
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "total_profit": 0,
            "average_confidence": 0
        }
    
    total_trades = len(trades)
    winning_trades = len([t for t in trades if t.get("profit", 0) > 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    total_profit = sum(t.get("profit", 0) for t in trades)
    avg_confidence = sum(t.get("confidence", 0) for t in trades) / total_trades
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "total_profit": total_profit,
        "average_confidence": avg_confidence,
        "winning_trades": winning_trades,
        "losing_trades": total_trades - winning_trades
    }

# WebSocket endpoint for real-time data
@api_router.websocket("/ws/market/{asset}")
async def websocket_market_feed(websocket: WebSocket, asset: str):
    await connection_manager.connect(websocket)
    
    try:
        while True:
            # Get latest market data
            candles = await iq_service.get_candles(asset, 60, 1)
            if candles:
                latest_candle = candles[-1]
                
                # Get current strategy signal
                historical_candles = await iq_service.get_candles(asset, 60, 100)
                if historical_candles:
                    df = pd.DataFrame(historical_candles)
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    signal = strategies.get_strategy_signal(asset, df, "combined")
                    
                    await websocket.send_text(json.dumps({
                        "type": "market_data",
                        "asset": asset,
                        "candle": {
                            "timestamp": latest_candle[0],
                            "open": latest_candle[1],
                            "high": latest_candle[2],
                            "low": latest_candle[3],
                            "close": latest_candle[4],
                            "volume": latest_candle[5]
                        },
                        "signal": signal.dict(),
                        "timestamp": time.time()
                    }))
            
            await asyncio.sleep(5)  # Update every 5 seconds
            
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {e}")
        connection_manager.disconnect(websocket)

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    logger.info("Starting IQOption Trading System...")
    await iq_service.connect()
    
    # Initialize ML models
    logger.info("🤖 Initializing ML-enhanced strategies...")
    try:
        await enhanced_strategies.initialize_ml_models("EURUSD")
        logger.info("✅ ML models ready for trading")
    except Exception as e:
        logger.error(f"ML initialization error: {e}")
        logger.info("⚠️ Falling back to traditional strategies")

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
