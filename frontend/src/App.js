import React, { useState, useEffect } from 'react';
import axios from 'axios';
import useWebSocket from 'react-use-websocket';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign, Target, AlertCircle, Play, Pause } from 'lucide-react';
import './App.css';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
const WS_URL = BACKEND_URL.replace('https://', 'wss://').replace('http://', 'ws://');

function App() {
  const [selectedAsset, setSelectedAsset] = useState('EURUSD');
  const [assets, setAssets] = useState([]);
  const [connectionStatus, setConnectionStatus] = useState({});
  const [marketData, setMarketData] = useState([]);
  const [currentSignal, setCurrentSignal] = useState(null);
  const [trades, setTrades] = useState([]);
  const [performance, setPerformance] = useState({});
  const [autoTrading, setAutoTrading] = useState(false);
  const [tradeAmount, setTradeAmount] = useState(1);
  const [selectedStrategy, setSelectedStrategy] = useState('combined');
  const [loading, setLoading] = useState(false);

  // WebSocket connection for real-time data
  const { lastMessage, connectionStatus: wsStatus } = useWebSocket(
    `${WS_URL}/api/ws/market/${selectedAsset}`,
    {
      onOpen: () => console.log('WebSocket Connected'),
      onClose: () => console.log('WebSocket Disconnected'),
      shouldReconnect: () => true,
    }
  );

  // Process WebSocket messages
  useEffect(() => {
    if (lastMessage !== null) {
      try {
        const data = JSON.parse(lastMessage.data);
        if (data.type === 'market_data') {
          setMarketData(prev => {
            const newData = [...prev, {
              ...data.candle,
              time: new Date(data.candle.timestamp * 1000).toLocaleTimeString()
            }].slice(-50); // Keep last 50 candles
            return newData;
          });
          setCurrentSignal(data.signal);
        }
      } catch (error) {
        console.error('Error parsing WebSocket data:', error);
      }
    }
  }, [lastMessage]);

  // Fetch initial data
  useEffect(() => {
    fetchConnectionStatus();
    fetchAssets();
    fetchTrades();
    fetchPerformance();
  }, []);

  // Fetch market data when asset changes
  useEffect(() => {
    if (selectedAsset) {
      fetchMarketData();
      fetchCurrentSignal();
    }
  }, [selectedAsset]);

  const fetchConnectionStatus = async () => {
    try {
      const response = await axios.get(`${API}/connection-status`);
      setConnectionStatus(response.data);
    } catch (error) {
      console.error('Error fetching connection status:', error);
    }
  };

  const fetchAssets = async () => {
    try {
      const response = await axios.get(`${API}/assets`);
      setAssets(response.data.assets);
    } catch (error) {
      console.error('Error fetching assets:', error);
    }
  };

  const fetchMarketData = async () => {
    try {
      const response = await axios.get(`${API}/market-data/${selectedAsset}?count=50`);
      const formattedData = response.data.data.map(candle => ({
        ...candle,
        time: new Date(candle.timestamp * 1000).toLocaleTimeString()
      }));
      setMarketData(formattedData);
    } catch (error) {
      console.error('Error fetching market data:', error);
    }
  };

  const fetchCurrentSignal = async () => {
    try {
      const response = await axios.get(`${API}/strategy-signal/${selectedAsset}?strategy=${selectedStrategy}`);
      setCurrentSignal(response.data);
    } catch (error) {
      console.error('Error fetching signal:', error);
    }
  };

  const fetchTrades = async () => {
    try {
      const response = await axios.get(`${API}/trades`);
      setTrades(response.data.trades);
    } catch (error) {
      console.error('Error fetching trades:', error);
    }
  };

  const fetchPerformance = async () => {
    try {
      const response = await axios.get(`${API}/performance`);
      setPerformance(response.data);
    } catch (error) {
      console.error('Error fetching performance:', error);
    }
  };

  const executeTrade = async () => {
    if (!currentSignal || currentSignal.signal === 'hold' || currentSignal.confidence < 0.7) {
      alert('No high-confidence signal available. Wait for a better opportunity.');
      return;
    }

    setLoading(true);
    try {
      const response = await axios.post(`${API}/execute-trade`, {
        asset: selectedAsset,
        amount: tradeAmount,
        strategy: selectedStrategy,
        auto_trade: true
      });

      if (response.data.success) {
        alert(`Trade executed successfully! ${response.data.trade?.direction} ${selectedAsset}`);
        fetchTrades();
        fetchPerformance();
      } else {
        alert(`Trade failed: ${response.data.message}`);
      }
    } catch (error) {
      alert(`Error executing trade: ${error.response?.data?.detail || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const getSignalColor = (signal) => {
    if (signal === 'call') return 'text-green-600';
    if (signal === 'put') return 'text-red-600';
    return 'text-gray-600';
  };

  const getSignalIcon = (signal) => {
    if (signal === 'call') return <TrendingUp className="w-5 h-5" />;
    if (signal === 'put') return <TrendingDown className="w-5 h-5" />;
    return <Activity className="w-5 h-5" />;
  };

  const formatConfidence = (confidence) => {
    return `${(confidence * 100).toFixed(1)}%`;
  };

  const strategies = [
    { value: 'adaptive_ml', label: '🤖 Adaptive ML (Highest Accuracy)' },
    { value: 'ml_enhanced', label: '🧠 ML Enhanced Strategy' },
    { value: 'combined', label: 'Combined Traditional Strategy' },
    { value: 'bollinger', label: 'Bollinger Bands' },
    { value: 'macd', label: 'MACD Strategy' },
    { value: 'rsi', label: 'RSI Strategy' },
    { value: 'trend', label: 'Trend Following' }
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center space-x-4">
              <h1 className="text-2xl font-bold text-gray-900">IQOption Strategy Bot</h1>
              <div className={`flex items-center space-x-2 px-3 py-1 rounded-full ${
                connectionStatus.connected ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  connectionStatus.connected ? 'bg-green-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-medium">
                  {connectionStatus.connected ? 'Connected' : 'Disconnected'}
                </span>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-gray-600">Practice Balance</div>
              <div className="text-lg font-semibold text-gray-900">
                ${connectionStatus.balance?.toFixed(2) || '0.00'}
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Trading Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-8">
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Trading Controls</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Asset</label>
                  <select
                    value={selectedAsset}
                    onChange={(e) => setSelectedAsset(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {assets.map(asset => (
                      <option key={asset} value={asset}>{asset}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Strategy</label>
                  <select
                    value={selectedStrategy}
                    onChange={(e) => setSelectedStrategy(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {strategies.map(strategy => (
                      <option key={strategy.value} value={strategy.value}>{strategy.label}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Amount</label>
                  <input
                    type="number"
                    value={tradeAmount}
                    onChange={(e) => setTradeAmount(parseFloat(e.target.value))}
                    min="1"
                    step="0.1"
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <button
                  onClick={executeTrade}
                  disabled={loading || !connectionStatus.connected || !currentSignal || currentSignal.signal === 'hold'}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {loading ? (
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      <span>Execute Trade</span>
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Current Signal */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Signal</h3>
              
              {currentSignal ? (
                <div className="space-y-3">
                  <div className={`flex items-center space-x-2 ${getSignalColor(currentSignal.signal)}`}>
                    {getSignalIcon(currentSignal.signal)}
                    <span className="font-semibold capitalize">{currentSignal.signal}</span>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600">Confidence</div>
                    <div className="text-2xl font-bold">{formatConfidence(currentSignal.confidence)}</div>
                  </div>
                  
                  <div>
                    <div className="text-sm text-gray-600">Strategy</div>
                    <div className="text-sm font-medium">{currentSignal.strategy_name}</div>
                  </div>

                  {currentSignal.confidence >= 0.7 && (
                    <div className="bg-green-50 border border-green-200 rounded-md p-3">
                      <div className="flex items-center space-x-2 text-green-800">
                        <Target className="w-4 h-4" />
                        <span className="text-sm font-medium">High Confidence Signal</span>
                      </div>
                    </div>
                  )}

                  {currentSignal.confidence < 0.7 && currentSignal.signal !== 'hold' && (
                    <div className="bg-yellow-50 border border-yellow-200 rounded-md p-3">
                      <div className="flex items-center space-x-2 text-yellow-800">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm font-medium">Low Confidence - Wait</span>
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="text-gray-500">Loading signal...</div>
              )}
            </div>
          </div>

          {/* Performance Stats */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Stats</h3>
              
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{performance.total_trades || 0}</div>
                  <div className="text-sm text-gray-600">Total Trades</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{performance.win_rate?.toFixed(1) || 0}%</div>
                  <div className="text-sm text-gray-600">Win Rate</div>
                </div>
                
                <div className="text-center">
                  <div className={`text-2xl font-bold ${performance.total_profit >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${performance.total_profit?.toFixed(2) || '0.00'}
                  </div>
                  <div className="text-sm text-gray-600">Total Profit</div>
                </div>
                
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {performance.average_confidence ? formatConfidence(performance.average_confidence) : '0%'}
                  </div>
                  <div className="text-sm text-gray-600">Avg Confidence</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Market Chart */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          <div className="bg-white rounded-lg shadow p-6">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-gray-900">{selectedAsset} Price Chart</h3>
              <div className={`px-2 py-1 rounded text-sm ${
                wsStatus === 'Open' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {wsStatus === 'Open' ? 'Live' : 'Disconnected'}
              </div>
            </div>
            
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={marketData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis domain={['dataMin - 0.001', 'dataMax + 0.001']} />
                <Tooltip />
                <Line type="monotone" dataKey="close" stroke="#2563eb" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Recent Trades */}
          <div className="bg-white rounded-lg shadow p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Trades</h3>
            
            <div className="space-y-3 max-h-80 overflow-y-auto">
              {trades.slice(0, 10).map((trade, index) => (
                <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                  <div>
                    <div className="font-medium">{trade.asset}</div>
                    <div className="text-sm text-gray-600">{trade.strategy}</div>
                  </div>
                  <div className="text-right">
                    <div className={`font-medium ${getSignalColor(trade.direction)}`}>
                      {trade.direction.toUpperCase()}
                    </div>
                    <div className="text-sm text-gray-600">${trade.amount}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium">{formatConfidence(trade.confidence)}</div>
                    <div className="text-xs text-gray-500">{trade.status}</div>
                  </div>
                </div>
              ))}
              
              {trades.length === 0 && (
                <div className="text-center text-gray-500 py-8">
                  No trades yet. Execute your first trade!
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Strategy Information */}
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Strategy Information</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="p-4 bg-blue-50 rounded-lg">
              <h4 className="font-semibold text-blue-900 mb-2">Combined Strategy</h4>
              <p className="text-sm text-blue-800">
                Uses multiple indicators (RSI, MACD, Bollinger Bands, Trend) with majority voting for highest accuracy signals.
              </p>
            </div>
            
            <div className="p-4 bg-green-50 rounded-lg">
              <h4 className="font-semibold text-green-900 mb-2">High Confidence Threshold</h4>
              <p className="text-sm text-green-800">
                Only trades with 70%+ confidence are recommended. Combined strategy requires 3/4 indicators to agree.
              </p>
            </div>
            
            <div className="p-4 bg-purple-50 rounded-lg">
              <h4 className="font-semibold text-purple-900 mb-2">Risk Management</h4>
              <p className="text-sm text-purple-800">
                Practice account, small trade amounts, and high confidence requirements protect your capital.
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
