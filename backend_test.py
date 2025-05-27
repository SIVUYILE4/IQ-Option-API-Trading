import requests
import sys
import time
from datetime import datetime

class IQOptionAPITester:
    def __init__(self, base_url):
        self.base_url = base_url
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def run_test(self, name, method, endpoint, expected_status, data=None):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        self.tests_run += 1
        print(f"\n🔍 Testing {name}...")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers)

            success = response.status_code == expected_status
            if success:
                self.tests_passed += 1
                print(f"✅ Passed - Status: {response.status_code}")
                result = {
                    "name": name,
                    "status": "PASSED",
                    "response": response.json() if response.text else {}
                }
            else:
                print(f"❌ Failed - Expected {expected_status}, got {response.status_code}")
                result = {
                    "name": name,
                    "status": "FAILED",
                    "expected": expected_status,
                    "actual": response.status_code,
                    "response": response.json() if response.text else {}
                }
            
            self.test_results.append(result)
            return success, response.json() if response.text and success else {}

        except Exception as e:
            print(f"❌ Failed - Error: {str(e)}")
            self.test_results.append({
                "name": name,
                "status": "ERROR",
                "error": str(e)
            })
            return False, {}

    def test_connection_status(self):
        """Test connection to IQOption API"""
        return self.run_test(
            "Connection Status",
            "GET",
            "api/connection-status",
            200
        )

    def test_available_assets(self):
        """Test available assets endpoint"""
        return self.run_test(
            "Available Assets",
            "GET",
            "api/assets",
            200
        )

    def test_market_data(self, asset="EURUSD"):
        """Test market data retrieval"""
        return self.run_test(
            f"Market Data for {asset}",
            "GET",
            f"api/market-data/{asset}",
            200
        )

    def test_strategy_signal(self, asset="EURUSD", strategy="combined"):
        """Test strategy signal endpoint"""
        return self.run_test(
            f"Strategy Signal for {asset} using {strategy}",
            "GET",
            f"api/strategy-signal/{asset}?strategy={strategy}",
            200
        )

    def test_execute_trade(self, asset="EURUSD", amount=1.0, strategy="combined", auto_trade=False):
        """Test trade execution endpoint"""
        return self.run_test(
            f"Execute Trade for {asset} using {strategy} (auto_trade={auto_trade})",
            "POST",
            "api/execute-trade",
            200,
            data={
                "asset": asset,
                "amount": amount,
                "strategy": strategy,
                "auto_trade": auto_trade
            }
        )

    def test_performance_stats(self):
        """Test performance statistics endpoint"""
        return self.run_test(
            "Performance Statistics",
            "GET",
            "api/performance",
            200
        )

    def test_recent_trades(self):
        """Test recent trades endpoint"""
        return self.run_test(
            "Recent Trades",
            "GET",
            "api/trades",
            200
        )

    def print_summary(self):
        """Print test summary"""
        print("\n" + "="*50)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        print("="*50)
        
        for result in self.test_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            print(f"{status_icon} {result['name']}: {result['status']}")
            
            if result["status"] == "FAILED":
                print(f"   Expected: {result['expected']}, Got: {result['actual']}")
            elif result["status"] == "ERROR":
                print(f"   Error: {result['error']}")
        
        print("="*50)
        return self.tests_passed == self.tests_run

def main():
    # Get the backend URL from the frontend .env file
    backend_url = "https://77cf5443-9361-4fe0-9428-ed529aa75ac1.preview.emergentagent.com"
    
    print(f"Testing IQOption Trading Strategy System API at: {backend_url}")
    print("="*50)
    
    # Setup tester
    tester = IQOptionAPITester(backend_url)
    
    # Run tests
    tester.test_connection_status()
    tester.test_available_assets()
    
    # Test market data for EURUSD
    success, market_data = tester.test_market_data("EURUSD")
    
    # Test different strategies
    strategies = ["combined", "rsi", "macd", "bollinger", "trend"]
    for strategy in strategies:
        tester.test_strategy_signal("EURUSD", strategy)
    
    # Test trade execution with auto_trade=false
    tester.test_execute_trade(auto_trade=False)
    
    # Test performance stats
    tester.test_performance_stats()
    
    # Test recent trades
    tester.test_recent_trades()
    
    # Print summary
    success = tester.print_summary()
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
