import logging
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OANDAClient:
    """OANDA v20 API client."""
    
    def __init__(self, access_token: str, account_id: str, env: str = "practice"):
        self.access_token = access_token
        self.account_id = account_id
        self.env = env
        
        if env == "practice":
            self.base_url = "https://api-fxpractice.oanda. com/v3"
        else:
            self.base_url = "https://api-fxlive.oanda.com/v3"
        
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "Accept-Datetime-Format": "Unix"
        }
        
        logger.info(f"OANDAClient initialized:  {env}")
    
    def get_candles(self, instrument: str, granularity: str, count: int = 100,
                   from_time: Optional[int] = None, to_time:  Optional[int] = None,
                   price:  str = "M") -> Dict: 
        """Fetch candles from OANDA."""
        
        url = f"{self.base_url}/instruments/{instrument}/candles"
        
        params = {
            "granularity": granularity,
            "count": count,
            "price": price,
        }
        
        if from_time:
            params["from"] = from_time
        if to_time:
            params["to"] = to_time
        
        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response. raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching candles:  {e}")
            raise
    
    def create_order(self, instrument: str, units: int, order_type: str = "MARKET",
                    stop_loss_price: Optional[float] = None,
                    take_profit_price: Optional[float] = None,
                    comment: str = "") -> Dict:
        """Create an order on OANDA."""
        
        url = f"{self.base_url}/accounts/{self.account_id}/orders"
        
        data = {
            "order": {
                "instrument": instrument,
                "units": units,
                "type": order_type,
                "comment": comment,
            }
        }
        
        if stop_loss_price: 
            data["order"]["stopLossOnFill"] = {"price": str(stop_loss_price)}
        if take_profit_price: 
            data["order"]["takeProfitOnFill"] = {"price": str(take_profit_price)}
        
        try: 
            response = requests.post(url, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions. RequestException as e:
            logger.error(f"Error creating order: {e}")
            raise
    
    def get_account_summary(self) -> Dict:
        """Get account summary."""
        
        url = f"{self.base_url}/accounts/{self.account_id}/summary"
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests. exceptions.RequestException as e:
            logger.error(f"Error fetching account:  {e}")
            raise
    
    def get_open_positions(self) -> Dict:
        """Get all open positions."""
        
        url = f"{self.base_url}/accounts/{self.account_id}/openPositions"
        
        try: 
            response = requests.get(url, headers=self.headers, timeout=10)
            response. raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching positions: {e}")
            raise
    
    def close_trade(self, trade_id: str, units: int) -> Dict:
        """Close a trade."""
        
        url = f"{self.base_url}/accounts/{self.account_id}/trades/{trade_id}/close"
        
        data = {"units": units}
        
        try:
            response = requests. put(url, headers=self.headers, json=data, timeout=10)
            response.raise_for_status()
            return response. json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error closing trade: {e}")
            raise

client = OANDAClient("", "")