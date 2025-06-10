"""
Financial Data Services Module for downloading and processing financial data from EODHD API.
This module provides classes to interact with stock data, fundamentals, and information on indices like the S&P 500.
"""

import os
import requests
import pandas as pd
import configparser
from datetime import datetime
from bs4 import BeautifulSoup
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any
from dotenv import load_dotenv


def get_api_key():
    """
    Get API key from environment file or config file
    
    Returns:
        str: EODHD API key
    """
    # First try to get from .env file
    load_dotenv()
    api_key = os.getenv('EODHD_API_KEY')
    
    # If not found, try from config.ini
    if not api_key:
        env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
        config = configparser.ConfigParser()
        config.read(os.path.join(env_dir, 'config.ini'))
        api_key = config['API_KEYS']['MY_API_KEY']
    
    return api_key


class APIConfig:
    """Configuration class for EODHD API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API configuration
        
        Args:
            api_key: EODHD API key (optional, will use environment/config if None)
        """
        self.api_key = api_key if api_key else get_api_key()
        self.base_url = "https://eodhd.com/api"
        self.fmt = "json"
    
    @property
    def auth_params(self) -> Dict[str, str]:
        """Return authentication parameters for API requests"""
        return {"api_token": self.api_key, "fmt": self.fmt}


class EODHDClient:
    """Client for interacting with EODHD API"""
    
    def __init__(self, config: Optional[APIConfig] = None):
        """
        Initialize API client
        
        Args:
            config: API configuration (optional, will create a new one if None)
        """
        self.config = config if config else APIConfig()
    
    def get(self, endpoint: str, params: Optional[Dict[str, str]] = None) -> Any:
        """
        Perform a GET request to the API
        
        Args:
            endpoint: API endpoint
            params: Additional request parameters
            
        Returns:
            Data returned by the API
            
        Raises:
            Exception: If API returns error status code
        """
        url = f"{self.config.base_url}/{endpoint}"
        request_params = self.config.auth_params.copy()
        
        if params:
            request_params.update(params)
            
        response = requests.get(url, params=request_params)
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
            
        return response.json()


class ExchangeData:
    """Class for managing exchange-related data"""
    
    def __init__(self, client: Optional[EODHDClient] = None):
        """
        Initialize exchange data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else EODHDClient()
    
    def get_supported_exchanges(self) -> pd.DataFrame:
        """
        Retrieve list of supported exchanges
        
        Returns:
            DataFrame containing supported exchanges
        """
        data = self.client.get(endpoint= "exchanges-list/")
        return pd.DataFrame(data)
    
    def get_tickers_from_exchange(self, exchange_code: str) -> pd.DataFrame:
        """
        Download all symbols from a specific exchange
        
        Args:
            exchange_code: Exchange code (e.g., 'US', 'LSE')
            
        Returns:
            DataFrame containing exchange symbols
        """
        endpoint = f"exchange-symbol-list/{exchange_code}"
        data = self.client.get(endpoint)
        return pd.DataFrame(data)


class IndexData:
    """Class for managing index-related data"""
    
    def __init__(self, client: Optional[EODHDClient] = None):
        """
        Initialize index data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else EODHDClient()
    
    def get_sp500_components(self) -> pd.DataFrame:
        """
        Retrieve current S&P 500 components
        
        Returns:
            DataFrame containing S&P 500 components
        """
        endpoint = "fundamentals/GSPC.INDX"
        data = self.client.get(endpoint)
        data = list(data.items())[2]
        return pd.DataFrame.from_dict(data[1], orient='index')
    
    def get_historical_sp500(self) -> pd.DataFrame:
        """
        Retrieve S&P 500 component history since 1990
        
        Returns:
            DataFrame containing S&P 500 component history
        """
        # This method uses Wikipedia data
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract current constituents
        current_constituents_table = soup.find('table', {'id': 'constituents'})
        current_constituents = pd.read_html(str(current_constituents_table))[0]
        
        # Extract historical changes
        changes_table = soup.find('table', {'id': 'changes'})
        changes = pd.read_html(str(changes_table), header=0)[0]
        changes.columns = ['Date', 'AddTicker', 'AddName', 'RemovedTicker', 'RemovedName', 'Reason']
        changes = changes.drop([0, 1]).reset_index(drop=True)
        changes['Date'] = pd.to_datetime(changes['Date'], format='mixed', errors='coerce')
        changes['year'] = changes['Date'].dt.year
        changes['month'] = changes['Date'].dt.month
        
        # Build history
        current_month = pd.to_datetime(datetime.now().strftime('%Y-%m-01'))
        month_seq = pd.date_range(start='1990-01-01', end=current_month, freq='MS')[::-1]
        spx_stocks = current_constituents.assign(Date=current_month)[['Date', 'Symbol', 'Security']]
        spx_stocks.columns = ['Date', 'Ticker', 'Name']
        last_run_stocks = spx_stocks.copy()
        
        for d in month_seq[1:]:
            y, m = d.year, d.month
            changes_in_month = changes[(changes['year'] == y) & (changes['month'] == m)]
            
            tickers_to_keep = last_run_stocks[~last_run_stocks['Ticker'].isin(changes_in_month['AddTicker'])].assign(Date=d)
            tickers_to_add = changes_in_month[changes_in_month['RemovedTicker'] != ''][['RemovedTicker', 'RemovedName']].assign(Date=d)
            tickers_to_add.columns = ['Ticker', 'Name', 'Date']
            
            this_month = pd.concat([tickers_to_keep, tickers_to_add])
            spx_stocks = pd.concat([spx_stocks, this_month])
            
            last_run_stocks = this_month
        
        return spx_stocks
    
    def get_sp500_data(self) -> pd.DataFrame:
        """
        Download S&P 500 ETF (SPY) data as a proxy for S&P 500
        
        Returns:
            DataFrame containing S&P 500 data
        """
        ticker = "SPY"
        
        # Get technical and price data
        ticker_data = TickerData(self.client)
        price_data = ticker_data.get_raw_price_data(ticker)
        
        fundamental_data = FundamentalData(self.client)
        technical_data = fundamental_data.get_technical_data(ticker)
        
        if not price_data.empty:
            # Merge data
            combined_data = technical_data.merge(
                price_data[['date', 'adjusted_close']], 
                on='date', 
                how='left'
            )
            combined_data['ticker'] = ticker
            return combined_data
        
        return pd.DataFrame()


class PriceData:
    """Class for managing individual stock data"""
    
    def __init__(self, client: Optional[EODHDClient] = None):
        """
        Initialize stock data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else EODHDClient()
    
    def get_raw_price_data(self, symbol: str) -> pd.DataFrame:
        """
        Download raw price data for a symbol
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            DataFrame containing price data
        """
        endpoint = f"eod/{symbol}"
        data = self.client.get(endpoint)
        return pd.DataFrame(data)
    
    def get_technical_data(self, symbol: str) -> pd.DataFrame:
        """
        Download split-adjusted technical data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame containing technical data
        """
        endpoint = f"technical/{symbol}"
        params = {"function": "splitadjusted"}
        data = self.client.get(endpoint, params)
        return pd.DataFrame(data)
        
    def get_historical_market_cap(self, symbol: str) -> pd.DataFrame:
        """
        Download historical market capitalization data
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame containing market cap history
        """
        endpoint = f"historical-market-cap/{symbol}"
        data = self.client.get(endpoint)
        return pd.DataFrame(data)
    
    def process_price_data(self, 
                          raw_data_list: List[pd.DataFrame], 
                          technical_data_list: List[pd.DataFrame], 
                          tickers: List[str]) -> pd.DataFrame:
        """
        Process and combine raw and technical price data
        
        Args:
            raw_data_list: List of raw data DataFrames
            technical_data_list: List of technical data DataFrames
            tickers: List of corresponding symbols
            
        Returns:
            Combined price data DataFrame
        """
        final_price_list = []
        
        for idx, ticker in enumerate(tqdm(tickers, desc="Processing price data")):
            tp = pd.DataFrame(technical_data_list[idx])
            rp = pd.DataFrame(raw_data_list[idx])
        
            if not rp.empty:
                fp = tp.merge(rp[['date', 'adjusted_close']], on='date', how='left')
                fp['ticker'] = ticker
                final_price_list.append(fp)
        
        if final_price_list:
            return pd.concat(final_price_list, ignore_index=True)
        return pd.DataFrame()
    
    def get_price_data_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Récupère et combine les données de prix pour une liste de symboles
        
        Args:
            tickers: Liste des symboles
            
        Returns:
            DataFrame combiné avec les données de prix
        """
        raw_data = []
        technical_data = []
        
        for ticker in tqdm(tickers, desc="Téléchargement des données"):
            raw_data.append(self.get_raw_price_data(ticker))
            technical_data.append(self.get_technical_data(ticker))
        
        return self.process_price_data(raw_data, technical_data, tickers)


class FundamentalData:
    """Class for managing company fundamental data"""
    
    def __init__(self, client: Optional[EODHDClient] = None):
        """
        Initialize fundamental data manager
        
        Args:
            client: EODHD API client (optional, will create a new one if None)
        """
        self.client = client if client else EODHDClient()
    
    def get_fundamental_data(self, symbol: str) -> Dict:
        """
        Download fundamental data for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing fundamental data
        """
        endpoint = f"fundamentals/{symbol}"
        return self.client.get(endpoint)
    
    def process_fundamental_data(self, 
                               fundamental_data_list: List[Dict], 
                               tickers: List[str], 
                               data_type: str) -> pd.DataFrame:
        """
        Process and extract specific fundamental data type
        
        Args:
            fundamental_data_list: List of fundamental data dictionaries
            tickers: List of corresponding symbols
            data_type: Type of data to extract ('general', 'Income_Statement', 
                      'Balance_Sheet', 'Cash_Flow', 'Earnings', 'outstandingShares')
            
        Returns:
            DataFrame containing extracted fundamental data
        """
        final_data_list = []
        
        for idx, ticker in enumerate(tqdm(tickers, desc=f"Processing {data_type} data")):
            try:
                if data_type == "general":
                    filtered_data = {k: v for k, v in fundamental_data_list[idx]['General'].items() 
                                     if isinstance(v, str)}
                    tp = pd.DataFrame([filtered_data])
                    
                elif data_type == "Income_Statement":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Financials']['Income_Statement']['quarterly'], 
                        orient='index'
                    )
                    
                elif data_type == "Balance_Sheet":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Financials']['Balance_Sheet']['quarterly'], 
                        orient='index'
                    )
                    
                elif data_type == "Cash_Flow":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Financials']['Cash_Flow']['quarterly'], 
                        orient='index'
                    )
                    
                elif data_type == "Earnings":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['Earnings']['History'], 
                        orient='index'
                    )
                    
                elif data_type == "outstandingShares":
                    tp = pd.DataFrame.from_dict(
                        fundamental_data_list[idx]['outstandingShares']['quarterly'], 
                        orient='index'
                    )
                    
                else:
                    raise ValueError(
                        "Invalid type. Use 'general', 'Income_Statement', 'Balance_Sheet', "
                        "'Cash_Flow', 'Earnings', or 'outstandingShares'."
                    )
                
                tp['ticker'] = ticker
                final_data_list.append(tp)
                
            except (KeyError, TypeError) as e:
                print(f"Error processing data for {ticker}: {e}")
        
        if final_data_list:
            return pd.concat(final_data_list, ignore_index=True)
        return pd.DataFrame()


class EODHDDataService:
    """
    Service principal pour l'accès aux données financières EODHD.
    Cette classe fournit une interface unifiée pour toutes les fonctionnalités.
    """
    
    def __init__(self, api_key: str):
        """
        Initialise le service de données
        
        Args:
            api_key: Clé API EODHD
        """
        self.config = APIConfig(api_key)
        self.client = EODHDClient(self.config)
        self.exchange_data = ExchangeData(self.client)
        self.ticker_data = TickerData(self.client)
        self.fundamental_data = FundamentalData(self.client)
        self.index_data = IndexData(self.client)
    
    def get_sp500_historical_composition(self) -> pd.DataFrame:
        """
        Récupère la composition historique du S&P 500
        
        Returns:
            DataFrame avec la composition historique
        """
        return self.index_data.get_historical_sp500()
    
    def get_sp500_components(self) -> pd.DataFrame:
        """
        Récupère les composants actuels du S&P 500
        
        Returns:
            DataFrame avec les composants actuels
        """
        return self.index_data.get_sp500_components()
    
    def get_ticker_list_from_exchange(self, exchange_code: str) -> pd.DataFrame:
        """
        Récupère la liste des actions d'une bourse
        
        Args:
            exchange_code: Code de la bourse
            
        Returns:
            DataFrame avec la liste des actions
        """
        return self.exchange_data.get_tickers_from_exchange(exchange_code)
    
    def get_price_data_for_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Récupère et combine les données de prix pour une liste de symboles
        
        Args:
            tickers: Liste des symboles
            
        Returns:
            DataFrame combiné avec les données de prix
        """
        raw_data = []
        technical_data = []
        
        for ticker in tqdm(tickers, desc="Téléchargement des données"):
            raw_data.append(self.ticker_data.get_raw_price_data(ticker))
            technical_data.append(self.fundamental_data.get_technical_data(ticker))
        
        return self.ticker_data.process_price_data(raw_data, technical_data, tickers)
    
    def get_fundamental_data(self, 
                             tickers: List[str]) -> List:
        """
        Récupère et traite les données fondamentales pour une liste de symboles
        
        Args:
            tickers: Liste des symboles
            data_type: Type de données fondamentales à extraire
            
        Returns:
            DataFrame avec les données fondamentales
        """
        fundamental_data = []
        
        for ticker in tqdm(tickers, desc=f"Téléchargement des données"):
            fundamental_data.append(self.fundamental_data.get_fundamental_data(ticker))
        
        return fundamental_data
    
    def retreat_fundamental_data(self,
                                 fundamental_data : List,
                                 data_type) -> pd.DataFrame:
        """
        Récupère et traite les données fondamentales pour une liste de symboles
        
        Args:
            tickers: Liste des symboles
            data_type: Type de données fondamentales à extraire
            
        Returns:
            DataFrame avec les données fondamentales
        """
        
        return self.fundamental_data.process_fundamental_data(fundamental_data, tickers, data_type)
    
    def get_sp500_price_data(self) -> pd.DataFrame:
        """
        Récupère les données de prix du S&P 500 (via SPY)
        
        Returns:
            DataFrame avec les données de prix du S&P 500
        """
        return self.index_data.get_sp500_data()


# Exemple d'utilisation:
if __name__ == "__main__":
    # Remplacez par votre clé API
    API_KEY = "votre_clé_api_ici"
    
    # Création du service
    service = EODHDDataService(API_KEY)
    
    # Exemples d'utilisation
    sp500_components = service.get_sp500_components()
    print(f"Nombre de composants S&P 500: {len(sp500_components)}")
    
    # Téléchargement des données pour quelques actions
    tickers = ["AAPL", "MSFT", "AMZN"]
    price_data = service.get_price_data_for_tickers(tickers)
    print(f"Données de prix récupérées: {len(price_data)} entrées")
    
    # Téléchargement des données fondamentales
    balance_sheet_data = service.get_fundamental_data_for_tickers(tickers, "Balance_Sheet")
    print(f"Données de bilan récupérées: {len(balance_sheet_data)} entrées")