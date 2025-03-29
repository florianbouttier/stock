# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 17:11:16 2024

@author: flbouttier
"""
# %% Package
import os
import requests
import json
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
import time
import numpy as np


# %% Set API key
api_key = "66cdc161eb4645.37830398"

# %% Définir le répertoire de l'environnement virtuel actif
env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
#env_dir = 'C:\\Users\\flori\\stock_market\\'
env_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'stock')
data_dir = os.path.join(env_dir, 'data')

# %% functions
def download_ticker_from_exchange(exchange_code):
    url_1 = f"https://eodhd.com/api/exchange-symbol-list/{exchange_code}?api_token={api_key}&fmt=json"
    response_1 = pd.DataFrame(requests.get(url_1).json())
    return response_1

def historical_sp500components():
    url = f'https://eodhd.com/api/fundamentals/GSPC.INDX?api_token={api_key}&fmt=json'
    data = requests.get(url).json()
    data = list(data.items())[2]
    data = pd.DataFrame.from_dict(data[1], orient='index')
    return data

def download_fundamental_data(symbol):
    url = f"https://eodhistoricaldata.com/api/fundamentals/{symbol}?api_token={api_key}&fmt=json"
    response = requests.get(url).json()
    return response

def download_raw_price_data(symbol):
    url = f"https://eodhd.com/api/eod/{symbol}?api_token={api_key}&fmt=json"
    response = pd.DataFrame(requests.get(url).json())
    return response

def download_technical_data(symbol):
    url = f"https://eodhd.com/api/technical/{symbol}?function=splitadjusted&api_token={api_key}&fmt=json"
    response = pd.DataFrame(requests.get(url).json())
    return response

def retreat_funda(fundamental, list_ticker, ty):
    final_price_list = []
    list_ticker = list(list_ticker)

    for ticker in tqdm(list_ticker):
        if ty == "general":
            tp = pd.DataFrame()
            filtered_data = {k: v for k, v in fundamental[list_ticker.index(ticker)]['General'].items() if isinstance(v, str)}
            tp = pd.DataFrame([filtered_data])
            tp['ticker'] = ticker
        elif ty == "Income_Statement":
            tp = pd.DataFrame.from_dict(fundamental[list_ticker.index(ticker)]['Financials']['Income_Statement']['quarterly'], orient='index')
            tp['ticker'] = ticker
        elif ty == "Balance_Sheet":
            tp = pd.DataFrame.from_dict(fundamental[list_ticker.index(ticker)]['Financials']['Balance_Sheet']['quarterly'], orient='index')
            tp['ticker'] = ticker
        elif ty == "Cash_Flow":
            tp = pd.DataFrame.from_dict(fundamental[list_ticker.index(ticker)]['Financials']['Cash_Flow']['quarterly'], orient='index')
            tp['ticker'] = ticker
        elif ty == "Earnings":
            tp = pd.DataFrame.from_dict(fundamental[list_ticker.index(ticker)]['Earnings']['History'], orient='index')
            tp['ticker'] = ticker
        elif ty == "outstandingShares":
            tp = pd.DataFrame.from_dict(fundamental[list_ticker.index(ticker)]['outstandingShares']['quarterly'], orient='index')
            tp['ticker'] = ticker
        else:
            raise ValueError("Invalid type. Use 'general', 'Income_Statement', 'Balance_Sheet', or 'Cash_Flow'.")
        final_price_list.append(tp)
    final_price = pd.concat(final_price_list, ignore_index=True)
    return final_price 

def download_historical_sp500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    current_constituents_table = soup.find('table', {'id': 'constituents'})
    current_constituents = pd.read_html(str(current_constituents_table))[0]
    
    changes_table = soup.find('table', {'id': 'changes'})
    changes = pd.read_html(str(changes_table), header=0)[0]
    changes.columns = ['Date', 'AddTicker', 'AddName', 'RemovedTicker', 'RemovedName', 'Reason']
    changes = changes.drop([0, 1]).reset_index(drop=True)
    changes['Date'] = pd.to_datetime(changes['Date'], format='%B %d, %Y')
    changes['year'] = changes['Date'].dt.year
    changes['month'] = changes['Date'].dt.month
    
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

def exchange_supported():
    url = f'https://eodhd.com/api/exchanges-list/?api_token={api_key}&fmt=json'
    data = requests.get(url).json()
    return pd.DataFrame(data)

def historical_capitalisation(symbol):
    url = f'https://eodhd.com/api/historical-market-cap/{symbol}?api_token={api_key}&fmt=json'
    data = pd.DataFrame(requests.get(url).json())
    return data

def retreatement_price(raw, technical, list_ticker):
    final_price_list = []
    list_ticker = list(list_ticker)

    for ticker in tqdm(list_ticker):
        tp = pd.DataFrame(technical[list_ticker.index(ticker)])
        rp = pd.DataFrame(raw[list_ticker.index(ticker)])
    
        if not rp.empty:
            fp = tp.merge(rp[['date', 'adjusted_close']], on='date', how='left')
            fp['ticker'] = ticker
        else:
            fp = pd.DataFrame()
    
        final_price_list.append(fp)
    
    final_price = pd.concat(final_price_list, ignore_index=True)
    return final_price

def download_sp500_data():
    list_ticker_sp500 = ["SPY"]
    raw_price_sp500 = [download_raw_price_data(ticker) for ticker in list_ticker_sp500]
    technical_price_sp500 = [download_fundamental_data(ticker) for ticker in list_ticker_sp500]
    
    final_price_sp500 = pd.concat([
        technical_price_sp500[t].assign(ticker=t).merge(
            raw_price_sp500[t][['date', 'adjusted_close']], on='date', how='left'
        ) for t in list_ticker_sp500 if not raw_price_sp500[t].empty
    ])
    return final_price_sp500
# %% Téléchargement et sauvegarde des données
historical_company_sp500 = download_historical_sp500()
# %%
historical_company_sp500.to_csv(os.path.join(data_dir, "SP500_Constituents.csv"), index=False)
historical_company = historical_company_sp500.copy()
# %%
historical_company = (historical_company
                      .merge(download_ticker_from_exchange('US')[['Code', 'Type']],
                             left_on=["Ticker"],
                             right_on=["Code"], how="left"))
historical_company = historical_company[(historical_company["Type"] == "Common Stock") | pd.isna(historical_company["Type"])]
# %%
list_ticker = historical_company.dropna(subset=['Ticker'])['Ticker'].unique()
list_ticker = [str(ticker).replace('.', '-') + ".US" for ticker in list_ticker]

funda = [download_fundamental_data(ticker) for ticker in tqdm(list_ticker)]
raw_price = [download_raw_price_data(ticker) for ticker in tqdm(list_ticker)]
technical_price = [download_technical_data(ticker) for ticker in tqdm(list_ticker)]

Finalprice = retreatement_price(raw=raw_price, technical=technical_price, list_ticker=list_ticker)
General = retreat_funda(funda, list_ticker=list_ticker, ty="general")
Income_Statement = retreat_funda(funda, list_ticker=list_ticker, ty="Income_Statement")
Balance_Sheet = retreat_funda(funda, list_ticker=list_ticker, ty="Balance_Sheet")
Cash_Flow = retreat_funda(funda, list_ticker=list_ticker, ty="Cash_Flow")
Earnings = retreat_funda(funda, list_ticker=list_ticker, ty="Earnings")
outstandingShares = retreat_funda(funda, list_ticker=list_ticker, ty="outstandingShares")

# %%
Selected_Exchange = "US"
Finalprice.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Finalprice.parquet'))
General.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_General.parquet'))
Income_Statement.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Income_statement.parquet'))
Balance_Sheet.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Balance_sheet.parquet'))
Cash_Flow.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Cash_flow.parquet'))
Earnings.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Earnings.parquet'))
outstandingShares.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_share.parquet'))

SP500Price = retreatement_price(
    raw=[download_raw_price_data(ticker) for ticker in ["SPY"]],
    technical=[download_technical_data(ticker) for ticker in ["SPY"]],
    list_ticker=["SPY"]
)
SP500Price.to_parquet(os.path.join(data_dir, "SP500Price.parquet"))
