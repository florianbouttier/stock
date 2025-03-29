
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:43:16 2024

@author: flbouttier
"""


# %%
%load_ext autoreload
%autoreload 2
import os
import glob
import requests
import json
import numpy as np
import scipy.stats as stats
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from tqdm import tqdm
import time
from datetime import timedelta
import scipy.stats as stats
from scipy import special
import matplotlib.pyplot as plt
import warnings
import re 
import seaborn as sns
from scipy.stats import rankdata
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.cluster import KMeans
import warnings
from datetime import datetime
sys.path.append(os.path.dirname(os.getcwd()))


import src.functions_backtest as funct_backtest

#env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
env_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'stock')
#env_dir = os.getcwd()
data_dir = os.path.join(env_dir, 'data')
os.chdir(data_dir)

# %% US
Finalprice= pd.read_parquet('US_Finalprice.parquet')
General= pd.read_parquet('US_General.parquet')
Income_Statement = pd.read_parquet('US_Income_statement.parquet')
Balance_Sheet= pd.read_parquet('US_Balance_sheet.parquet')
Cash_Flow = pd.read_parquet('US_Cash_flow.parquet')
Earnings= pd.read_parquet('US_Earnings.parquet')
US_historical_company = pd.read_csv("SP500_Constituents.csv")
SP500Price = pd.read_parquet('SP500Price.parquet')
Finalprice['year_month'] = pd.to_datetime(Finalprice['date']).dt.to_period('M')

US_historical_company['ticker'] = US_historical_company['Ticker'].apply(lambda x: re.sub(r'\.', '-', x) if isinstance(x, str) else x)
US_historical_company['ticker'] = US_historical_company['ticker'] + '.US'
US_historical_company['Month'] = pd.to_datetime(US_historical_company['Date']).dt.to_period('M')
Finalprice['year_month'] = pd.to_datetime(Finalprice['date']).dt.to_period('M')
mr = funct_backtest.calculate_monthly_returns(df = Finalprice)
Selection_Stocks = funct_backtest.calculate_pe_ratios(balance = Balance_Sheet, earnings = Earnings, monthly_return=mr)
#Selection_Stocks['Market_Cap'] = Selection_Stocks['close']*Selection_Stocks['commonStockSharesOutstanding']
Selection_Stocks_ = (Selection_Stocks[(Selection_Stocks['PE']<100) & (Selection_Stocks['PE']>0)]
                    .dropna(subset = ['PE', 'Market_Cap'])
                    .merge(US_historical_company[['Month','ticker']],
                            how = "inner",
                            left_on = ['ticker','year_month'],
                            right_on = ['ticker','Month']))

SP500_Monthly = (       
    SP500Price
    .sort_values('date') # Trier par date
    .assign(DR_SP500=lambda x: x['adjusted_close'] / x['adjusted_close'].shift(1))  # Calculer le rendement
    .assign(Month=lambda x: pd.to_datetime(x['date']).dt.to_period('M'))  # Convertir la colonne 'date' en période mensuelle
    .groupby('Month')  # Grouper par mois
    .agg({'DR_SP500': 'prod'})  # Agréger par produit pour obtenir le rendement cumulé
    .reset_index()
)


Finalprice = Finalprice[Finalprice['ticker'].isin(US_historical_company[US_historical_company['Month'] > '2000-01']['ticker'].unique())]
Historical_Company = US_historical_company

# %% Functions
def fot(df,ticker,columns = 'ticker') : #doing it all the time for test
    return df[df[columns] == ticker]
def retreate_prices(df):
    # Ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # Create a complete date range
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    
    # Set the index and unstack, then reindex with the complete date range
    df_full = df.set_index(['date', 'ticker']).unstack('ticker').reindex(all_dates).stack('ticker', dropna=False).reset_index().rename(columns={'level_0': 'date'})
    
    # Forward fill missing values
    df_full['adjusted_close'] = df_full.groupby('ticker')['adjusted_close'].ffill()
    
    # Select relevant columns
    df_full = df_full[['date', 'ticker', 'adjusted_close']]
    
    return df_full
def ema_moving_average(series, n,wilder = False):

    previous = series.ewm(span=n, adjust=False).mean()
    """
    alpha = 2 / (n + 1)

    # Calculate the exponentially weighted mean manually
    ema = [None] * (n - 1)  # Initial values as None
    ema.append(series[:n].mean())  # First EMA value as the mean of the first n values

    for value in series[n:]:
        ema.append((value * alpha) + (ema[-1] * (1 - alpha)))

    previous = pd.Series(ema)"""
    return previous 
def Price_VS_Index(Index,Prices) : 
    
    Index['date'] = pd.to_datetime(Index['date'])
    full_date_range = pd.date_range(start=Index['date'].min(), end=Index['date'].max())
    
    Index.set_index('date', inplace=True)
    Index = Index.reindex(full_date_range)
    Index['adjusted_close'] = Index['adjusted_close'].ffill()
    Index.ffill(inplace=True)
    Index = Index.reset_index().rename(columns = {"index" : "date"})
    
    #Prices_Augmented = retreate_prices(Prices)
    Prices_Augmented = Prices.copy()    
    Prices_Augmented['date'] = pd.to_datetime(Prices_Augmented['date'])
    Index['date'] = pd.to_datetime(Index['date'])
    
    Prices_Augmented= Prices_Augmented.merge(Index[['date','adjusted_close']],how = "left",on = "date")
    Prices_Augmented['Close_VS_SP500'] = Prices_Augmented['adjusted_close_x']/ Prices_Augmented['adjusted_close_y']
    Prices_Augmented= Prices_Augmented.rename(columns = {"adjusted_close_x" : "Close"})
    
    Prices_Augmented.sort_values(by=['ticker', 'date'], inplace=True)
    Prices_Augmented['Close_Lag'] = Prices_Augmented.groupby('ticker')['Close'].shift(1)
    Prices_Augmented['DR'] = Prices_Augmented['Close'] / Prices_Augmented['Close_Lag']
    Prices_Augmented['Month'] = Prices_Augmented['date'].dt.to_period('M')
    
    #Prices_Augmented = Prices_Augmented.merge(Price[['date','ticker']],on = ['date','ticker'],how = "inner")
    
    return Prices_Augmented[['date','Month','ticker','Close','Close_VS_SP500','DR']]
def Learning_Monthly(Data, Historical_Company, N_Long, N_Short, func_MovingAverage, N_Asset):
    
    #df = Data.dropna(subset=['Close_VS_SP500']).sort_values(by=['ticker', 'date'])
    df = Data

    for n, col in [(N_Short, 'Short_VS_SP500'), (N_Long, 'Long_VS_SP500')]:
        df[col] = df.groupby('ticker')['Close_VS_SP500'].transform(lambda x: func_MovingAverage(x, n=n))
    
    df = df.groupby('ticker').apply(lambda x: x.iloc[N_Long:],include_groups = False).reset_index(drop=False)

    df = df.dropna(subset = ['Short_VS_SP500','Long_VS_SP500'])
    pd.options.mode.chained_assignment = None  # default='warn'
    df.loc[:, 'MTR'] = df['Short_VS_SP500'] / df['Long_VS_SP500']

    # Merge with Historical_Company and select latest date per month
    df = df.merge(Historical_Company, on=['Month', 'ticker'], how='inner')
    df = df.reset_index(drop=True)  # Réinitialise l'index après le merge

    # Sélection des lignes avec la date maximale par mois et ticker
    max_date_indices = df.groupby(['Month', 'ticker'])['date'].idxmax()
    df = df.loc[max_date_indices.values]
    # Calculate statistics
    grouped = df.groupby('Month')
    df = df.assign(
        Mean=grouped['MTR'].transform('mean'),
        Ecart=grouped['MTR'].transform('std'),
        quantile_MTR=lambda x: stats.norm.cdf(x['MTR'], x['Mean'], x['Ecart'])
    )
    
    # Select top N_Asset per month based on MTR
    df = df.sort_values(by=['Month', 'MTR'], ascending=[True, False]).groupby('Month').head(N_Asset)
    
    df = df.assign(N_Long=N_Long, N_Short=N_Short, N_Asset=N_Asset)[['Month', 'date', 'ticker', 'N_Long', 'N_Short', 'N_Asset', 'MTR', 'quantile_MTR']]
    
    # Compute Final_Return
    Final_Return = (Data.groupby(['Month', 'ticker'])['DR'].prod()
                    .reset_index()
                    .sort_values(by=['ticker', 'Month'])
                    .assign(application_month=lambda x: x.groupby('ticker')['Month'].shift(1))
                    .merge(df, left_on=['application_month', 'ticker'], right_on=['Month', 'ticker'], how='inner')
                    .rename(columns={"Month_x": "Month"})[['Month', 'application_month', 'date', 'ticker', 'DR', 'N_Long', 'N_Short', 'N_Asset', 'MTR', 'quantile_MTR']]
                    )

    return Final_Return,df
def decreasing_sum(liste, halfperiod, mode):
    n = len(liste)
    if n == 0:
        return 0.0

    Weight = np.zeros(n)

    if mode == "exponential":
        p = np.log(2) / halfperiod
        Weight = np.exp(-p * np.arange(n))
    elif mode == "tanh":
        p = np.log(3) / (2 * halfperiod)
        Weight = 1 - np.tanh(p * np.arange(n))
    elif mode == "special":
        alpha = halfperiod
        Weight = np.maximum(1 - (1 + (1 + alpha * np.arange(n)) * (np.log(1 + alpha * np.arange(n)) - 1) / (alpha**2)), 0)
    elif mode == "linear":
        Weight = np.maximum(1 - np.arange(n) / halfperiod, 0)
    elif mode == "quadratic":
        Weight = np.maximum(1 - (np.arange(n) / halfperiod)**2, 0)
    elif mode == "sigmoidal":
        k = np.log(3) / halfperiod
        Weight = 1 / (1 + np.exp(k * (np.arange(n) - halfperiod)))
    elif mode == "mean":
        Len1 = min(halfperiod, n)
        Weight[:Len1] = 1
    else:
        raise ValueError(f"Unknown mode: {mode}.")

    Weight_sum = np.sum(Weight)
    Weight = np.divide(Weight, Weight_sum) if Weight_sum != 0 else np.full(n, 1.0 / n)

    return np.dot(Weight, liste)
def custom_SMA(series, n):
    """Fonction personnalisée pour calculer la moyenne mobile simple (SMA)."""
    return series.rolling(window=n, min_periods=1).mean()
def increase(values, n, diff=True, annual_base=4):
    values = pd.Series(values)  # S'assure que values est une Series
    
    if diff:
        return values - values.shift(n)  # Différence simple

    v0, v1 = values.shift(n), values  # Valeur initiale et valeur actuelle
    #denom = (v0.abs() + v1.abs()) / 2
    denom = v0.abs()
    growth = np.where(denom == 0, np.nan, (v1 - v0) / denom)
    
    base = 1 + growth
    # On vérifie que base est positif pour éviter d'élever un nombre négatif à une puissance fractionnaire
    result = np.where(base > 0, base ** (annual_base / n) - 1, base)
    
    return pd.Series(result, index=values.index)
def retreating_fundamental(income_statement, balance_sheet, cash_flow, earning,price):
  
    income = income_statement[['date', 'ticker', 'filing_date', 'totalRevenue', 'grossProfit', 
                               'operatingIncome', 'incomeBeforeTax', 'netIncome', 'ebit', 'ebitda']]
    
    
    cash = cash_flow[['date', 'ticker', 'filing_date', 'freeCashFlow']]
  
    balance = balance_sheet[['date', 'ticker', 'filing_date', 'totalAssets', 'totalLiab', 
                             'totalStockholderEquity', 'netDebt', 'commonStockSharesOutstanding']]
    
    earning = earning[['date', 'ticker', 'reportDate', 'beforeAfterMarket', 'epsActual']].dropna(subset=['epsActual'])
    
    income['date'],cash['date'],balance['date'],earning ['date'] = pd.to_datetime(income['date']),pd.to_datetime(cash['date']),pd.to_datetime(balance['date']),pd.to_datetime(earning ['date'])
    
    fundamental = income.merge(balance, on=['date', 'ticker'], how='outer') \
                                 .merge(cash, on=['date', 'ticker'], how='outer') \
                                 .merge(earning, on=['date', 'ticker'], how='outer')
                                 
                                 
    
    balance.loc[:, 'filing_date'] = pd.to_datetime(balance['filing_date']) + timedelta(days=1)
    balance.loc[:, 'date'] = pd.to_datetime(balance['date']) 
    balance= (balance.rename(columns={'date': 'Quarter', 'filing_date': 'date'})
           .sort_values(by=['ticker', 'Quarter'])
           .groupby(['ticker', 'date'], as_index=False)
           .apply(lambda x: x[x['Quarter'] == x['Quarter'].max()]).reset_index() 
           [['ticker', 'date', 'Quarter','commonStockSharesOutstanding']]) 
    balance['Quarter'] = pd.to_datetime(balance['Quarter'])
    
    #Earnings
    earning['beforeAfterMarket'] = earning['beforeAfterMarket'].fillna("AfterMarket_Replaced")
    earning['reportDate'] = pd.to_datetime(earning['reportDate'])
    earning['date'] = pd.to_datetime(earning['date'])
    # Ajuster 'reportDate' si 'beforeAfterMarket' est "BeforeMarket"
    earning['reportDate'] = earning.apply(
        lambda row: row['reportDate'] + timedelta(days=1) if row['beforeAfterMarket'] != "BeforeMarket" else row['reportDate'],
        axis=1
    )
    
    # Renommer les colonnes
    earning = earning.rename(columns={'date': 'Quarter', 'reportDate': 'date'})
    
    # Remplacer les dates manquantes par 'Quarter' + 3 mois
    """
    earning['date'] = earning.apply(
        lambda row: row['Quarter'] + pd.DateOffset(months=3) if pd.isna(row['date']) else row['date'],
        axis=1
    )"""
    
    # Filtrer pour garder le dernier 'Quarter' par groupe 'ticker' et 'date'
    earning = earning.sort_values(by=['ticker', 'date', 'Quarter'], ascending=[True, True, False]) \
                       .drop_duplicates(subset=['ticker', 'date'], keep='first')
    
    # Trier et calculer la moyenne mobile simple (Rolling_epsActual)
    earning = earning.sort_values(by=['ticker', 'date'])
    earning['Rolling_epsActual'] = earning.groupby('ticker')['epsActual'] \
                                            .transform(lambda x: 4 * custom_SMA(x, n=4))
    price['date'] = pd.to_datetime(price['date'])                                        
    
    Merge = earning.merge(balance,left_on = ["Quarter","ticker"],right_on = ["Quarter","ticker"])                                            
    Merge = Merge.dropna(subset = ['date_x','date_y'])
    Merge['date_x'],Merge['date_y'] = pd.to_datetime(Merge['date_x']),pd.to_datetime(Merge['date_y'])
    Merge['Final_Date'] = np.maximum(Merge['date_x'], Merge['date_y'])
    Merge_ = Merge.merge(price, left_on=["Final_Date", "ticker"], right_on=["date", "ticker"], how="outer")
    return  Merge
def augmenting_ratios(data,kpi_list,date_list) : 
    
    data = data.sort_values(['ticker']+list([date_list]), ascending=[True, True])

    # Fonction pour compter les jours consécutifs de valeurs positives
    def days_since_last_negative(group, kpi):
        last_negative_date = group['date'].where(group[kpi] < 0).ffill()
        return (group['date'] - last_negative_date).dt.days

    for kpi in  kpi_list :
        data[f"{kpi}_days_increase"] = (
        data.groupby('ticker', group_keys=False)[['date', kpi]]
            .apply(days_since_last_negative, kpi)
            .reset_index(level=0, drop=True))
        
    return data   
def calculate_pe_ratios(balance, earnings, monthly_return):
    # 1. Traitement du bilan comptable
    balance_clean = (
        balance[['ticker', 'date', 'filing_date', 'commonStockSharesOutstanding']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
            filing_date=lambda x: pd.to_datetime(x['filing_date'])
        )
        .sort_values('filing_date')
        .groupby(['ticker', 'quarter_end'])
        .last()  # Prendre la dernière version du rapport pour chaque trimestre
        .reset_index()
                    )

    # 2. Traitement des résultats
    earnings_clean = (
        earnings[['ticker', 'date', 'reportDate', 'epsActual']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
            report_date=lambda x: pd.to_datetime(x['reportDate'])
        )
        .sort_values('report_date')
        .groupby(['ticker', 'quarter_end'])
        .last()  # Prendre la dernière révision des résultats
        .reset_index()
        ).dropna(subset = ['epsActual'])
    earnings_clean['Rolling_epsActual'] = earnings_clean.sort_values(['ticker','report_date']).groupby('ticker')['epsActual'] \
                                            .transform(lambda x: 4 * custom_SMA(x, n=4))
    
    monthly_return['date'] = pd.to_datetime(monthly_return['date'])
    price_merge = (monthly_return
             .merge(earnings_clean[['ticker', 'quarter_end', 'report_date', 'Rolling_epsActual']],
                    left_on = ['ticker', 'date'],
                    right_on =['ticker', 'report_date'],
                    how='outer')
             .sort_values(by=['ticker', 'date']))
    price_merge['final_date'] = price_merge[['date','report_date']].max(axis=1)
    price_merge = price_merge.sort_values(by='final_date')

    # Appliquer ffill sur last_close et Rolling_epsActual par ticker
    price_merge[['last_close', 'Rolling_epsActual']] = (
        price_merge.groupby('ticker')[['last_close', 'Rolling_epsActual']].ffill()
        )

    price_merge['PE'] = price_merge['last_close']/price_merge['Rolling_epsActual']
    price_merge = price_merge.groupby(['ticker', 'year_month'],group_keys = False).apply(lambda x: x.loc[x['date'].idxmax()],include_groups=False).reset_index()
    price_merge_lvl2 = (price_merge
                        .merge(balance_clean[['ticker', 'filing_date', 'commonStockSharesOutstanding']],
                                left_on = ['ticker', 'date'],
                                right_on =['ticker', 'filing_date'],
                                how='outer')
                        .sort_values(by=['ticker', 'date']))
    price_merge_lvl2['final_date_V2'] = price_merge_lvl2[['date','filing_date']].max(axis=1)
    price_merge_lvl2 = price_merge_lvl2.sort_values(by='final_date_V2').reset_index(drop = True)

    price_merge_lvl2[['commonStockSharesOutstanding']] = (
        price_merge_lvl2.groupby('ticker')[['commonStockSharesOutstanding']].ffill()
        )
    
    price_merge_lvl2 = price_merge_lvl2.groupby(['ticker', 'year_month'],group_keys = False).apply(lambda x: x.loc[x['date'].idxmax()],include_groups=False)
    price_merge_lvl2 = price_merge_lvl2.reset_index()
    price_merge_lvl2['Market_Cap'] = price_merge_lvl2['last_close']*pd.to_numeric(price_merge_lvl2['commonStockSharesOutstanding'])
    return price_merge_lvl2[['ticker','year_month','PE','commonStockSharesOutstanding','Market_Cap']]
def calculate_fundamental_ratios(balance,
                                 cashflow,
                                 income,
                                 earnings,
                                 list_kpi_toincrease = ['totalRevenue_rolling', 'grossProfit_rolling', 'operatingIncome_rolling', 'incomeBeforeTax_rolling', 'netIncome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freeCashFlow_rolling', 'epsActual_rolling'],
                                 list_ratios_toincrease = ['ROIC', 'NetMargin'],
                                 list_kpi_toaccelerate = ['epsActual_rolling'],
                                 list_lag_increase = [1,4,4*5],
                                 list_ratios_to_augment = ["ROIC_lag4", "ROIC_lag1", "NetMargin_lag4"],
                                 list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']) :



    # 1. Traitement du bilan comptable
    balance_clean = (
        balance[['ticker', 'date', 'filing_date', 'commonStockSharesOutstanding','totalStockholderEquity','netDebt']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
            filing_date_balance =lambda x: pd.to_datetime(x['filing_date'])
        )
        .sort_values('filing_date_balance')
        .groupby(['ticker', 'quarter_end'])
        .last()  # Prendre la dernière version du rapport pour chaque trimestre
        .reset_index()
        .drop(columns = ['filing_date'])
    )
    for columns in ['totalStockholderEquity','netDebt','commonStockSharesOutstanding'] : 
        balance_clean [f"{columns}_rolling"] =  balance_clean.sort_values(['ticker','filing_date_balance']).groupby('ticker')[columns].transform(lambda x: custom_SMA(x, n=4))

    # 2. Traitement des résultats
    earnings_clean = (
        earnings[['ticker', 'date', 'reportDate', 'epsActual']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
            filing_date_earning =lambda x: pd.to_datetime(x['reportDate'])
        )
        .sort_values('filing_date_earning')
        .groupby(['ticker', 'quarter_end'])
        .last()  # Prendre la dernière révision des résultats
        .reset_index()
        .drop(columns = ['reportDate'])
        .dropna(subset = ['epsActual'])
    )
    earnings_clean['epsActual_rolling'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsActual'] \
                                            .transform(lambda x: 4 * custom_SMA(x, n=4))
    "earnings_clean['Rolling_epsActual'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsActual'] \
                                            .transform(lambda x: 4 * custom_SMA(x, n=4))"
     # 3. Income
    columns_to_annualise_income = ['totalRevenue', 'grossProfit', 'operatingIncome', 
                       'incomeBeforeTax', 'netIncome', 'ebit', 'ebitda']
     
    income_clean = (
         income[['ticker', 'date', 'filing_date', 'totalRevenue','grossProfit','operatingIncome','incomeBeforeTax','netIncome','ebit','ebitda']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
             filing_date_income=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_income')
         .groupby(['ticker', 'quarter_end'])
         .last()  # Prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    for columns in columns_to_annualise_income : 
        income_clean[f"{columns}_rolling"] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_SMA(x, n=4))
        #income_clean[columns] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_SMA(x, n=4))
                                          
    cash_clean = (
         cashflow[['ticker', 'date', 'filing_date', 'freeCashFlow']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
             filing_date_cash=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_cash')
         .groupby(['ticker', 'quarter_end'])
         .last()  # Prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    for columns in ['freeCashFlow'] : 
        cash_clean[f"{columns}_rolling"] =  cash_clean.sort_values(['filing_date_cash']).groupby('ticker')[columns].transform(lambda x: 4 * custom_SMA(x, n=4))                                        
    """
    funda = (income_clean
             .merge(cash_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(balance_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(earnings_clean[['ticker', 'quarter_end','filing_date_earning', 'epsActual','epsActual_rolling']],on = ['ticker','quarter_end'],how = 'outer')
             .assign(NetMargin = lambda x : x['ebit_rolling']/x['totalRevenue_rolling'])
             .assign(ROIC = lambda x : x['ebit_rolling']/(x['totalStockholderEquity_rolling']+x['netDebt_rolling'].fillna(0)))
             )
    """
    funda = (income_clean
            .merge(cash_clean,on = ['ticker','quarter_end'],how = 'outer')
            .merge(balance_clean,on = ['ticker','quarter_end'],how = 'outer')
            .merge(earnings_clean[['ticker', 'quarter_end','filing_date_earning', 'epsActual','epsActual_rolling']],on = ['ticker','quarter_end'],how = 'outer')
            .assign(NetMargin = lambda x : x['ebit_rolling']/x['totalRevenue_rolling'])
            .assign(ROIC = lambda x : x['ebit_rolling']/(x['totalStockholderEquity_rolling']+x['netDebt_rolling'].fillna(0)))
            .assign(ebitpershare_rolling = lambda x : x['ebit_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
            .assign(ebitdapershare_rolling = lambda x : x['ebitda_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
            .assign(netincomepershare_rolling = lambda x : x['netIncome_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
            .assign(fcfpershare_rolling = lambda x : x['freeCashFlow_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
            .assign(eps_fcf = lambda x : x['epsActual_rolling']/(x['fcfpershare_rolling'])-1)
            .assign(eps_netincome = lambda x : x['epsActual_rolling']/(x['netincomepershare_rolling'])-1)
            )

    
    
    for col in list_kpi_toincrease:
        for lag in list_lag_increase : 
            funda = funda.astype({col: 'float'})
            funda[f"{col}_lag{lag}"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, lag, diff=False))
            #funda[f"{col}_YoY_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4, diff=False))
            #funda[f"{col}_YoY_5"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4*5, diff=False))
            #funda[f"{col}_QoQ_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 1, diff=False))

    for col in list_ratios_toincrease :
        for lag in list_lag_increase : 
            funda[f"{col}_lag{lag}"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, lag, diff=True))
            #funda[f"{col}_YoY_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4, diff=True))
            #funda[f"{col}_YoY_5"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4*5, diff=True))
            #funda[f"{col}_QoQ_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 1, diff=True))
    for col in list_kpi_toaccelerate :
        for lag in list_lag_increase : 
            funda[f"{col}_lag{lag}_lag1"] = funda.groupby('ticker')[col].transform(
                lambda x: increase(increase(x, lag, diff=False), 1, diff=True))

    funda = (funda
            .drop(columns = ['date','date_x','date_y'])
            .assign(date=funda[list_date_to_maximise].max(axis=1))
            )
    
    funda = augmenting_ratios(funda,list_ratios_to_augment , 'date')
    
    return funda
def score(x,alpha) : 
    x = x-1
    sco = np.log(1+alpha*x)
    sco = np.where(np.isnan(sco), -np.inf, sco)
    return sco
def Ranking(values, tresh):
    """Applique le ranking sur une série de valeurs."""
    # Calcul du quantile sur toute la série
    quantiles = rankdata(values.fillna(values.min() - 1), method="average") / len(values)
    
    # Remplacer les 0 par -inf pour exclusion
    quantiles = np.where(quantiles == 0, -np.inf, quantiles)
    
    # Appliquer le ranking : mettre 0 si inférieur au seuil
    ranked_values = quantiles * (np.maximum(0, quantiles - tresh) > 0).astype(int)
    
    return ranked_values
def BestModel(Data,mode,param_temp,param_alpha) : 
    
    Data['Score'] = score(Data['Return'],alpha = param_alpha)
    def Best_Model_Date(Data,date, mode,param_alpha, param_temp):
    
        filtered_data = Data[Data['Month'] < date]
        """
        grouped_data = filtered_data.sort_values(by='Month', ascending=False) \
                                .groupby(['Model'])
        summarized_data = (grouped_data['Score']
                           .apply(decreasing_sum, halfperiod=param_temp, mode =  mode) 
                           .reset_index(name='Score'))
        """
        summarized_data  = (filtered_data
                .sort_values(by='Month', ascending=False) 
                .groupby(['Model'], group_keys=False)
                .apply(lambda x: decreasing_sum(x['Score'], halfperiod=param_temp, mode=mode),include_groups = False)
                .rename('Score')  # Renomme la colonne résultante en 'Score'
                .reset_index(name='Score') 
                )
       
        best_model = (summarized_data[summarized_data['Score'] == summarized_data['Score'].max()].sample(1)
                      .assign(
                              Month=date,
                              param_alpha=param_alpha,
                              param_temp=param_temp
                              )
                      )
        return best_model
    
    results = []
    Data = Data.sort_values(['Month'])
    List_Date = Data['Month'].unique().tolist()
    List_Date.append(pd.to_datetime(datetime.now()).to_period('M') + 1)
    List_Date = List_Date[1:]
    for date_loop in List_Date :
        
            results_loop = Best_Model_Date(Data = Data,
                                           date = date_loop,
                                           mode = mode,
                                           param_temp = param_temp,
                                           param_alpha = param_alpha)
        
            results.append(results_loop )
    
    results = pd.concat(results, ignore_index=True) 
    return results
def calculate_monthly_returns(df):
    # Étape 1 : Trouver la dernière ligne pour chaque (ticker, year_month)
    last_rows = df.groupby(['ticker', 'year_month']).apply(lambda x: x.loc[x['date'].idxmax()],include_groups = False)
    
    # Étape 2 : Extraire 'last_close' et 'adjusted_close_last'
    last_closes_df = last_rows[['date','close', 'adjusted_close']].reset_index().rename(columns={'close': 'last_close', 'adjusted_close': 'adjusted_close_last'})
    
    # Étape 3 : Trier par 'ticker' et 'year_month'
    last_closes_df = last_closes_df.sort_values(by=['ticker', 'year_month'])
    
    # Étape 4 : Calculer le rendement mensuel basé sur 'adjusted_close_last'
    last_closes_df['monthly_return'] = last_closes_df.groupby('ticker')['adjusted_close_last'].transform(lambda x: x / x.shift(1))
    
    # Étape 5 : Sélectionner les colonnes pertinentes
    final_df = last_closes_df[['ticker','date', 'year_month', 'last_close', 'monthly_return']]
    
    return final_df    
def learning_process_technical(Prices,Historical_Company,Index_Price,Stocks_Filter,Sector,func_MovingAverage,Liste_NLong,Liste_NShort,Liste_NAsset,Max_PerSector,Final_NMaxAsset,List_Alpha,List_Temp,mode,param_temp_Lvl2,param_alpha_Lvl2) : 
    
    Parameters = pd.DataFrame([(n_long, n_short, n_asset) 
                           for n_long in Liste_NLong 
                           for n_short in Liste_NShort 
                           for n_asset in Liste_NAsset 
                           if n_long > 3 * n_short],
    columns=['N_Long', 'N_Short', 'N_Asset'])
    All_Return_Monthly = pd.DataFrame()
    All_Detaillled_Portfolios = pd.DataFrame()
    Prices = Prices.dropna(subset=['Close_VS_SP500']).sort_values(by=['ticker', 'date'])

    print("Start backtest level 0")
    for i in tqdm(range(len(Parameters)), desc="Processing"):
        row = Parameters.iloc[i]
        result = Learning_Monthly(
            Data = Prices.copy(),
            Historical_Company = Historical_Company,
            N_Long=row['N_Long'],
            N_Short=row['N_Short'],
            func_MovingAverage=func_MovingAverage,  
            N_Asset=row['N_Asset'])
        All_Return_Monthly = pd.concat([All_Return_Monthly, result[0]], ignore_index=True)
        All_Detaillled_Portfolios = pd.concat([All_Detaillled_Portfolios, result[1]], ignore_index=True)
    
    print("End of  backtest level 0")
    All_Return_Monthly_AfterSelection =  (All_Return_Monthly
                                            .merge(Stocks_Filter[['year_month','ticker']],
                                                    how = "inner",
                                                    left_on = ['Month','ticker'],
                                                    right_on = ['year_month','ticker'])
                                            .merge(Sector,on = "ticker",how = "left")
                                            .sort_values('quantile_MTR', ascending=False)
                                            .groupby(['Month', 'N_Long', 'N_Short', 'N_Asset', 'Sector'], group_keys=False)
                                            .apply(lambda g: g.head(Max_PerSector))
                                            .sort_values('quantile_MTR', ascending=False)
                                            .groupby(['Month', 'N_Long', 'N_Short', 'N_Asset'], group_keys=False)
                                            .apply(lambda g: g.head(Final_NMaxAsset)))
    
    All_Return_Monthly_AfterSelection['Model'] = All_Return_Monthly_AfterSelection['N_Long'].astype(str) + "-"+ All_Return_Monthly_AfterSelection['N_Short'].astype(str) +"-"+All_Return_Monthly_AfterSelection['N_Asset'].astype(str) 
    All_Return_Monthly_AfterSelection_Summarised  = (
        All_Return_Monthly_AfterSelection
        .groupby(['Month', 'Model'], as_index=False)
        .agg(
            DR=('DR', 'mean'), #We suppose equiponderation
            N=('DR', 'size')
            )
        .merge(
            Index_Price[['Month', 'DR_SP500']], 
            on='Month', 
            how='left'
                )
        .assign(Return=lambda x: x['DR'] / x['DR_SP500'])
                                            )
    Lvl1_BestModel = []
    print("Start learning lvl 1")
    for alpha in tqdm(List_Alpha):
        for temp in List_Temp :
            A = BestModel(
                Data = All_Return_Monthly_AfterSelection_Summarised,
                mode = mode,
                param_temp = temp,
                param_alpha = alpha)
            Lvl1_BestModel.append(A)
    print("End of  learning lvl 1")       
    Lvl1_BestModel = (pd.concat(Lvl1_BestModel, ignore_index=True)
                        .rename(columns = {"Model" : "Model_Lvl0"})
                        .assign(Model_Lvl1 = lambda x: x['param_alpha'].astype(str)+"-"+x['param_temp'].astype(str))
                        .drop(columns=['Score','param_alpha','param_temp'])
                        )
    Lvl1_Return = (All_Return_Monthly_AfterSelection_Summarised[['Model','Month','Return']]
                    .merge(Lvl1_BestModel,
                            left_on = ["Model","Month"],
                            right_on = ["Model_Lvl0","Month"],
                            how = "inner")
                    .drop(columns = ['Model'])
                    .rename(columns = {"Model_Lvl1" : "Model"})
                    )
    
    Lvl2_BestModel  = (BestModel(Data = Lvl1_Return,
                            mode = mode,
                            param_temp = param_temp_Lvl2,
                            param_alpha = param_alpha_Lvl2)[['Month','Model']]
                        .rename(columns = {"Model" : "Model_Lvl1"})
                        .merge(Lvl1_BestModel,
                            how = "inner",
                            on = ["Month","Model_Lvl1"])
                        .merge(All_Return_Monthly_AfterSelection_Summarised[['Month', 'Model', 'DR','DR_SP500','Return']],
                            how = "inner",
                            left_on = ['Month','Model_Lvl0'],
                            right_on = ['Month','Model'])
                        .drop(columns = ['Model']))
    
    Detail = (Lvl2_BestModel[['Month','Model_Lvl0','Model_Lvl1']]
                .merge(All_Return_Monthly_AfterSelection[['Month','ticker','Model','Sector','DR']],
                        left_on = ['Month','Model_Lvl0'],
                        right_on = ['Month','Model'])
                .drop(columns = ['Model'])
                )
    
    Detail[Detail['Month'] == max(Detail['Month'])][['Month','Model_Lvl0']].drop_duplicates()
    All_Historical_Component =  (Detail[Detail['Month'] == max(Detail['Month'])][['Month','Model_Lvl0']].drop_duplicates()
                                 .merge(All_Detaillled_Portfolios
                                        .assign(Model_Lvl0 = lambda x : x['N_Long'].astype(str) + "-"+ x['N_Short'].astype(str) +"-"+x['N_Asset'].astype(str)),
                                        on = ['Month','Model_Lvl0'],
                                        how = 'inner')
                                            .merge(Stocks_Filter[['year_month','ticker']],
                                                   how = "inner",
                                                   left_on = ['Month','ticker'],
                                                   right_on = ['year_month','ticker'])
                                            .merge(Sector,on = "ticker",how = "left")
                                            .sort_values('quantile_MTR', ascending=False)
                                            .groupby(['Month', 'N_Long', 'N_Short', 'N_Asset', 'Sector'], group_keys=False)
                                            .apply(lambda g: g.head(Max_PerSector))
                                            .sort_values('quantile_MTR', ascending=False)
                                            .groupby(['Month', 'N_Long', 'N_Short', 'N_Asset'], group_keys=False)
                                            .apply(lambda g: g.head(Final_NMaxAsset)))
    
    
    return Lvl2_BestModel  , Detail,All_Historical_Component [['date','ticker','N_Long', 'N_Short', 'N_Asset','MTR','Sector']]
def learning_fundamental(balance,cashflow,income,earnings, general,monthly_return,Historical_Company,col_learning,tresh,n_max_sector,
                         list_kpi_toinvert = ['PE'],
                         list_kpi_toincrease = ['totalRevenue_rolling', 'grossProfit_rolling', 'operatingIncome_rolling', 'incomeBeforeTax_rolling', 'netIncome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freeCashFlow_rolling', 'epsActual_rolling'],
                         list_ratios_toincrease = ['ROIC', 'NetMargin'],
                         list_kpi_toaccelerate = ['epsActual_rolling'],
                         list_lag_increase = [1,4,4*5],
                         list_ratios_to_augment = ["ROIC_lag4", "ROIC_lag1", "NetMargin_lag4"],
                         list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']) : 
    
    

    
    Ratios = calculate_fundamental_ratios(balance = balance,
                                          cashflow = cashflow,
                                          income = income,
                                          earnings = earnings,
                                          list_kpi_toincrease = list_kpi_toincrease,
                                          list_ratios_toincrease = list_ratios_toincrease,
                                          list_kpi_toaccelerate = list_kpi_toaccelerate,
                                          list_lag_increase = list_lag_increase,
                                          list_ratios_to_augment = list_ratios_to_augment,
                                          list_date_to_maximise = list_date_to_maximise)
    PE = calculate_pe_ratios(balance = balance, 
                             earnings = earnings, 
                             monthly_return = monthly_return)
    Ratios['year_month'] = Ratios['date'].dt.to_period('M')
    final_merged = []
    list_date_loop = sorted(Ratios[Ratios['year_month'] >= '2000-01'].dropna(subset = ['year_month'])['year_month'].unique())
    list_date_loop.append(max(list_date_loop) + 1)
    for date_loop in tqdm(list_date_loop) : 
      if not(pd.isna(date_loop)) : 
            
       
            
        Historical_Company_Loop = Historical_Company[Historical_Company ['Month'] < date_loop]
        Historical_Company_Loop  = Historical_Company_Loop[Historical_Company_Loop['Month'] == Historical_Company_Loop['Month'].max()]['ticker'].unique()
        
        
        Ratios_Loop = Ratios[Ratios['year_month'] < date_loop]
        Ratios_Loop = Ratios_Loop[(Ratios_Loop['date'] == Ratios_Loop.groupby('ticker')['date'].transform('max')) & 
                                  (Ratios_Loop['ticker'].isin(Historical_Company_Loop))]
        Ratios_Loop['date_diff_ratios'] = ((date_loop-1).to_timestamp(how='end') - pd.to_datetime(Ratios_Loop['date']) ).dt.days
      
        PE_Loop = PE[PE['year_month'] < date_loop]
        PE_Loop = PE_Loop[(PE_Loop['year_month'] == PE_Loop.groupby('ticker')['year_month'].transform('max')) & 
                                  (PE_Loop['ticker'].isin(Historical_Company_Loop))]
        PE_Loop['date_diff_PE'] = ((date_loop-1).to_timestamp(how='end') - pd.to_datetime(PE_Loop['year_month'].dt.to_timestamp(how='end'))).dt.days
        
        Merge_Loop = PE_Loop.merge(Ratios_Loop,on = 'ticker')
        for kpi_toinvert in list_kpi_toinvert : 
            Merge_Loop[f"{kpi_toinvert }_inverted"] = 1/(Merge_Loop[kpi_toinvert]+0.001)

        Merge_Loop = Merge_Loop[['ticker']+col_learning]
        for c in col_learning :
                Merge_Loop[f"{c}_quantile"] = Ranking(Merge_Loop[c],tresh)
        Merge_Loop['rank'] = Merge_Loop.filter(regex='_quantile$').prod(axis=1)
        Merge_Loop = (
            Merge_Loop.merge(general,on = 'ticker')
            .sort_values(by='rank', ascending=False)
            .assign(one=1)
            .assign(one=lambda x: x.groupby('Sector')['one'].cumsum())
            .loc[lambda x: x['one'] <= n_max_sector]
            .drop(columns='one')
            .loc[lambda x: x['rank'] > 0]  
            .assign(year_month = date_loop)
            )
        final_merged.append(Merge_Loop)

    # Concaténer tous les résultats de Merge_Loop
    final_result = pd.concat(final_merged, ignore_index=True)
    
    return_model = final_result.merge(monthly_return,
                       how = 'left',
                       on = ['ticker','year_month'])
    
    result_summarised = return_model.groupby(['year_month']).agg(
                            monthly_return=('monthly_return', 'mean'),
                            N =('monthly_return', 'count')).reset_index()
    
    result_summarised_yearly = result_summarised .dropna(subset= ['monthly_return'])
    result_summarised_yearly['year'] = result_summarised_yearly['year_month'].dt.year

    result = pd.DataFrame()
    for year in result_summarised_yearly['year'].unique() : 
        Full_Monthly_Return_Loop = (result_summarised_yearly[result_summarised_yearly['year']>= year]
                                    .agg(Return=('monthly_return', lambda x: np.prod(x)**(12/len(x))-1),
                                         Worst=('monthly_return', 'min'),
                                         Vol=('monthly_return', 'std'),
                                         N_Min = ('N','min'),
                                         N_Max = ('N','max'),
                                         N_Mean = ('N','mean'))
                                .reset_index()
                                .assign(Year = year))
        result = pd.concat([result, Full_Monthly_Return_Loop])
    result = result.pivot(index = 'Year',columns = 'index',values = ['monthly_return','N']).dropna(axis = 1).reset_index()
    

    return return_model , result_summarised ,result         
def return_benchmark(Prices,Historical_Company,Index_Price,Stocks_Filter,Sector) : 
    
    Prices_DR = (Prices
                   .sort_values('Month')  # Trier par date
                   .groupby(['ticker'])
                   .apply(lambda x: x.assign(DR=x['Close'] / x['Close'].shift(1)), include_groups=False)
                   .reset_index()
                   )
    Benchmark_Base = (Prices_DR
              .merge(Historical_Company,
                     how="inner",
                     on=['Month', 'ticker'])
              .groupby(['ticker', 'Month'])
              .agg({'DR': 'prod'})
              .reset_index()
              .groupby(['Month'])
              .agg({'DR': 'mean'})
              .reset_index()
              .assign(Model = 'Base')
             )
    
    Benchmark_After_Selection =  (Prices_DR
                                          .merge(Stocks_Filter[['year_month','ticker']],
                                                 how = "inner",
                                                 left_on = ['Month','ticker'],
                                                 right_on = ['year_month','ticker'])
                                          .merge(Sector,on = "ticker",how = "left")
                                          .groupby(['ticker', 'Month'])
                                          .agg({'DR': 'prod'})
                
                                          .reset_index()
                                          .groupby(['Month'])
                                          .agg({'DR': 'mean'})
                                          .reset_index()
                                          .assign(Model = 'After_Selection')
                                         )
    Bench = pd.concat([Benchmark_Base,
                       Benchmark_After_Selection,
                       (Index_Price
                        .rename(columns = {"DR_SP500" : "DR"})
                        .assign(Model = 'Index'))  ])

    return Bench

def scoping_fundamental(balance,
                        cashflow,
                        income,
                        earnings,
                        list_kpi_toincrease = ['totalRevenue_rolling', 'grossProfit_rolling', 'operatingIncome_rolling', 'incomeBeforeTax_rolling', 'netIncome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freeCashFlow_rolling', 'epsActual_rolling'],
                                 list_ratios_toincrease = ['ROIC', 'NetMargin'],
                                 list_kpi_toaccelerate = ['epsActual_rolling'],
                                 list_lag_increase = [1,4,4*5],
                                 list_ratios_to_augment = ["ROIC_lag4", "ROIC_lag1", "NetMargin_lag4"],
                                 list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']) :



    # 1. Traitement du bilan comptable
    balance_clean = (
        balance[['ticker', 'date', 'filing_date', 'commonStockSharesOutstanding','totalStockholderEquity','netDebt']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
            filing_date_balance =lambda x: pd.to_datetime(x['filing_date'])
        )
        .sort_values('filing_date_balance')
        .groupby(['ticker', 'quarter_end'])
        .last()  # Prendre la dernière version du rapport pour chaque trimestre
        .reset_index()
        .drop(columns = ['filing_date'])
    )
    for columns in ['totalStockholderEquity','netDebt','commonStockSharesOutstanding'] : 
        balance_clean [f"{columns}_rolling"] =  balance_clean.sort_values(['ticker','filing_date_balance']).groupby('ticker')[columns].transform(lambda x: custom_SMA(x, n=4))

    # 2. Traitement des résultats
    earnings_clean = (
        earnings[['ticker', 'date', 'reportDate', 'epsActual']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
            filing_date_earning =lambda x: pd.to_datetime(x['reportDate'])
        )
        .sort_values('filing_date_earning')
        .groupby(['ticker', 'quarter_end'])
        .last()  # Prendre la dernière révision des résultats
        .reset_index()
        .drop(columns = ['reportDate'])
        .dropna(subset = ['epsActual'])
    )
    earnings_clean['epsActual_rolling'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsActual'] \
                                            .transform(lambda x: 4 * custom_SMA(x, n=4))
    "earnings_clean['Rolling_epsActual'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsActual'] \
                                            .transform(lambda x: 4 * custom_SMA(x, n=4))"
     # 3. Income
    columns_to_annualise_income = ['totalRevenue', 'grossProfit', 'operatingIncome', 
                       'incomeBeforeTax', 'netIncome', 'ebit', 'ebitda']
     
    income_clean = (
         income[['ticker', 'date', 'filing_date', 'totalRevenue','grossProfit','operatingIncome','incomeBeforeTax','netIncome','ebit','ebitda']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
             filing_date_income=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_income')
         .groupby(['ticker', 'quarter_end'])
         .last()  # Prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    for columns in columns_to_annualise_income : 
        income_clean[f"{columns}_rolling"] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_SMA(x, n=4))
        #income_clean[columns] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_SMA(x, n=4))
                                          
    cash_clean = (
         cashflow[['ticker', 'date', 'filing_date', 'freeCashFlow']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
             filing_date_cash=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_cash')
         .groupby(['ticker', 'quarter_end'])
         .last()  # Prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    
    for columns in ['freeCashFlow'] : 
        cash_clean[f"{columns}_rolling"] =  cash_clean.sort_values(['filing_date_cash']).groupby('ticker')[columns].transform(lambda x: 4 * custom_SMA(x, n=4))                                        

    funda = (funda
            .drop(columns = ['date','date_x','date_y'])
            .assign(date=funda[list_date_to_maximise].max(axis=1))
            )
    funda = (income_clean
             .merge(cash_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(balance_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(earnings_clean[['ticker', 'quarter_end','filing_date_earning', 'epsActual','epsActual_rolling']],on = ['ticker','quarter_end'],how = 'outer')
             .assign(NetMargin = lambda x : x['ebit_rolling']/x['totalRevenue_rolling'])
             .assign(ROIC = lambda x : x['ebit_rolling']/(x['totalStockholderEquity_rolling']+x['netDebt_rolling'].fillna(0)))
             .assign(ebitpershare_rolling = lambda x : x['ebit_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
             .assign(ebitdapershare_rolling = lambda x : x['ebitda_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
             .assign(netincomepershare_rolling = lambda x : x['netIncome_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
             .assign(fcfpershare_rolling = lambda x : x['freeCashFlow_rolling']/(x['commonStockSharesOutstanding_rolling'].fillna(0)))
             )
    view = funda[['date','ticker','epsActual_rolling',
                  'ebitpershare_rolling','ebitdapershare_rolling',
                  'netincomepershare_rolling','fcfpershare_rolling']]
    
def clean_financial_data(funda):
    """
    Nettoie et prépare les données financières pour l'analyse
    
    Args:
        funda: DataFrame contenant les données financières brutes
        
    Returns:
        DataFrame: Données financières nettoyées et préparées
    """
    # Création d'une copie pour éviter de modifier les données originales
    df = funda.copy()
    
    # Conversion des colonnes object numériques en float
    numeric_object_columns = [
        'totalRevenue', 'grossProfit', 'operatingIncome', 'incomeBeforeTax',
        'netIncome', 'ebit', 'ebitda', 'freeCashFlow',
        'commonStockSharesOutstanding', 'totalStockholderEquity', 'netDebt'
    ]
    
    for col in numeric_object_columns:
        if col in df.columns:
            # Remplacement des valeurs non numériques par NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Conversion des colonnes de date en datetime si elles sont de type object
    date_columns = ['date_x', 'date_y', 'date']
    for col in date_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Conversion de quarter_end en datetime pour faciliter l'analyse temporelle
    # Préservation du format period[Q-DEC] dans une colonne séparée
    if 'quarter_end' in df.columns and df['quarter_end'].dtype == 'period[Q-DEC]':
        df['quarter_end_period'] = df['quarter_end']
        df['quarter_end'] = df['quarter_end'].dt.to_timestamp()
    
    # Création d'une colonne année-trimestre textuelle pour l'analyse
    df['year_quarter_str'] = df['quarter_end'].dt.strftime('%Y-Q%q')
    
    # Vérification de la présence des colonnes essentielles pour l'analyse
    essential_columns = [
        'epsActual_rolling', 'netincomepershare_rolling', 'fcfpershare_rolling',
        'netIncome_rolling', 'freeCashFlow_rolling'
    ]
    
    missing_cols = [col for col in essential_columns if col not in df.columns]
    if missing_cols:
        warnings.warn(f"Colonnes essentielles manquantes: {missing_cols}")
    
    # Filtrage des lignes avec des valeurs manquantes dans les colonnes cruciales
    # Nous gardons toutes les lignes mais ajoutons des flags pour identifier les problèmes
    df['has_eps_data'] = ~df['epsActual_rolling'].isna()
    df['has_netincome_data'] = ~df['netIncome_rolling'].isna()
    df['has_fcf_data'] = ~df['freeCashFlow_rolling'].isna()
    df['has_shares_data'] = ~df['commonStockSharesOutstanding_rolling'].isna()
    
    # Calcul des métriques de qualité de données par ticker
    ticker_data_quality = df.groupby('ticker')[
        ['has_eps_data', 'has_netincome_data', 'has_fcf_data', 'has_shares_data']
    ].mean()
    
    # Ajout d'un score global de qualité des données
    ticker_data_quality['data_quality_score'] = ticker_data_quality.mean(axis=1)
    
    # Fusion du score de qualité avec le DataFrame principal
    df = df.merge(
        ticker_data_quality['data_quality_score'].rename('ticker_data_quality'),
        left_on='ticker',
        right_index=True
    )
    
    # Calcul de ratios supplémentaires pour l'analyse
    # Avec gestion des divisions par zéro
    df['fcf_to_netincome_ratio'] = np.where(
        (df['netIncome_rolling'] != 0) & (~df['netIncome_rolling'].isna()) & (~df['freeCashFlow_rolling'].isna()),
        df['freeCashFlow_rolling'] / df['netIncome_rolling'],
        np.nan
    )
    
    df['eps_to_netincomepershare_ratio'] = np.where(
        (df['netincomepershare_rolling'] != 0) & (~df['netincomepershare_rolling'].isna()) & (~df['epsActual_rolling'].isna()),
        df['epsActual_rolling'] / df['netincomepershare_rolling'],
        np.nan
    )
    
    # Identification des valeurs aberrantes dans les ratios
    df['fcf_netincome_anomaly'] = np.abs(df['fcf_to_netincome_ratio'] - 1) > 0.3
    df['eps_netincome_anomaly'] = np.abs(df['eps_to_netincomepershare_ratio'] - 1) > 0.2
    
    return df

def analyze_financial_ratios(funda, min_quarters=4, top_n_tickers=None):
    """
    Analyse approfondie des ratios financiers par ticker avec focus sur les écarts
    entre EPS rolling et les autres métriques par action.
    
    Args:
        funda: DataFrame contenant les données financières
        min_quarters: Nombre minimum de trimestres requis pour l'analyse (défaut: 4)
        top_n_tickers: Limiter l'analyse aux N tickers avec le plus de données (défaut: None = tous)
        
    Returns:
        dict: Dictionnaire contenant les résultats d'analyse par ticker
    """
    # Nettoyage des données
    print("Nettoyage et préparation des données...")
    df = clean_financial_data(funda)
    
    # Affichage des statistiques sur les données manquantes
    missing_data = df.isnull().sum()
    print("\nDonnées manquantes par colonne:")
    print(missing_data[missing_data > 0])
    
    # Sélection des tickers avec suffisamment de données
    ticker_counts = df.groupby('ticker').size()
    valid_tickers = ticker_counts[ticker_counts >= min_quarters].index.tolist()
    
    if top_n_tickers:
        # Sélection des N tickers avec le plus de données
        ticker_quality = df.groupby('ticker')['ticker_data_quality'].mean()
        valid_tickers = ticker_quality.loc[valid_tickers].nlargest(top_n_tickers).index.tolist()
    
    print(f"\nAnalyse de {len(valid_tickers)} tickers avec au moins {min_quarters} trimestres de données")
    
    # Initialisation du dictionnaire des résultats
    results = {}
    
    # Définition des métriques clés pour l'analyse
    key_metrics = [
        'epsActual_rolling', 'netincomepershare_rolling', 'fcfpershare_rolling', 
        'ebitpershare_rolling', 'ebitdapershare_rolling', 'fcf_to_netincome_ratio', 
        'eps_to_netincomepershare_ratio'
    ]
    
    # Analyse par ticker
    for ticker in valid_tickers:
        print(f"Analyse du ticker: {ticker}")
        ticker_data = df[df['ticker'] == ticker].sort_values('quarter_end')
        
        # Statistiques descriptives des ratios clés
        valid_metrics = [m for m in key_metrics if not ticker_data[m].isna().all()]
        
        if not valid_metrics:
            print(f"  Aucune métrique valide pour {ticker}, ticker ignoré")
            continue
        
        stats_df = ticker_data[valid_metrics].describe().T
        
        # Calcul des corrélations entre les métriques
        corr_matrix = ticker_data[valid_metrics].corr()
        
        # Analyse de la cohérence des métriques au fil du temps
        if 'epsActual_rolling' in valid_metrics and 'netincomepershare_rolling' in valid_metrics:
            valid_rows = ticker_data.dropna(subset=['epsActual_rolling', 'netincomepershare_rolling'])
            if len(valid_rows) >= 3:
                eps_vs_netincome = valid_rows['epsActual_rolling'] / valid_rows['netincomepershare_rolling']
                consistency_score = 1 - (eps_vs_netincome.std() / eps_vs_netincome.mean() if eps_vs_netincome.mean() != 0 else 0)
            else:
                consistency_score = np.nan
        else:
            consistency_score = np.nan
        
        # Détection des tendances (croissance, stabilité, déclin)
        trend_analysis = {}
        for metric in valid_metrics:
            series = ticker_data[metric].dropna()
            if len(series) >= 6:  # Au moins 6 points pour une analyse de tendance fiable
                x = np.arange(len(series))
                y = series.values
                slope, _, r_value, p_value, _ = stats.linregress(x, y)
                trend_analysis[metric] = {
                    'slope': slope,
                    'r_squared': r_value**2,
                    'p_value': p_value,
                    'trend': 'Croissance' if slope > 0 and p_value < 0.05 else 
                             'Déclin' if slope < 0 and p_value < 0.05 else 'Stable',
                    'annualized_growth': ((1 + slope)**(4/len(series)) - 1) * 100 if slope > 0 else 
                                         ((1 + slope)**(4/len(series)) - 1) * 100
                }
        
        # Calcul de la saisonnalité des métriques (si applicable)
        seasonality = {}
        for metric in valid_metrics:
            series = ticker_data[metric].dropna()
            if len(series) >= 8:  # Au moins 8 points pour détecter la saisonnalité (2 ans)
                try:
                    result = seasonal_decompose(series, model='additive', period=4)
                    seasonality[metric] = {
                        'seasonal_strength': np.std(result.seasonal) / np.std(result.trend),
                        'has_seasonality': np.std(result.seasonal) / np.std(result.trend) > 0.1
                    }
                except:
                    pass
        
        # Identification des anomalies spécifiques
        anomalies = ticker_data[ticker_data['fcf_netincome_anomaly'] | ticker_data['eps_netincome_anomaly']]
        
        # Calcul du score de qualité des données financières
        quality_metrics = {
            'eps_netincome_consistency': consistency_score,
            'data_completeness': ticker_data['ticker_data_quality'].mean(),
            'anomaly_percentage': len(anomalies) / len(ticker_data) if len(ticker_data) > 0 else 0
        }
        
        # Calcul du FCF vs Net Income alignment si les deux métriques sont disponibles
        if 'fcfpershare_rolling' in valid_metrics and 'netincomepershare_rolling' in valid_metrics:
            valid_rows = ticker_data.dropna(subset=['fcfpershare_rolling', 'netincomepershare_rolling'])
            if len(valid_rows) >= 3:
                fcf_ni_corr = valid_rows['fcfpershare_rolling'].corr(valid_rows['netincomepershare_rolling'])
                quality_metrics['fcf_netincome_alignment'] = max(0, fcf_ni_corr)
            else:
                quality_metrics['fcf_netincome_alignment'] = np.nan
        else:
            quality_metrics['fcf_netincome_alignment'] = np.nan
        
        # Calcul du score de qualité global (avec gestion des NaN)
        valid_scores = [v for k, v in quality_metrics.items() if not np.isnan(v)]
        quality_score = np.mean(valid_scores) if valid_scores else np.nan
        
        # Classification des trimestres par performance
        performance_classification = {}
        if len(ticker_data) >= min_quarters:
            perf_metrics = [m for m in ['epsActual_rolling', 'netincomepershare_rolling', 'fcfpershare_rolling'] 
                          if m in valid_metrics]
            
            if len(perf_metrics) >= 2:  # Au moins 2 métriques disponibles
                try:
                    performance_data = ticker_data[perf_metrics].dropna()
                    if len(performance_data) >= 3:  # Au moins 3 points pour le clustering
                        # Normalisation des données
                        perf_normalized = (performance_data - performance_data.mean()) / performance_data.std()
                        kmeans = KMeans(n_clusters=min(3, len(performance_data)), random_state=42)
                        performance_data['cluster'] = kmeans.fit_predict(perf_normalized)
                        
                        # Caractérisation des clusters
                        cluster_means = performance_data.groupby('cluster')[perf_metrics].mean()
                        
                        # Détermination du meilleur et du pire cluster
                        cluster_performance = cluster_means.mean(axis=1)
                        best_cluster = cluster_performance.idxmax()
                        worst_cluster = cluster_performance.idxmin()
                        
                        # Mapping clusters aux périodes
                        best_periods = ticker_data.loc[performance_data[performance_data['cluster'] == best_cluster].index, 'year_quarter_str'].tolist()
                        worst_periods = ticker_data.loc[performance_data[performance_data['cluster'] == worst_cluster].index, 'year_quarter_str'].tolist()
                        
                        performance_classification = {
                            'best_periods': best_periods,
                            'worst_periods': worst_periods,
                            'cluster_profiles': cluster_means.to_dict()
                        }
                except Exception as e:
                    performance_classification = f"Erreur lors de la classification: {str(e)}"
        
        # Analyse comparative des différents ratios par action
        ratio_comparisons = {}
        ratio_pairs = [
            ('epsActual_rolling', 'netincomepershare_rolling', 'EPS vs Net Income/Share'),
            ('fcfpershare_rolling', 'netincomepershare_rolling', 'FCF/Share vs Net Income/Share'),
            ('ebitpershare_rolling', 'netincomepershare_rolling', 'EBIT/Share vs Net Income/Share'),
            ('ebitdapershare_rolling', 'netincomepershare_rolling', 'EBITDA/Share vs Net Income/Share'),
            ('fcfpershare_rolling', 'epsActual_rolling', 'FCF/Share vs EPS')
        ]
        
        for ratio1, ratio2, name in ratio_pairs:
            if ratio1 in valid_metrics and ratio2 in valid_metrics:
                valid_data = ticker_data[[ratio1, ratio2]].dropna()
                if len(valid_data) >= 3:
                    ratio_values = valid_data[ratio1] / valid_data[ratio2]
                    mean_ratio = ratio_values.mean()
                    std_ratio = ratio_values.std()
                    correlation = valid_data[ratio1].corr(valid_data[ratio2])
                    
                    ratio_comparisons[name] = {
                        'mean_ratio': mean_ratio,
                        'std_ratio': std_ratio,
                        'correlation': correlation,
                        'consistency': 1 - (std_ratio / mean_ratio) if mean_ratio != 0 else 0
                    }
        
        # Recommandations basées sur l'analyse
        recommendations = []
        
        # Recommandation sur la cohérence EPS vs Net Income
        if 'EPS vs Net Income/Share' in ratio_comparisons:
            eps_ni_ratio = ratio_comparisons['EPS vs Net Income/Share']['mean_ratio']
            eps_ni_consistency = ratio_comparisons['EPS vs Net Income/Share']['consistency']
            
            if eps_ni_ratio < 0.9:
                recommendations.append(f"L'EPS est significativement inférieur au revenu net par action (ratio: {eps_ni_ratio:.2f}). "
                                      "Vérifier les ajustements comptables ou les actions diluées.")
            elif eps_ni_ratio > 1.1:
                recommendations.append(f"L'EPS est significativement supérieur au revenu net par action (ratio: {eps_ni_ratio:.2f}). "
                                      "Vérifier les programmes de rachat d'actions ou les ajustements exceptionnels.")
                
            if eps_ni_consistency < 0.7 and not np.isnan(eps_ni_consistency):
                recommendations.append(f"Faible cohérence entre EPS et revenu net par action (score: {eps_ni_consistency:.2f}). "
                                      "Examiner les ajustements non récurrents ou les changements dans les actions en circulation.")
            
        # Recommandation sur l'écart FCF vs Net Income
        if 'FCF/Share vs Net Income/Share' in ratio_comparisons:
            fcf_ni_ratio = ratio_comparisons['FCF/Share vs Net Income/Share']['mean_ratio']
            if fcf_ni_ratio < 0.7:
                recommendations.append(f"Le FCF par action est significativement inférieur au revenu net par action (ratio: {fcf_ni_ratio:.2f}). "
                                      "Examiner les investissements en capital ou les changements dans le fonds de roulement.")
            elif fcf_ni_ratio > 1.3:
                recommendations.append(f"Le FCF par action est significativement supérieur au revenu net par action (ratio: {fcf_ni_ratio:.2f}). "
                                      "Cela peut indiquer une bonne gestion du fonds de roulement ou des charges non monétaires élevées.")
        
        # Recommandation basée sur les tendances
        for metric, analysis in trend_analysis.items():
            if analysis['trend'] == 'Déclin' and analysis['r_squared'] > 0.6:
                recommendations.append(f"Tendance à la baisse significative pour {metric} (R²: {analysis['r_squared']:.2f}, "
                                      f"croissance annualisée: {analysis['annualized_growth']:.1f}%). "
                                      "Examiner les causes sous-jacentes.")
            elif analysis['trend'] == 'Croissance' and analysis['r_squared'] > 0.6:
                recommendations.append(f"Tendance à la hausse significative pour {metric} (R²: {analysis['r_squared']:.2f}, "
                                      f"croissance annualisée: {analysis['annualized_growth']:.1f}%). "
                                      "Vérifier la durabilité de cette croissance.")
        
        # Compilation des résultats pour ce ticker
        results[ticker] = {
            'statistics': stats_df.to_dict(),
            'correlations': corr_matrix.to_dict(),
            'trend_analysis': trend_analysis,
            'seasonality': seasonality,
            'anomalies': {
                'count': len(anomalies),
                'percentage': (len(anomalies) / len(ticker_data)) * 100 if len(ticker_data) > 0 else 0,
                'periods': anomalies['year_quarter_str'].tolist() if len(anomalies) > 0 else [],
                'details': anomalies[['year_quarter_str', 'fcf_to_netincome_ratio', 'eps_to_netincomepershare_ratio']].to_dict('records') if len(anomalies) > 0 else []
            },
            'quality_metrics': quality_metrics,
            'quality_score': quality_score,
            'quality_assessment': 'Excellente' if quality_score > 0.8 else
                                'Bonne' if quality_score > 0.6 else
                                'Moyenne' if quality_score > 0.4 else
                                'Faible' if quality_score > 0.2 else 'Très faible',
            'performance_classification': performance_classification,
            'ratio_comparisons': ratio_comparisons,
            'recommendations': recommendations,
            'data_completeness': {
                'total_quarters': len(ticker_data),
                'quarters_with_eps': ticker_data['epsActual_rolling'].notna().sum(),
                'quarters_with_netincome': ticker_data['netIncome_rolling'].notna().sum(),
                'quarters_with_fcf': ticker_data['freeCashFlow_rolling'].notna().sum()
            }
        }
    
    return results

def visualize_ticker_comparison(funda, tickers=None, metrics=None, start_date=None, end_date=None):
    """
    Génère des visualisations comparatives pour les tickers spécifiés
    
    Args:
        funda: DataFrame contenant les données financières
        tickers: Liste des tickers à comparer (None = tous, max 10)
        metrics: Liste des métriques à visualiser (None = métriques par défaut)
        start_date: Date de début pour l'analyse (None = toutes les dates)
        end_date: Date de fin pour l'analyse (None = toutes les dates)
    """
    # Nettoyage des données
    df = clean_financial_data(funda)
    
    # Filtrage par date si spécifié
    if start_date:
        start_date = pd.to_datetime(start_date)
        df = df[df['quarter_end'] >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        df = df[df['quarter_end'] <= end_date]
    
    # Sélection des tickers
    if tickers is None:
        # Sélectionner les 10 tickers avec le plus de données complètes
        ticker_completeness = df.groupby('ticker')['ticker_data_quality'].mean()
        tickers = ticker_completeness.nlargest(10).index.tolist()
    elif len(tickers) > 10:
        print("Limitation à 10 tickers pour la lisibilité des graphiques")
        tickers = tickers[:10]
    
    if metrics is None:
        metrics = ['epsActual_rolling', 'netincomepershare_rolling', 'fcfpershare_rolling']
    
    # Préparation des couleurs pour les graphiques
    colors = plt.cm.tab10(np.linspace(0, 1, len(tickers)))
    
    # 1. Graphique d'évolution temporelle des EPS par ticker
    plt.figure(figsize=(14, 8))
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker].sort_values('quarter_end')
        if 'epsActual_rolling' in ticker_data.columns and not ticker_data['epsActual_rolling'].isna().all():
            plt.plot(ticker_data['quarter_end'], ticker_data['epsActual_rolling'], 
                     label=f"{ticker} - EPS", color=colors[i], linewidth=2)
    
    plt.title('Évolution des EPS (rolling) par ticker', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('EPS (rolling)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 2. Comparaison des ratios FCF/Net Income par ticker
    plt.figure(figsize=(12, 8))
    
    ratio_data = []
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker]
        ticker_data = ticker_data.dropna(subset=['freeCashFlow_rolling', 'netIncome_rolling'])
        if len(ticker_data) > 0 and (ticker_data['netIncome_rolling'] != 0).all():
            ratio = ticker_data['freeCashFlow_rolling'] / ticker_data['netIncome_rolling']
            ratio_data.append({
                'ticker': ticker,
                'mean_ratio': ratio.mean(),
                'std_ratio': ratio.std(),
                'count': len(ticker_data)
            })
    
    ratio_df = pd.DataFrame(ratio_data)
    if len(ratio_df) > 0:
        ratio_df = ratio_df.sort_values('mean_ratio')
        
        plt.barh(ratio_df['ticker'], ratio_df['mean_ratio'], xerr=ratio_df['std_ratio'], 
                alpha=0.7, capsize=5, color=[colors[tickers.index(t)] for t in ratio_df['ticker']])
        
        # Ajout du nombre d'observations
        for i, (_, row) in enumerate(ratio_df.iterrows()):
            plt.text(max(0.1, row['mean_ratio'] - row['std_ratio'] - 0.3), i, 
                     f"n={row['count']}", va='center', fontsize=9)
        
        plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='Parité FCF/Net Income')
        plt.title('Ratio moyen FCF/Net Income par ticker', fontsize=16)
        plt.xlabel('Ratio', fontsize=12)
        plt.ylabel('Ticker', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # 3. Comparaison EPS vs Net Income par action
    plt.figure(figsize=(14, 8))
    
    for i, ticker in enumerate(tickers):
        ticker_data = df[df['ticker'] == ticker].dropna(subset=['epsActual_rolling', 'netincomepershare_rolling'])
        if len(ticker_data) > 0:
            plt.scatter(ticker_data['epsActual_rolling'], ticker_data['netincomepershare_rolling'], 
                       label=ticker, alpha=0.7, color=colors[i], s=50)
    
    # Ligne de parité
    max_val = max(df['epsActual_rolling'].max(), df['netincomepershare_rolling'].max())
    min_val = min(df['epsActual_rolling'].min(), df['netincomepershare_rolling'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ligne de parité')
    
    plt.title('EPS vs Net Income par action', fontsize=16)
    plt.xlabel('EPS (rolling)', fontsize=12)
    plt.ylabel('Net Income par action (rolling)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 4. Corrélation entre FCF et Net Income par ticker
    plt.figure(figsize=(12, 8))
    
    corr_data = []
    for ticker in tickers:
        ticker_data = df[df['ticker'] == ticker].dropna(subset=['fcfpershare_rolling', 'netincomepershare_rolling'])
        if len(ticker_data) >= 4:  # Au moins 4 points pour une corrélation significative
            correlation = ticker_data['fcfpershare_rolling'].corr(ticker_data['netincomepershare_rolling'])
            corr_data.append({
                'ticker': ticker,
                'correlation': correlation,
                'count': len(ticker_data)
            })
    
    corr_df = pd.DataFrame(corr_data)
    if len(corr_df) > 0:
        corr_df = corr_df.sort_values('correlation')
        
        bars = plt.barh(corr_df['ticker'], corr_df['correlation'], 
                alpha=0.7, color=[colors[tickers.index(t)] for t in corr_df['ticker']])
        
        # Ajout du nombre d'observations
        for i, (_, row) in enumerate(corr_df.iterrows()):
            plt.text(max(-0.9, row['correlation'] - 0.1), i, 
                     f"n={row['count']}", va='center', fontsize=9)
        
        plt.title('Corrélation entre FCF et Net Income par action', fontsize=16)
        plt.xlabel('Coefficient de corrélation', fontsize=12)
        plt.ylabel('Ticker', fontsize=12)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        plt.xlim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # 5. Matrice de corrélation des métriques clés (moyenne sur tous les tickers)
    key_metrics = ['epsActual_rolling', 'netincomepershare_rolling', 
                  'fcfpershare_rolling', 'ebitpershare_rolling', 
                  'ebitdapershare_rolling']
    
    # Filtrer les métriques disponibles
    available_metrics = [m for m in key_metrics if m in df.columns and not df[m].isna().all()]
    
    if len(available_metrics) >= 2:  # Au moins 2 métriques pour la corrélation
        plt.figure(figsize=(10, 8))
        corr_matrix = df[available_metrics].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, fmt='.2f')
        plt.title('Matrice de corrélation des métriques financières', fontsize=16)
        plt.tight_layout()
        plt.show()

def generate_ticker_report(funda, ticker, output_format='text'):
    """
    Génère un rapport détaillé pour un ticker spécifique
    
    Args:
        funda: DataFrame contenant les données financières
        ticker: Le ticker à analyser
        output_format: Format de sortie ('text' ou 'html')
        
    Returns:
        str: Rapport formaté pour le ticker spécifié
    """
    # Nettoyage des données
    df = clean_financial_data(funda)
    
    # Filtrage pour le ticker spécifique
    ticker_data = df[df['ticker'] == ticker].sort_values('quarter_end')
    
    if len(ticker_data) == 0:
        return f"Aucune donnée disponible pour le ticker {ticker}"
    
    # Analyse du ticker
    results = analyze_financial_ratios(funda, top_n_tickers=1)
    
    if ticker not in results:
        return f"Analyse impossible pour le ticker {ticker} - données insuffisantes"
    
    ticker_results = results[ticker]
    
    # Construction du rapport
    if output_format == 'text':
        report = [
            f"=== RAPPORT D'ANALYSE FINANCIÈRE: {ticker} ===",
            f"Périodes analysées: {ticker_results['data_completeness']['total_quarters']} trimestres",
            f"Score de qualité des données: {ticker_results['quality_score']:.2f} - {ticker_results['quality_assessment']}",
            "",
            "--- STATISTIQUES CLÉS ---"
        ]
        
        # Statistiques des métriques principales
        for metric in ['epsActual_rolling', 'netincomepershare_rolling', 'fcfpershare_rolling']:
            if metric in ticker_results['statistics']:
                stats = ticker_results['statistics'][metric]
                report.append(f"{metric}:")
                report.append(f"  Moyenne: {stats['mean']:.4f}")
                report.append(f"  Médiane: {stats['50%']:.4f}")
                report.append(f"  Min/Max: {stats['min']:.4f} / {stats['max']:.4f}")
                report.append(f"  Écart-type: {stats['std']:.4f}")
                report.append("")
        
        # Comparaisons des ratios
        report.append("--- COMPARAISONS DES RATIOS ---")
        for name, comparison in ticker_results['ratio_comparisons'].items():
            report.append(f"{name}:")
            report.append(f"  Ratio moyen: {comparison['mean_ratio']:.4f}")
            report.append(f"  Cohérence: {comparison['consistency']:.4f}")
            report.append(f"  Corrélation: {comparison['correlation']:.4f}")
            report.append("")
        
        # Tendances
        report.append("--- ANALYSE DES TENDANCES ---")
        for metric, trend in ticker_results['trend_analysis'].items():
            report.append(f"{metric}: {trend['trend']}")
            report.append(f"  Croissance annualisée: {trend['annualized_growth']:.2f}%")
            report.append(f"  Confiance (R²): {trend['r_squared']:.4f}")
            report.append("")
        
        # Anomalies
        report.append("--- ANOMALIES DÉTECTÉES ---")
        report.append(f"Nombre d'anomalies: {ticker_results['anomalies']['count']} ({ticker_results['anomalies']['percentage']:.1f}% des périodes)")
        if ticker_results['anomalies']['count'] > 0:
            report.append("Périodes avec anomalies:")
            for period in ticker_results['anomalies']['periods'][:5]:  # Limiter à 5 périodes
                report.append(f"  - {period}")
            if len(ticker_results['anomalies']['periods']) > 5:
                report.append(f"  ... et {len(ticker_results['anomalies']['periods']) - 5} autres périodes")
        report.append("")
        
        # Recommandations
        report.append("--- RECOMMANDATIONS ---")
        if ticker_results['recommendations']:
            for i, rec in enumerate(ticker_results['recommendations'], 1):
                report.append(f"{i}. {rec}")
        else:
            report.append("Aucune recommandation spécifique.")
        
        return "\n".join(report)
    
    elif output_format == 'html':
        # Implémentation du format HTML si nécessaire
        return "<p>Format HTML non implémenté</p>"
    
    else:
        return "Format de sortie non supporté. Utilisez 'text' ou 'html'."

def analyze_all_tickers(funda, min_quarters=4, top_n=20, output_file=None):
    """
    Analyse tous les tickers et génère un rapport de synthèse
    
    Args:
        funda: DataFrame contenant les données financières
        min_quarters: Nombre minimum de trimestres requis (défaut: 4)
        top_n: Nombre de tickers à analyser (défaut: 20)
        output_file: Fichier de sortie pour le rapport (défaut: None = affichage console)
        
    Returns:
        DataFrame: Tableau de synthèse des résultats
    """
    print(f"Analyse des {top_n} meilleurs tickers avec au moins {min_quarters} trimestres de données...")
    
    # Analyse des ratios financiers
    results = analyze_financial_ratios(funda, min_quarters=min_quarters, top_n_tickers=top_n)
    
    # Création d'un DataFrame de synthèse
    summary_data = []
    
    for ticker, data in results.items():
        # Extraction des métriques clés
        eps_ni_ratio = None
        fcf_ni_ratio = None
        eps_fcf_ratio = None
        
        if 'EPS vs Net Income/Share' in data['ratio_comparisons']:
            eps_ni_ratio = data['ratio_comparisons']['EPS vs Net Income/Share']['mean_ratio']
        
        if 'FCF/Share vs Net Income/Share' in data['ratio_comparisons']:
            fcf_ni_ratio = data['ratio_comparisons']['FCF/Share vs Net Income/Share']['mean_ratio']
        
        if 'FCF/Share vs EPS' in data['ratio_comparisons']:
            eps_fcf_ratio = 1 / data['ratio_comparisons']['FCF/Share vs EPS']['mean_ratio']
        
        # Tendances de croissance
        eps_growth = None
        ni_growth = None
        fcf_growth = None
        
        if 'epsActual_rolling' in data['trend_analysis']:
            eps_growth = data['trend_analysis']['epsActual_rolling']['annualized_growth']
        
        if 'netincomepershare_rolling' in data['trend_analysis']:
            ni_growth = data['trend_analysis']['netincomepershare_rolling']['annualized_growth']
        
        if 'fcfpershare_rolling' in data['trend_analysis']:
            fcf_growth = data['trend_analysis']['fcfpershare_rolling']['annualized_growth']
        
        # Compilation des données de synthèse
        summary_data.append({
            'ticker': ticker,
            'quality_score': data['quality_score'],
            'quality_assessment': data['quality_assessment'],
            'total_quarters': data['data_completeness']['total_quarters'],
            'anomaly_percentage': data['anomalies']['percentage'],
            'eps_ni_ratio': eps_ni_ratio,
            'fcf_ni_ratio': fcf_ni_ratio,
            'eps_fcf_ratio': eps_fcf_ratio,
            'eps_growth': eps_growth,
            'ni_growth': ni_growth,
            'fcf_growth': fcf_growth,
            'recommendation_count': len(data['recommendations'])
        })
    
    # Création du DataFrame de synthèse
    summary_df = pd.DataFrame(summary_data)
    
    # Tri par score de qualité
    summary_df = summary_df.sort_values('quality_score', ascending=False)
    
    # Affichage ou export du rapport
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"Rapport de synthèse exporté vers {output_file}")
    else:
        print("\n=== RAPPORT DE SYNTHÈSE ===")
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(summary_df)
    
    # Visualisation comparative des tickers du top 10
    top_tickers = summary_df.head(10)['ticker'].tolist()
    print("\nGénération des visualisations comparatives pour le top 10 des tickers...")
    visualize_ticker_comparison(funda, tickers=top_tickers)
    
    return summary_df


# %% Applications
A = learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = funct_backtest.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['ROIC', 'ROIC_lag4', 'PE_inverted'],
    tresh = 0.8,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = [],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance']) 

B = learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = funct_backtest.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['ROIC', 'ROIC_lag4', 'PE_inverted','eps_netincome'],
    tresh = 0.7,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = [],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance']) 
# %%

def compare_models(models_data, start_year=None, end_year=None, risk_free_rate=0.02):
    """
    Compare performance metrics for multiple investment models.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and DataFrames as values.
        Each DataFrame must have 'year_month' (as pd.Period), 'monthly_return', and 'N' columns.
    start_year : int, optional
        Starting year for analysis. If None, uses earliest available data.
    end_year : int, optional
        Ending year for analysis. If None, uses latest available data.
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculation. Default is 2%.
        
    Returns:
    --------
    tuple
        (performance_metrics_df, cumulative_returns_df, fig)
        - DataFrame with performance metrics
        - DataFrame with cumulative returns over time
        - Matplotlib figure with performance visualization
    """
    # Process and align data
    processed_data = {}
    min_date = pd.Period('2100-01', freq='M')
    max_date = pd.Period('1900-01', freq='M')
    
    for model_name, df in models_data.items():
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Ensure year_month is a Period object
        if not isinstance(df_copy['year_month'].iloc[0], pd.Period):
            try:
                df_copy['year_month'] = df_copy['year_month'].dt.to_period('M')
            except:
                # If it's not a datetime, try to convert it
                df_copy['year_month'] = pd.to_datetime(df_copy['year_month']).dt.to_period('M')
        
        # Filter by start and end years if provided
        if start_year:
            df_copy = df_copy[df_copy['year_month'].dt.year >= start_year]
        if end_year:
            df_copy = df_copy[df_copy['year_month'].dt.year <= end_year]
            
        if df_copy.empty:
            print(f"Warning: No data available for {model_name} in selected time period")
            continue
            
        # Update min and max dates
        min_date = min(min_date, df_copy['year_month'].min())
        max_date = max(max_date, df_copy['year_month'].max())
        
        # Create a clean Series with period index and returns
        returns_series = df_copy.set_index('year_month')['monthly_return']
        processed_data[model_name] = returns_series
    
    # Combine all returns into a single DataFrame
    all_returns = pd.DataFrame(processed_data)
    
    # Convert Period index to timestamp for easier plotting
    all_returns.index = all_returns.index.to_timestamp()
    all_returns.sort_index(inplace=True)
    
    # Calculate cumulative returns (starting with $1)
    cumulative_returns = (all_returns + 1).cumprod()
    
    # Calculate performance metrics
    metrics = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].dropna()
        
        # Skip if insufficient data
        if len(model_returns) < 12:
            print(f"Warning: Insufficient data for {model}")
            continue
            
        # Calculate metrics
        total_months = len(model_returns)
        total_years = total_months / 12
        
        # Returns
        total_return = cumulative_returns[model].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (1 / total_years) - 1
        monthly_mean = model_returns.mean()
        
        # Risk metrics
        monthly_std = model_returns.std()
        annualized_vol = monthly_std * np.sqrt(12)
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        
        # Drawdown calculation
        rolling_max = cumulative_returns[model].cummax()
        drawdown = (cumulative_returns[model] / rolling_max - 1)
        max_drawdown = drawdown.min()
        
        # Positive months
        positive_months = (model_returns > 0).sum() / total_months
        
        # Calculate CAGR for different time periods
        cagr_3yr = None
        cagr_5yr = None
        cagr_10yr = None
        
        if total_years >= 3:
            returns_3yr = model_returns.iloc[-36:]
            cagr_3yr = (1 + returns_3yr).prod() ** (1/3) - 1
        
        if total_years >= 5:
            returns_5yr = model_returns.iloc[-60:]
            cagr_5yr = (1 + returns_5yr).prod() ** (1/5) - 1
            
        if total_years >= 10:
            returns_10yr = model_returns.iloc[-120:]
            cagr_10yr = (1 + returns_10yr).prod() ** (1/10) - 1
        
        # Store metrics
        metrics[model] = {
            'Start Date': min_date.strftime('%Y-%m'),
            'End Date': max_date.strftime('%Y-%m'),
            'Total Return': total_return,
            'CAGR': annualized_return,
            'CAGR (3Y)': cagr_3yr,
            'CAGR (5Y)': cagr_5yr,
            'CAGR (10Y)': cagr_10yr,
            'Monthly Mean': monthly_mean,
            'Monthly Volatility': monthly_std,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_drawdown,
            'Positive Months %': positive_months,
            'Number of Stocks (Avg)': models_data[model]['N'].mean() if 'N' in models_data[model].columns else None
        }
    
    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    
    # Format percentages and ratios
    for col in ['Total Return', 'CAGR', 'CAGR (3Y)', 'CAGR (5Y)', 'CAGR (10Y)', 
                'Monthly Mean', 'Monthly Volatility', 'Annualized Volatility', 
                'Max Drawdown', 'Positive Months %']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    
    if 'Sharpe Ratio' in metrics_df.columns:
        metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
    
    # Create visualizations
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Cumulative Returns
    ax1 = plt.subplot(2, 2, 1)
    cumulative_returns.plot(ax=ax1)
    ax1.set_title('Cumulative Returns')
    ax1.set_ylabel('Value of $1 Investment')
    ax1.grid(True)
    ax1.set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Drawdowns
    ax2 = plt.subplot(2, 2, 2)
    for model in all_returns.columns:
        rolling_max = cumulative_returns[model].cummax()
        drawdown = (cumulative_returns[model] / rolling_max - 1)
        drawdown.plot(ax=ax2, label=model)
    ax2.set_title('Drawdowns')
    ax2.set_ylabel('Drawdown')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: Rolling 12-month returns
    ax3 = plt.subplot(2, 2, 3)
    rolling_annual = all_returns.rolling(12).apply(lambda x: np.prod(1 + x) - 1)
    rolling_annual.plot(ax=ax3)
    ax3.set_title('Rolling 12-Month Returns')
    ax3.set_ylabel('12-Month Return')
    ax3.grid(True)
    
    # Plot 4: Return distribution
    ax4 = plt.subplot(2, 2, 4)
    for model in all_returns.columns:
        sns.kdeplot(all_returns[model].dropna(), ax=ax4, label=model)
    ax4.set_title('Return Distribution')
    ax4.set_xlabel('Monthly Return')
    ax4.grid(True)
    
    plt.tight_layout()
    
    return metrics_df, cumulative_returns, fig

def plot_monthly_returns_heatmap(model_data, model_name):
    """
    Create a heatmap of monthly returns for a single model.
    
    Parameters:
    -----------
    model_data : DataFrame
        DataFrame with 'year_month' (as pd.Period) and 'monthly_return' columns.
    model_name : str
        Name of the model for the plot title.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the heatmap
    """
    # Make a copy to avoid modifying the original
    df = model_data.copy()
    
    # Ensure year_month is a Period object
    if not isinstance(df['year_month'].iloc[0], pd.Period):
        try:
            df['year_month'] = df['year_month'].dt.to_period('M')
        except:
            df['year_month'] = pd.to_datetime(df['year_month']).dt.to_period('M')
    
    # Extract year and month
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month
    
    # Pivot the data for the heatmap
    heatmap_data = df.pivot(index='year', columns='month', values='monthly_return')
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create a custom colormap centered at 0
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    # Create the heatmap
    sns.heatmap(heatmap_data, 
                cmap=cmap, 
                center=0,
                annot=True, 
                fmt='.1%', 
                linewidths=.5, 
                ax=ax,
                cbar_kws={'label': 'Monthly Return'})
    
    # Set month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    ax.set_xticklabels(month_names)
    
    # Set title
    ax.set_title(f'Monthly Returns Heatmap - {model_name}', fontsize=16)
    
    return fig


# Comparer les modèles
models = {
    'Model A': A[1],
    'Model B': B[1]
}