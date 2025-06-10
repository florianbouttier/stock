
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:43:16 2024

@author: flbouttier
"""

# %%
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
sys.path.append(os.path.dirname(os.getcwd()))
import importlib
import backtest_functions

env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
data_dir = os.path.join(env_dir, 'data')
os.chdir(data_dir)

# %% PA
prices= pd.read_parquet('PA_Finalprice.parquet')
prices['year_month'] = pd.to_datetime(prices['date']).dt.to_period('M')
prices['volume'] = prices['volume']*prices['close']
# %% Retreatement    
def create_index(df,columns_to_filter = 'volume',nb_shares = 100) -> pd.DataFrame : 
    
    view_eom = df.groupby('year_month').apply(lambda x: x[x['date'] == x['date'].max()]).reset_index(drop=True)
    selected_stock = view_eom.groupby('year_month').apply(lambda x: x.nlargest(100, 'volume')).reset_index(drop=True)
    selected_stock['year_month'] = selected_stock['year_month'] +1 
    
    df = df.merge(selected_stock[['year_month','ticker']],how = 'inner',on = ['year_month','ticker'])
    df = backtest_functions.retreat_prices(df)
    index = df.groupby('date')['adjusted_close'].agg('mean').reset_index()
    index = index.sort_values(by = 'date').reset_index(drop=True)
    index['index_daily_return'] = index['adjusted_close'].pct_change()
    return df,index

def prices_vs_index(index,prices) : 
    
    index['date'] = pd.to_datetime(index['date'])
    full_date_range = pd.date_range(start=index['date'].min(), end=index['date'].max())
    
    index.set_index('date', inplace=True)
    index = index.reindex(full_date_range)
    index['adjusted_close'] = index['adjusted_close'].ffill()
    index.ffill(inplace=True)
    index = index.reset_index().rename(columns = {"index" : "date"})
    
    #prices_augmented = retreate_prices(prices)
    prices_augmented = prices.copy()    
    prices_augmented['date'] = pd.to_datetime(prices_augmented['date'])
    index['date'] = pd.to_datetime(index['date'])
    
    prices_augmented= prices_augmented.merge(index[['date','adjusted_close']],how = "left",on = "date")
    prices_augmented['close_vs_sp500'] = prices_augmented['adjusted_close_x']/ prices_augmented['adjusted_close_y']
    prices_augmented= prices_augmented.rename(columns = {"adjusted_close_x" : "close"})
    
    prices_augmented.sort_values(by=['ticker', 'date'], inplace=True)
    prices_augmented['close_lag'] = prices_augmented.groupby('ticker')['close'].shift(1)
    prices_augmented['dr'] = prices_augmented['close'] / prices_augmented['close_lag']
    prices_augmented['month'] = prices_augmented['date'].dt.to_period('M')
    
    #prices_augmented = prices_augmented.merge(price[['date','ticker']],on = ['date','ticker'],how = "inner")
    
    return prices_augmented[['date','month','ticker','close','close_vs_sp500','dr']]

# %%

prices,pa_index = create_index(prices,columns_to_filter = 'volume',nb_shares = 100)
prices = backtest_functions.prices_vs_index(index = pa_index.reset_index(drop=True),prices = prices)

A_TR = fb.learning_process_technical(
    Prices = fb.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
    Historical_Company = US_historical_company[['Month','ticker']], 
    Stocks_Filter = Selection_Stocks_,
    Index_Price = SP500_Monthly,
    Sector = General[['ticker','Sector']],
    func_MovingAverage = fb.ema_moving_average,
    Liste_NLong = [50+20*i for i in range(8)],
    Liste_NShort = [1]+[5+5*i for i in range(6)],
    Liste_NAsset= [20], 
    Final_NMaxAsset = 5,
    Max_PerSector = 2,
    List_Alpha =  [1+0.5*i for i in range(5)],
    List_Temp = [12*(4 + 2*i) for i in range(4)],
    mode = "mean",
    param_temp_Lvl2 = 5*12,
    param_alpha_Lvl2 = 1)

B_TR = fb.learning_process_technical(
    Prices = fb.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
    Historical_Company = US_historical_company[['Month','ticker']], 
    Stocks_Filter = Selection_Stocks_Bis,
    Index_Price = SP500_Monthly,
    Sector = General[['ticker','Sector']],
    func_MovingAverage = fb.ema_moving_average,
    Liste_NLong = [50+20*i for i in range(8)],
    Liste_NShort = [1]+[5+5*i for i in range(6)],
    Liste_NAsset= [20], 
    Final_NMaxAsset = 5,
    Max_PerSector = 2,
    List_Alpha =  [1+0.5*i for i in range(5)],
    List_Temp = [12*(4 + 2*i) for i in range(4)],
    mode = "mean",
    param_temp_Lvl2 = 5*12,
    param_alpha_Lvl2 = 1)

C_TR = fb.learning_process_technical(
    Prices = fb.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
    Historical_Company = US_historical_company[['Month','ticker']], 
    Stocks_Filter = Selection_Stocks_,
    Index_Price = SP500_Monthly,
    Sector = General[['ticker','Sector']],
    func_MovingAverage = fb.ema_moving_average,
    Liste_NLong = [50+20*i for i in range(15)],
    Liste_NShort = [1]+[5+5*i for i in range(9)],
    Liste_NAsset= [20], 
    Final_NMaxAsset = 5,
    Max_PerSector = 2,
    List_Alpha =  [1+0.5*i for i in range(7)],
    List_Temp = [12*(4 + 2*i) for i in range(4)],
    mode = "mean",
    param_temp_Lvl2 = 4*12,
    param_alpha_Lvl2 = 3)


# %% Leraning fundamental models
A_funda = fb.learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = fb.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['ROIC', 'ROIC_lag4_days_increase', 'PE_inverted'],
    earning_choice = 'epsActual_rolling',
    list_date_to_maximise_earning_choice = ['filing_date_earning', 'filing_date_balance'],
    tresh = 0.8,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = ['ROIC_lag4'],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance','filing_date_earning']) 


B_funda = fb.learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = fb.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['ROIC', 'ROIC_lag4_days_increase'],
    earning_choice = 'netIncome_rolling',
    list_date_to_maximise_earning_choice = ['filing_date_income', 'filing_date_balance'],
    tresh = 0.9,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = ['ROIC_lag4'],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance']) 


C_funda = fb.learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = fb.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['ROIC', 'ROIC_lag4_days_increase', 'PE_inverted'],
    earning_choice = 'freeCashFlow_rolling',
    list_date_to_maximise_earning_choice = ['filing_date_earning', 'filing_date_balance'],
    tresh = 0.8,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = ['ROIC_lag4'],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance','filing_date_earning']) 



# %% Comparer les modèles
models = {
    'A Funda': (A_funda[1].assign(monthly_return = lambda x : x['monthly_return']-1).dropna()),
    'B Funda': (B_funda[1].assign(monthly_return = lambda x : x['monthly_return']-1).dropna()),
    'C Funda': (C_funda[1].assign(monthly_return = lambda x : x['monthly_return']-1).dropna()),
    'Technical A': (A_TR[0].rename(columns={'Month': 'year_month',
                                            'DR': 'monthly_return'})
              .assign(monthly_return = lambda x : x['monthly_return']-1).dropna()
              )[['year_month','monthly_return']] ,
    'Technical B': (B_TR[0].rename(columns={'Month': 'year_month',
                                            'DR': 'monthly_return'})
              .assign(monthly_return = lambda x : x['monthly_return']-1).dropna()
              )[['year_month','monthly_return']] ,
    'Technical C': (C_TR[0].rename(columns={'Month': 'year_month',
                                            'DR': 'monthly_return'})
              .assign(monthly_return = lambda x : x['monthly_return']-1).dropna()
              )[['year_month','monthly_return']] ,
    'SP500': (SP500_Monthly.rename(columns={'Month': 'year_month',
                                            'DR_SP500': 'monthly_return'})
              .assign(monthly_return = lambda x : x['monthly_return']-1).dropna()
              ) 
}

# Exécuter l'analyse
metrics, cumulative, correlation, worst_periods, figures = fb.compare_models(models, start_year=2006)
print(metrics)
Funda_A_VS_SP500 = (A_funda[1]
                    .assign(monthly_return = lambda x : x['monthly_return']-1)
                    .merge(SP500_Monthly.rename(columns={'Month': 'year_month','DR_SP500': 'monthly_return_SP500'}),
                           on = ['year_month'])
                    .assign(monthly_return = lambda x : x['monthly_return']/x['monthly_return_SP500'])
                    .dropna())

Funda_B_VS_SP500 = (B_funda[1]
                    .assign(monthly_return = lambda x : x['monthly_return']-1)
                    .merge(SP500_Monthly.rename(columns={'Month': 'year_month','DR_SP500': 'monthly_return_SP500'}),
                           on = ['year_month'])
                    .assign(monthly_return = lambda x : x['monthly_return']/x['monthly_return_SP500'])
                    .dropna())

Funda_C_VS_SP500 = (C_funda[1]
                    .assign(monthly_return = lambda x : x['monthly_return']-1)
                    .merge(SP500_Monthly.rename(columns={'Month': 'year_month','DR_SP500': 'monthly_return_SP500'}),
                           on = ['year_month'])
                    .assign(monthly_return = lambda x : x['monthly_return']/x['monthly_return_SP500'])
                    .dropna())

Technical_A_VS_SP500 = (B_TR[0].rename(columns={'Month': 'year_month',
                                            'DR': 'monthly_return'})
                    .assign(monthly_return = lambda x : x['monthly_return']-1)
                    .merge(SP500_Monthly.rename(columns={'Month': 'year_month','DR_SP500': 'monthly_return_SP500'}),
                           on = ['year_month'])
                    .assign(monthly_return = lambda x : x['monthly_return']/x['monthly_return_SP500'])
                    .dropna())[['year_month','monthly_return']]


models_vs_stp500 = {
    'A Funda': Funda_A_VS_SP500,
    'B Funda': Funda_B_VS_SP500,
    'C Funda': Funda_B_VS_SP500,
    'Technical A': Technical_A_VS_SP500
}
metrics, cumulative, correlation, worst_periods, figures = fb.compare_models(models_vs_stp500, start_year=2000)

print(A_funda[3][['ticker','ROIC','Sector','year_month']])
print(B_funda[3][['ticker','ROIC','Sector','year_month']])
print(C_funda[3][['ticker','ROIC','Sector','year_month']])
print(A_TR[2])
# %%
