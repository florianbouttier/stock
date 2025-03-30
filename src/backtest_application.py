
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
sys.path.append(os.path.dirname(os.getcwd()))


import src.functions_backtest as fb

env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
#env_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'stock')
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
# %% Retreatement    
US_historical_company['ticker'] = US_historical_company['Ticker'].apply(lambda x: re.sub(r'\.', '-', x) if isinstance(x, str) else x)
US_historical_company['ticker'] = US_historical_company['ticker'] + '.US'
US_historical_company['Month'] = pd.to_datetime(US_historical_company['Date']).dt.to_period('M')
Finalprice['year_month'] = pd.to_datetime(Finalprice['date']).dt.to_period('M')
mr = fb.calculate_monthly_returns(df = Finalprice)
Selection_Stocks = fb.calculate_pe_ratios(balance = Balance_Sheet, earnings = Earnings,cashflow=Cash_Flow, income=Income_Statement,earning_choice= 'netIncome_rolling',monthly_return=mr,list_date_to_maximise = ['filing_date_income', 'filing_date_balance'])
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

# %% Learnings technical result
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
    param_alpha_Lvl2 = 2)

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
    col_learning = ['ROIC', 'ROIC_lag4_days_increase', 'PE_inverted'],
    earning_choice = 'epsActual_rolling',
    list_date_to_maximise_earning_choice = ['filing_date_earning', 'filing_date_balance'],
    tresh = 0.8,
    n_max_sector = 3,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = ['ROIC_lag4'],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance','filing_date_earning']) 


C_funda = fb.learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = fb.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['epsActual_rolling_lag4_lag1_days_increase', 'PE_inverted'],
    earning_choice = 'epsActual_rolling',
    list_date_to_maximise_earning_choice = ['filing_date_earning', 'filing_date_balance'],
    tresh = 0.8,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = ['epsActual_rolling'],
    list_ratios_toincrease = [],
    list_kpi_toaccelerate = ['epsActual_rolling'],
    list_lag_increase = [4],
    list_ratios_to_augment = ['epsActual_rolling_lag4_lag1'],
    list_date_to_maximise = ['filing_date_balance','filing_date_earning']) 


# Comparer les modèles
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

Funda_A_VS_SP500 = (A_funda[1]
                    .assign(monthly_return = lambda x : x['monthly_return']-1)
                    .merge(SP500_Monthly.rename(columns={'Month': 'year_month','DR_SP500': 'monthly_return_SP500'}),
                           on = ['year_month'])
                    .assign(monthly_return = lambda x : x['monthly_return']/x['monthly_return_SP500'])
                    .dropna())

Technical_A_VS_SP500 = (A_TR[0].rename(columns={'Month': 'year_month',
                                            'DR': 'monthly_return'})
                    .assign(monthly_return = lambda x : x['monthly_return']-1)
                    .merge(SP500_Monthly.rename(columns={'Month': 'year_month','DR_SP500': 'monthly_return_SP500'}),
                           on = ['year_month'])
                    .assign(monthly_return = lambda x : x['monthly_return']/x['monthly_return_SP500'])
                    .dropna())[['year_month','monthly_return']]


models_vs_stp500 = {
    'A Funda': Funda_A_VS_SP500,
    
    'Technical A': Technical_A_VS_SP500
}
metrics, cumulative, correlation, worst_periods, figures = fb.compare_models(models_vs_stp500, start_year=2000)