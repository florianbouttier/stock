
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


import src.functions_backtest as funct_backtest

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

# %% Launching technical analysis
A = funct_backtest.learning_process_technical(
    Prices = funct_backtest.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
    Historical_Company = US_historical_company[['Month','ticker']], 
    Stocks_Filter = Selection_Stocks_,
    Index_Price = SP500_Monthly,
    Sector = General[['ticker','Sector']],
    func_MovingAverage = funct_backtest.ema_moving_average,
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


B = funct_backtest.learning_process_technical(
        Prices = funct_backtest.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
        Historical_Company = US_historical_company[['Month','ticker']], 
        Stocks_Filter = Selection_Stocks_,
        Index_Price = SP500_Monthly,
        Sector = General[['ticker','Sector']],
        func_MovingAverage = funct_backtest.ema_moving_average,
        Liste_NLong = [50+20*i for i in range(8)],
        Liste_NShort = [1]+[5+5*i for i in range(6)],
        Liste_NAsset= [20], 
        Final_NMaxAsset = 5,
        Max_PerSector = 2,
        List_Alpha =  [1+0.5*i for i in range(10)],
        List_Temp = [12*(2 + 2*i) for i in range(6)],
        mode = "mean",
        param_temp_Lvl2 = 5*12,
        param_alpha_Lvl2 = 2)

C = funct_backtest.learning_process_technical(
    Prices = funct_backtest.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
    Historical_Company = US_historical_company[['Month','ticker']], 
    Stocks_Filter = Selection_Stocks_,
    Index_Price = SP500_Monthly,
    Sector = General[['ticker','Sector']],
    func_MovingAverage = funct_backtest.ema_moving_average,
    Liste_NLong = [50+20*i for i in range(8)],
    Liste_NShort = [1]+[5+5*i for i in range(6)],
    Liste_NAsset= [20], 
    Final_NMaxAsset = 5,
    Max_PerSector = 2,
    List_Alpha =  [1+0.5*i for i in range(5)],
    List_Temp = [12*(4 + 2*i) for i in range(4)],
    mode = "mean",
    param_temp_Lvl2 = 5*12,
    param_alpha_Lvl2 = 3)


Benchmark = funct_backtest.return_benchmark(
    Prices = funct_backtest.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()), 
    Historical_Company = US_historical_company[['Month','ticker']], 
    Stocks_Filter = Selection_Stocks_,
    Index_Price = SP500_Monthly,
    Sector = General[['ticker','Sector']])

Full_Monthly_Return = (pd.concat([(A[0][['Month', 'DR']].assign(Model='ModelA')),
                                (B[0][['Month', 'DR']].assign(Model='ModelB')),
                                (C[0][['Month', 'DR']].assign(Model='ModelC')),
                                Benchmark])
                        .assign(Year=lambda x: x['Month'].dt.year))

Full_Yearly_Return = (Full_Monthly_Return
                    .groupby(['Year','Model'])
                    .agg({'DR': 'prod'})
                    .reset_index()
                    .assign(Label =  lambda x:  ((x['DR'] - 1) * 100).round(2).astype(str) + '%')
                    )

result = pd.DataFrame()
for year in Full_Monthly_Return['Year'].unique() : 
    Full_Monthly_Return_Loop = (Full_Monthly_Return[Full_Monthly_Return['Year']>= year]
                                .groupby(['Model'])
                                .agg(Return=('DR', lambda x: np.prod(x)**(12/len(x))-1),
                                    Worst=('DR', 'min'),
                                    Vol=('DR', 'std'))
                            .reset_index()
                            .assign(Year = year)
                            .assign(Sharpe = lambda x : x['Return']/x['Vol']))
    result = pd.concat([result, Full_Monthly_Return_Loop])


# %% Launching technical analysis
result = result.assign(Label =  lambda x:  ((x['Return']) * 100).round(2).astype(str) + '%')
plt.figure(figsize=(10*0.8, 8*0.8))
sns.set(style="darkgrid")
cmap = sns.color_palette("viridis", as_cmap=True)
g = sns.heatmap(
    result.pivot(index='Year', columns='Model', values='Return'),
    annot=result .pivot(index='Year', columns='Model', values='Label'),
    fmt="", 
    cmap=cmap,
    cbar_kws={'label': 'Return'}
)

plt.title('Return by Strategy and Year')
plt.tight_layout()
plt.show()

# Plotting
plt.figure(figsize=(10, 8))
sns.set(style="whitegrid")

# Use viridis colormap for fill
cmap = sns.color_palette("viridis", as_cmap=True)

# Create the tile plot
g = sns.heatmap(
    Full_Monthly_Return.pivot(index='Year', columns='Model', values='DR'),
    annot=Full_Monthly_Return .pivot(index='Year', columns='Model', values='Label'),
    fmt="", 
    cmap=cmap,
    cbar_kws={'label': 'Return'}
)

plt.title('Return by Strategy and Year')
plt.tight_layout()
plt.show()


# %% funda
AA = funct_backtest.learning_fundamental(
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

BB = funct_backtest.learning_fundamental(
    balance = Balance_Sheet,
    cashflow = Cash_Flow,
    income = Income_Statement,
    earnings = Earnings, 
    general = General,
    monthly_return = funct_backtest.calculate_monthly_returns(Finalprice),
    Historical_Company = US_historical_company[['Month','ticker']],
    col_learning = ['ROIC', 'ROIC_lag4', 'PE_inverted'],
    tresh = 0.75,
    n_max_sector = 2,
    list_kpi_toinvert = ['PE'],
    list_kpi_toincrease = [],
    list_ratios_toincrease = ['ROIC'],
    list_kpi_toaccelerate = [],
    list_lag_increase = [4],
    list_ratios_to_augment = [],
    list_date_to_maximise = ['filing_date_income', 'filing_date_balance']) 



CC = funct_backtest.learning_fundamental(balance = Balance_Sheet,
                          cashflow = Cash_Flow,
                          income = Income_Statement,
                          earnings = Earnings, 
                          general = General,
                          monthly_return = funct_backtest.calculate_monthly_returns(Finalprice),
                          Historical_Company = US_historical_company[['Month','ticker']],
                          col_learning = ['epsActual_rolling_lag4', 'epsActual_rolling_lag4_lag1', 'PE_inverted'],
                          tresh = 0.8,
                          n_max_sector = 2,
                          list_kpi_toinvert = ['PE'],
                          list_kpi_toincrease = ['epsActual_rolling'],
                          list_ratios_toincrease = [],
                          list_kpi_toaccelerate = ['epsActual_rolling'],
                          list_lag_increase = [4],
                          list_ratios_to_augment = [],
                          list_date_to_maximise = ['filing_date_earning']) 


last_month = max(AA[0]['year_month'])
ActualPortfolio = pd.concat([AA[0][AA[0]['year_month'] == last_month][['ticker','Sector','ROIC']],
                                A[2][['ticker','Sector','MTR']]])

print(ActualPortfolio.sort_values('ticker'))

"""
#NotOccidental
PricesEUR = pd.read_parquet('EODHD/Data/EUR_Prices.parquet')
Index = pd.read_parquet('EODHD/Data/EUR_Index.parquet')
EarningsEUR = pd.read_parquet('EODHD/Data/EUR_Earnings.parquet')
General = pd.concat([pd.read_parquet(file) for file in glob.glob(os.path.join('EODHD/Data/General', '*.parquet'))], ignore_index=True)
Index['Month'] = Index['date']
Index = Index[['Month','ticker','MarketCap']]

EurIndex = pd.read_parquet('EODHD/Data/EurIndex.parquet')
EurIndexPrice = pd.read_parquet('EODHD/Data/EurIndexPrice.parquet')
EURPrices = pd.read_parquet('EODHD/Data/EurPrice.parquet')

EURPrices_ = retreate_prices(df = EURPrices).dropna(subset = ['adjusted_close'])
EURPrices_ = Price_VS_Index(Index = EurIndexPrice.reset_index(), 
                            Prices = EURPrices_)

EurIndexPrice = EurIndexPrice.reset_index()
EurIndexPrice['Year'] = EurIndexPrice['date'].dt.year
"""

# %%


def dataquality(prices,balance,cashflow,income,earnings) : 
    A = (prices
            .sort_values(['date'])
            .groupby(['ticker'])
            .apply(lambda x : x.assign(DR_close = x['close']/x['close'].shift(1)),include_groups=False)
            .reset_index()
            .drop(columns = ['level_1'])
            .groupby(['ticker'])
            .apply(lambda x : x.assign(DR_adjusted_close = x['adjusted_close']/x['adjusted_close'].shift(1)),include_groups=False)
            .reset_index())