
import pandas as pd
from tqdm import tqdm
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
def calculate_pe_ratios(balance, earnings,cashflow,income, earning_choice,monthly_return,list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']):
    # 1. Traitement du bilan comptable
    fundamental = calculate_fundamental_ratios(balance=balance,
                                            cashflow=cashflow,
                                            income = income,
                                            earnings=earnings,
                                            list_kpi_toincrease = [],
                                            list_ratios_toincrease = [],
                                            list_kpi_toaccelerate = [],
                                            list_lag_increase = [],
                                            list_ratios_to_augment = [],
                                            list_date_to_maximise = list_date_to_maximise)
    
    if earning_choice != 'epsActual_rolling' : 
        fundamental = fundamental.assign(Rolling_epsActual =lambda x: x[earning_choice]/x['commonStockSharesOutstanding_rolling'])
    if earning_choice == 'epsActual_rolling' : 
        fundamental = fundamental.assign(Rolling_epsActual =lambda x: x['epsActual_rolling'])
        
    monthly_return['date'] = pd.to_datetime(monthly_return['date'])
    price_merge = (monthly_return
             .merge(fundamental[['ticker', 'date', 'Rolling_epsActual','commonStockSharesOutstanding']],
                    left_on = ['ticker', 'date'],
                    right_on =['ticker', 'date'],
                    how='outer')
             .sort_values(by=['ticker', 'date']))
    #price_merge['final_date'] = price_merge[['date','report_date']].max(axis=1)
    #price_merge = price_merge.sort_values(by='final_date')

    # Appliquer ffill sur last_close et Rolling_epsActual par ticker
    price_merge[['last_close', 'Rolling_epsActual','commonStockSharesOutstanding']] = (
        price_merge.groupby('ticker')[['last_close', 'Rolling_epsActual','commonStockSharesOutstanding']].ffill()
        )

    price_merge['PE'] = price_merge['last_close']/price_merge['Rolling_epsActual']
    #price_merge['PE'] = price_merge['last_close']/price_merge['Rolling_epsActual']
    price_merge['Market_Cap'] = price_merge['last_close']*pd.to_numeric(price_merge['commonStockSharesOutstanding'])
    price_merge['year_month'] = pd.to_datetime(price_merge['date']).dt.to_period('M')
    price_merge_last_day = price_merge.groupby(['ticker', 'year_month'],group_keys = False).apply(lambda x: x.loc[x['date'].idxmax()],include_groups=False).reset_index()
 
    return price_merge_last_day[['ticker','year_month','PE','commonStockSharesOutstanding','Market_Cap']]
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
def learning_fundamental(balance,
                         cashflow,
                         income,
                         earnings,
                         general,
                         monthly_return,
                         Historical_Company,
                         col_learning,
                         earning_choice,
                         list_date_to_maximise_earning_choice,
                         tresh,
                         n_max_sector,
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
                             cashflow = cashflow,
                             income = income, 
                             earning_choice = earning_choice,
                             monthly_return = monthly_return,
                             list_date_to_maximise = list_date_to_maximise_earning_choice)
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
            Merge_Loop[f"{kpi_toinvert }_inverted"] = 1/(Merge_Loop[kpi_toinvert]+0.00001)

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
    

    return return_model , result_summarised ,result,return_model[return_model['year_month'] == max(return_model['year_month'])]       
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
    

def compare_models(models_data, start_year=None, end_year=None, risk_free_rate=0.02):
    """
    Compare performance metrics for multiple investment models.
    
    Parameters:
    -----------
    models_data : dict
        Dictionary with model names as keys and DataFrames as values.
        Each DataFrame must have a 'year_month' column and a column with returns.
    start_year : int, optional
        Starting year for analysis. If None, uses earliest available data.
    end_year : int, optional
        Ending year for analysis. If None, uses latest available data.
    risk_free_rate : float, optional
        Annual risk-free rate for Sharpe ratio calculation. Default is 2%.
        
    Returns:
    --------
    tuple
        (performance_metrics_df, cumulative_returns_df, correlation_matrix, worst_periods_df, figures)
        - DataFrame with performance metrics
        - DataFrame with cumulative returns over time
        - Correlation matrix of returns
        - DataFrame with worst periods
        - Dictionary of matplotlib figures
    """
    # Process and align data
    processed_data = {}
    min_date = pd.Period('2100-01', freq='M')
    max_date = pd.Period('1900-01', freq='M')
    
    for model_name, df in models_data.items():
        # Make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Find the returns column - assume it's the column with 'return' in the name or the second column
        return_cols = [col for col in df_copy.columns if 'return' in col.lower()]
        if return_cols:
            return_col = return_cols[0]
        else:
            # If no column has 'return' in name, try to identify it
            if 'monthly_return' in df_copy.columns:
                return_col = 'monthly_return'
            elif len(df_copy.columns) > 1:
                # Assume the second column is returns (after year_month)
                return_col = df_copy.columns[1]
            else:
                raise ValueError(f"Could not identify returns column for model {model_name}")
        
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
        returns_series = df_copy.set_index('year_month')[return_col]
        processed_data[model_name] = returns_series
    
    # Combine all returns into a single DataFrame
    all_returns = pd.DataFrame(processed_data)
    
    # Calculate correlation matrix
    correlation_matrix = all_returns.corr()
    
    # Convert Period index to timestamp for easier plotting
    all_returns_ts = all_returns.copy()
    all_returns_ts.index = all_returns_ts.index.to_timestamp()
    all_returns_ts.sort_index(inplace=True)
    
    # Calculate cumulative returns (starting with $1)
    cumulative_returns = (all_returns_ts + 1).cumprod()
    
    # Calculate worst periods
    worst_periods = {}
    for model in all_returns.columns:
        # Add year and month columns
        model_returns = all_returns[model].reset_index()
        model_returns['year'] = model_returns['year_month'].dt.year
        model_returns['month'] = model_returns['year_month'].dt.month
        
        # Worst month
        worst_month_idx = model_returns[model].idxmin()
        worst_month = model_returns.loc[worst_month_idx]
        worst_month_date = worst_month['year_month']
        worst_month_return = worst_month[model]
        
        # Calculate annual returns
        annual_returns = model_returns.groupby('year')[model].apply(
            lambda x: np.prod(1 + x) - 1
        )
        
        # Worst year
        worst_year_idx = annual_returns.idxmin()
        worst_year_return = annual_returns.loc[worst_year_idx]
        
        worst_periods[model] = {
            'Worst Month': f"{worst_month_date.strftime('%Y-%m')}: {worst_month_return:.2%}",
            'Worst Year': f"{worst_year_idx}: {worst_year_return:.2%}"
        }
    
    worst_periods_df = pd.DataFrame(worst_periods).T
    
    # Calculate annual returns for each model
    annual_returns_data = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].reset_index()
        model_returns['year'] = model_returns['year_month'].dt.year
        annual_returns = model_returns.groupby('year')[model].apply(
            lambda x: np.prod(1 + x) - 1
        )
        annual_returns_data[model] = annual_returns
    
    annual_returns_df = pd.DataFrame(annual_returns_data)
    
    # Calculate performance metrics
    metrics = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].dropna()
        model_returns_ts = all_returns_ts[model].dropna()
        
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
            returns_3yr = model_returns_ts.iloc[-36:]
            cagr_3yr = (1 + returns_3yr).prod() ** (1/3) - 1
        
        if total_years >= 5:
            returns_5yr = model_returns_ts.iloc[-60:]
            cagr_5yr = (1 + returns_5yr).prod() ** (1/5) - 1
            
        if total_years >= 10:
            returns_10yr = model_returns_ts.iloc[-120:]
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
    
    # Calculate CAGR since each year
    cagr_by_year = calculate_cagr_by_year(all_returns)
    
    # Create combined visualization
    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # Plot 1: Cumulative Returns (top left)
    ax1 = plt.subplot(2, 2, 1)
    cumulative_returns.plot(ax=ax1)
    ax1.set_title('Cumulative Returns', fontsize=16)
    ax1.set_ylabel('Value of $1 Investment', fontsize=12)
    ax1.grid(True)
    ax1.set_yscale('log')  # Log scale for better visualization
    ax1.legend(fontsize=10)
    
    # Plot 2: Correlation Matrix (top right)
    ax2 = plt.subplot(2, 2, 2)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                mask=mask, vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('Return Correlation Matrix', fontsize=16)
    
    # Plot 3: CAGR by Start Year (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(cagr_by_year, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax3)
    ax3.set_title('CAGR by Start Year', fontsize=16)
    ax3.set_ylabel('Start Year', fontsize=12)
    ax3.set_xlabel('Model', fontsize=12)
    
    # Plot 4: Annual Returns (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    sns.heatmap(annual_returns_df, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax4)
    ax4.set_title('Annual Returns by Year', fontsize=16)
    ax4.set_ylabel('Year', fontsize=12)
    ax4.set_xlabel('Model', fontsize=12)
    
    plt.tight_layout()
    """"""
    # Also create individual heatmaps for each model
    individual_heatmaps = {}
    for model in all_returns.columns:
        fig_heatmap = plot_monthly_returns_heatmap(all_returns[model], model)
        individual_heatmaps[model] = fig_heatmap
    
    figures = {
        'main_figure': fig,
        'monthly_heatmaps': individual_heatmaps
    }
    """"""
    return metrics_df, cumulative_returns, correlation_matrix, worst_periods_df, figures
def plot_monthly_returns_heatmap(returns_series, model_name):
    """
    Create a heatmap of monthly returns for a single model.
    
    Parameters:
    -----------
    returns_series : Series
        Series with Period index and monthly returns as values.
    model_name : str
        Name of the model for the plot title.
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the heatmap
    """
    # Make a copy and reset index
    df = returns_series.reset_index()
    
    # Extract year and month
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month
    
    # Pivot the data for the heatmap
    heatmap_data = df.pivot(index='year', columns='month', values=model_name)
    
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
    
    plt.tight_layout()
    return fig
def calculate_cagr_by_year(returns_df):
    """
    Calculate CAGR starting from each year for each model.
    
    Parameters:
    -----------
    returns_df : DataFrame
        DataFrame with Period index and monthly returns for each model as columns.
        
    Returns:
    --------
    DataFrame
        DataFrame with start years as index and models as columns, values are CAGR.
    """
    # Get unique years
    years = sorted(set(returns_df.index.year))
    
    # Initialize results DataFrame
    cagr_results = {}
    
    # For each model
    for model in returns_df.columns:
        model_cagr = {}
        
        # For each start year
        for start_year in years:
            # Filter data starting from this year
            filtered_returns = returns_df.loc[returns_df.index.year >= start_year, model]
            
            # Skip if less than 12 months of data
            if len(filtered_returns) < 12:
                model_cagr[start_year] = np.nan
                continue
            
            # Calculate total return
            total_return = (1 + filtered_returns).prod() - 1
            
            # Calculate years
            years_count = len(filtered_returns) / 12
            
            # Calculate CAGR
            cagr = (1 + total_return) ** (1 / years_count) - 1
            model_cagr[start_year] = cagr
        
        cagr_results[model] = model_cagr
    
    # Convert to DataFrame
    cagr_df = pd.DataFrame(cagr_results)
    
    # Drop rows where all values are NaN
    cagr_df = cagr_df.dropna(how='all')
    
    return cagr_df
