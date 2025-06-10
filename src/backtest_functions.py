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


def retreat_prices(df):
    # ensure 'date' is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # create a complete date range
    all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max())
    
    # set the index and unstack, then reindex with the complete date range
    df_full = df.set_index(['date', 'ticker']).unstack('ticker').reindex(all_dates).stack('ticker', dropna=False).reset_index().rename(columns={'level_0': 'date'})
    
    # forward fill missing values
    df_full['adjusted_close'] = df_full.groupby('ticker')['adjusted_close'].ffill()
    
    # select relevant columns
    df_full = df_full[['date', 'ticker', 'adjusted_close']]
    
    return df_full
def ema_moving_average(Series, n,wilder = False):

    previous = Series.ewm(span=n, adjust=False).mean()

    return previous 
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
def learning_monthly(data, historical_company, n_long, n_short, func_movingaverage, n_asset):
    
    #df = data.dropna(subset=['close_vs_sp500']).sort_values(by=['ticker', 'date'])
    df = data

    for n, col in [(n_short, 'short_vs_sp500'), (n_long, 'long_vs_sp500')]:
        df[col] = df.groupby('ticker')['close_vs_sp500'].transform(lambda x: func_movingaverage(x, n=n))
    
    df = df.groupby('ticker').apply(lambda x: x.iloc[n_long:],include_groups = False).reset_index(drop=False)

    df = df.dropna(subset = ['short_vs_sp500','long_vs_sp500'])
    pd.options.mode.chained_assignment = None  # default='warn'
    df.loc[:, 'mTr'] = df['short_vs_sp500'] / df['long_vs_sp500']

    # merge with historical_company and select latest date per month
    df = df.merge(historical_company, on=['month', 'ticker'], how='inner')
    df = df.reset_index(drop=True)  # réinitialise l'index après le merge

    # sélection des lignes avec la date maximale par mois et ticker
    max_date_indices = df.groupby(['month', 'ticker'])['date'].idxmax()
    df = df.loc[max_date_indices.values]
    # calculate statistics
    grouped = df.groupby('month')
    df = df.assign(
        mean=grouped['mTr'].transform('mean'),
        ecart=grouped['mTr'].transform('std'),
        quantile_mTr=lambda x: stats.norm.cdf(x['mTr'], x['mean'], x['ecart'])
    )
    
    # select top n_asset per month based on mTr
    df = df.sort_values(by=['month', 'mTr'], ascending=[True, False]).groupby('month').head(n_asset)
    
    df = df.assign(n_long=n_long, n_short=n_short, n_asset=n_asset)[['month', 'date', 'ticker', 'n_long', 'n_short', 'n_asset', 'mTr', 'quantile_mTr']]
    
    # compute final_return
    final_return = (data.groupby(['month', 'ticker'])['dr'].prod()
                    .reset_index()
                    .sort_values(by=['ticker', 'month'])
                    .assign(application_month=lambda x: x.groupby('ticker')['month'].shift(1))
                    .merge(df, left_on=['application_month', 'ticker'], right_on=['month', 'ticker'], how='inner')
                    .rename(columns={"month_x": "month"})[['month', 'application_month', 'date', 'ticker', 'dr', 'n_long', 'n_short', 'n_asset', 'mTr', 'quantile_mTr']]
                    )

    return final_return,df
def decreasing_sum(liste, halfPeriod, mode):
    n = len(liste)
    if n == 0:
        return 0.0

    weight = np.zeros(n)

    if mode == "exponential":
        p = np.log(2) / halfPeriod
        weight = np.exp(-p * np.arange(n))
    elif mode == "tanh":
        p = np.log(3) / (2 * halfPeriod)
        weight = 1 - np.tanh(p * np.arange(n))
    elif mode == "special":
        alpha = halfPeriod
        weight = np.maximum(1 - (1 + (1 + alpha * np.arange(n)) * (np.log(1 + alpha * np.arange(n)) - 1) / (alpha**2)), 0)
    elif mode == "linear":
        weight = np.maximum(1 - np.arange(n) / halfPeriod, 0)
    elif mode == "quadratic":
        weight = np.maximum(1 - (np.arange(n) / halfPeriod)**2, 0)
    elif mode == "sigmoidal":
        k = np.log(3) / halfPeriod
        weight = 1 / (1 + np.exp(k * (np.arange(n) - halfPeriod)))
    elif mode == "mean":
        len1 = min(halfPeriod, n)
        weight[:len1] = 1
    else:
        raise valueerror(f"unknown mode: {mode}.")

    weight_sum = np.sum(weight)
    weight = np.divide(weight, weight_sum) if weight_sum != 0 else np.full(n, 1.0 / n)

    return np.dot(weight, liste)
def custom_sma(Series, n):
    """fonction personnalisée pour calculer la moyenne mobile simple (sma)."""
    return Series.rolling(window=n, min_Periods=1).mean()
def increase(values, n, diff=True, annual_base=4):
    values = pd.Series(values)  # s'assure que values est une Series
    
    if diff:
        return values - values.shift(n)  # différence simple

    v0, v1 = values.shift(n), values  # valeur initiale et valeur actuelle
    #denom = (v0.abs() + v1.abs()) / 2
    denom = v0.abs()
    growth = np.where(denom == 0, np.nan, (v1 - v0) / denom)
    
    base = 1 + growth
    # on vérifie que base est positif pour éviter d'élever un nombre négatif à une puissance fractionnaire
    result = np.where(base > 0, base ** (annual_base / n) - 1, base)
    
    return pd.Series(result, index=values.index)
def retreating_fundamental(income_statement, balance_sheet, cash_flow, earning,price):
  
    income = income_statement[['date', 'ticker', 'filing_date', 'totalrevenue', 'grossprofit', 
                               'operatingincome', 'incomebeforeTax', 'netincome', 'ebit', 'ebitda']]
    
    
    cash = cash_flow[['date', 'ticker', 'filing_date', 'freecashflow']]
  
    balance = balance_sheet[['date', 'ticker', 'filing_date', 'totalassets', 'totalliab', 
                             'totalstockholderequity', 'netdebt', 'commonstocksharesoutstanding']]
    
    earning = earning[['date', 'ticker', 'reportdate', 'beforeaftermarket', 'epsactual']].dropna(subset=['epsactual'])
    
    income['date'],cash['date'],balance['date'],earning ['date'] = pd.to_datetime(income['date']),pd.to_datetime(cash['date']),pd.to_datetime(balance['date']),pd.to_datetime(earning ['date'])
    
    fundamental = income.merge(balance, on=['date', 'ticker'], how='outer') \
                                 .merge(cash, on=['date', 'ticker'], how='outer') \
                                 .merge(earning, on=['date', 'ticker'], how='outer')
                                 
                                 
    
    balance.loc[:, 'filing_date'] = pd.to_datetime(balance['filing_date']) + timedelta(days=1)
    balance.loc[:, 'date'] = pd.to_datetime(balance['date']) 
    balance= (balance.rename(columns={'date': 'quarter', 'filing_date': 'date'})
           .sort_values(by=['ticker', 'quarter'])
           .groupby(['ticker', 'date'], as_index=False)
           .apply(lambda x: x[x['quarter'] == x['quarter'].max()]).reset_index() 
           [['ticker', 'date', 'quarter','commonstocksharesoutstanding']]) 
    balance['quarter'] = pd.to_datetime(balance['quarter'])
    
    #earnings
    earning['beforeaftermarket'] = earning['beforeaftermarket'].fillna("aftermarket_replaced")
    earning['reportdate'] = pd.to_datetime(earning['reportdate'])
    earning['date'] = pd.to_datetime(earning['date'])
    # ajuster 'reportdate' si 'beforeaftermarket' est "beforemarket"
    earning['reportdate'] = earning.apply(
        lambda row: row['reportdate'] + timedelta(days=1) if row['beforeaftermarket'] != "beforemarket" else row['reportdate'],
        axis=1
    )
    
    # renommer les colonnes
    earning = earning.rename(columns={'date': 'quarter', 'reportdate': 'date'})
    
    # remplacer les dates manquantes par 'quarter' + 3 mois
    """
    earning['date'] = earning.apply(
        lambda row: row['quarter'] + pd.dateoffset(months=3) if pd.isna(row['date']) else row['date'],
        axis=1
    )"""
    
    # filtrer pour garder le dernier 'quarter' par groupe 'ticker' et 'date'
    earning = earning.sort_values(by=['ticker', 'date', 'quarter'], ascending=[True, True, False]) \
                       .drop_duplicates(subset=['ticker', 'date'], keep='first')
    
    # Trier et calculer la moyenne mobile simple (rolling_epsactual)
    earning = earning.sort_values(by=['ticker', 'date'])
    earning['rolling_epsactual'] = earning.groupby('ticker')['epsactual'] \
                                            .transform(lambda x: 4 * custom_sma(x, n=4))
    price['date'] = pd.to_datetime(price['date'])                                        
    
    merge = earning.merge(balance,left_on = ["quarter","ticker"],right_on = ["quarter","ticker"])                                            
    merge = merge.dropna(subset = ['date_x','date_y'])
    merge['date_x'],merge['date_y'] = pd.to_datetime(merge['date_x']),pd.to_datetime(merge['date_y'])
    merge['final_date'] = np.maximum(merge['date_x'], merge['date_y'])
    merge_ = merge.merge(price, left_on=["final_date", "ticker"], right_on=["date", "ticker"], how="outer")
    return  merge
def augmenting_ratios(data,kpi_list,date_list) : 
    
    data = data.sort_values(['ticker']+list([date_list]), ascending=[True, True])

    # fonction pour compter les jours consécutifs de valeurs positives
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
    
    if earning_choice != 'epsactual_rolling' : 
        fundamental = fundamental.assign(rolling_epsactual =lambda x: x[earning_choice]/x['commonstocksharesoutstanding_rolling'])
    if earning_choice == 'epsactual_rolling' : 
        fundamental = fundamental.assign(rolling_epsactual =lambda x: x['epsactual_rolling'])
        
    monthly_return['date'] = pd.to_datetime(monthly_return['date'])
    price_merge = (monthly_return
             .merge(fundamental[['ticker', 'date', 'rolling_epsactual','commonstocksharesoutstanding']],
                    left_on = ['ticker', 'date'],
                    right_on =['ticker', 'date'],
                    how='outer')
             .sort_values(by=['ticker', 'date']))
    #price_merge['final_date'] = price_merge[['date','report_date']].max(axis=1)
    #price_merge = price_merge.sort_values(by='final_date')

    # appliquer ffill sur last_close et rolling_epsactual par ticker
    price_merge[['last_close', 'rolling_epsactual','commonstocksharesoutstanding']] = (
        price_merge.groupby('ticker')[['last_close', 'rolling_epsactual','commonstocksharesoutstanding']].ffill()
        )

    price_merge['pe'] = price_merge['last_close']/price_merge['rolling_epsactual']
    #price_merge['pe'] = price_merge['last_close']/price_merge['rolling_epsactual']
    price_merge['market_cap'] = price_merge['last_close']*pd.to_numeric(price_merge['commonstocksharesoutstanding'])
    price_merge['year_month'] = pd.to_datetime(price_merge['date']).dt.to_Period('m')
    price_merge_last_day = price_merge.groupby(['ticker', 'year_month'],group_keys = False).apply(lambda x: x.loc[x['date'].idxmax()],include_groups=False).reset_index()
 
    return price_merge_last_day[['ticker','year_month','pe','commonstocksharesoutstanding','market_cap']]
def calculate_fundamental_ratios(balance,
                                 cashflow,
                                 income,
                                 earnings,
                                 list_kpi_toincrease = ['totalrevenue_rolling', 'grossprofit_rolling', 'operatingincome_rolling', 'incomebeforeTax_rolling', 'netincome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freecashflow_rolling', 'epsactual_rolling'],
                                 list_ratios_toincrease = ['roic', 'netmargin'],
                                 list_kpi_toaccelerate = ['epsactual_rolling'],
                                 list_lag_increase = [1,4,4*5],
                                 list_ratios_to_augment = ["roic_lag4", "roic_lag1", "netmargin_lag4"],
                                 list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']) :



    # 1. Traitement du bilan comptable
    balance_clean = (
        balance[['ticker', 'date', 'filing_date', 'commonstocksharesoutstanding','totalstockholderequity','netdebt']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
            filing_date_balance =lambda x: pd.to_datetime(x['filing_date'])
        )
        .sort_values('filing_date_balance')
        .groupby(['ticker', 'quarter_end'])
        .last()  # prendre la dernière version du rapport pour chaque trimestre
        .reset_index()
        .drop(columns = ['filing_date'])
    )
    for columns in ['totalstockholderequity','netdebt','commonstocksharesoutstanding'] : 
        balance_clean [f"{columns}_rolling"] =  balance_clean.sort_values(['ticker','filing_date_balance']).groupby('ticker')[columns].transform(lambda x: custom_sma(x, n=4))

    # 2. Traitement des résultats
    earnings_clean = (
        earnings[['ticker', 'date', 'reportdate', 'epsactual']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
            filing_date_earning =lambda x: pd.to_datetime(x['reportdate'])
        )
        .sort_values('filing_date_earning')
        .groupby(['ticker', 'quarter_end'])
        .last()  # prendre la dernière révision des résultats
        .reset_index()
        .drop(columns = ['reportdate'])
        .dropna(subset = ['epsactual'])
    )
    earnings_clean['epsactual_rolling'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsactual'] \
                                            .transform(lambda x: 4 * custom_sma(x, n=4))
    "earnings_clean['rolling_epsactual'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsactual'] \
                                            .transform(lambda x: 4 * custom_sma(x, n=4))"
     # 3. income
    columns_to_annualise_income = ['totalrevenue', 'grossprofit', 'operatingincome', 
                       'incomebeforeTax', 'netincome', 'ebit', 'ebitda']
     
    income_clean = (
         income[['ticker', 'date', 'filing_date', 'totalrevenue','grossprofit','operatingincome','incomebeforeTax','netincome','ebit','ebitda']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
             filing_date_income=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_income')
         .groupby(['ticker', 'quarter_end'])
         .last()  # prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    for columns in columns_to_annualise_income : 
        income_clean[f"{columns}_rolling"] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_sma(x, n=4))
        #income_clean[columns] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_sma(x, n=4))
                                          
    cash_clean = (
         cashflow[['ticker', 'date', 'filing_date', 'freecashflow']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
             filing_date_cash=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_cash')
         .groupby(['ticker', 'quarter_end'])
         .last()  # prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    for columns in ['freecashflow'] : 
        cash_clean[f"{columns}_rolling"] =  cash_clean.sort_values(['filing_date_cash']).groupby('ticker')[columns].transform(lambda x: 4 * custom_sma(x, n=4))                                        
    """
    funda = (income_clean
             .merge(cash_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(balance_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(earnings_clean[['ticker', 'quarter_end','filing_date_earning', 'epsactual','epsactual_rolling']],on = ['ticker','quarter_end'],how = 'outer')
             .assign(netmargin = lambda x : x['ebit_rolling']/x['totalrevenue_rolling'])
             .assign(roic = lambda x : x['ebit_rolling']/(x['totalstockholderequity_rolling']+x['netdebt_rolling'].fillna(0)))
             )
    """
    funda = (income_clean
            .merge(cash_clean,on = ['ticker','quarter_end'],how = 'outer')
            .merge(balance_clean,on = ['ticker','quarter_end'],how = 'outer')
            .merge(earnings_clean[['ticker', 'quarter_end','filing_date_earning', 'epsactual','epsactual_rolling']],on = ['ticker','quarter_end'],how = 'outer')
            .assign(netmargin = lambda x : x['ebit_rolling']/x['totalrevenue_rolling'])
            .assign(roic = lambda x : x['ebit_rolling']/(x['totalstockholderequity_rolling']+x['netdebt_rolling'].fillna(0)))
            .assign(ebitpershare_rolling = lambda x : x['ebit_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
            .assign(ebitdapershare_rolling = lambda x : x['ebitda_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
            .assign(netincomepershare_rolling = lambda x : x['netincome_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
            .assign(fcfpershare_rolling = lambda x : x['freecashflow_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
            .assign(eps_fcf = lambda x : x['epsactual_rolling']/(x['fcfpershare_rolling'])-1)
            .assign(eps_netincome = lambda x : x['epsactual_rolling']/(x['netincomepershare_rolling'])-1)
            )

    
    
    for col in list_kpi_toincrease:
        for lag in list_lag_increase : 
            funda = funda.astype({col: 'float'})
            funda[f"{col}_lag{lag}"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, lag, diff=False))
            #funda[f"{col}_yoy_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4, diff=False))
            #funda[f"{col}_yoy_5"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4*5, diff=False))
            #funda[f"{col}_qoq_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 1, diff=False))

    for col in list_ratios_toincrease :
        for lag in list_lag_increase : 
            funda[f"{col}_lag{lag}"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, lag, diff=True))
            #funda[f"{col}_yoy_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4, diff=True))
            #funda[f"{col}_yoy_5"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 4*5, diff=True))
            #funda[f"{col}_qoq_1"] = funda.groupby('ticker')[col].transform(lambda x: increase(x, 1, diff=True))
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
def ranking(values, tresh):
    """applique le ranking sur une série de valeurs."""
    # calcul du quantile sur toute la série
    quantiles = rankdata(values.fillna(values.min() - 1), method="average") / len(values)
    
    # remplacer les 0 par -inf pour exclusion
    quantiles = np.where(quantiles == 0, -np.inf, quantiles)
    
    # appliquer le ranking : mettre 0 si inférieur au seuil
    ranked_values = quantiles * (np.maximum(0, quantiles - tresh) > 0).astype(int)
    
    return ranked_values
def bestmodel(data,mode,param_temp,param_alpha) : 
    
    data['score'] = score(data['total_return'],alpha = param_alpha)
    def best_model_date(data,date, mode,param_alpha, param_temp):
    
        filtered_data = data[data['month'] < date]
 
        summarized_data  = (filtered_data
                .sort_values(by='month', ascending=False) 
                .groupby(['model'], group_keys=False)
                .apply(lambda x: decreasing_sum(x['score'], halfPeriod=param_temp, mode=mode),include_groups = False)
                .rename('score')  # renomme la colonne résultante en 'score'
                .reset_index(name='score') 
                )
       
        best_model = (summarized_data[summarized_data['score'] == summarized_data['score'].max()].sample(1)
                      .assign(
                              month=date,
                              param_alpha=param_alpha,
                              param_temp=param_temp
                              )
                      )
        return best_model
    
    results = []
    data = data.sort_values(['month'])
    list_date = data['month'].unique().tolist()
    list_date.append(pd.to_datetime(datetime.now()).to_period('m') + 1)
    list_date = list_date[1:]
    for date_loop in list_date :
        
            results_loop = best_model_date(data = data,
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
    
    # Étape 2 : extraire 'last_close' et 'adjusted_close_last'
    last_closes_df = last_rows[['date','close', 'adjusted_close']].reset_index().rename(columns={'close': 'last_close', 'adjusted_close': 'adjusted_close_last'})
    
    # Étape 3 : Trier par 'ticker' et 'year_month'
    last_closes_df = last_closes_df.sort_values(by=['ticker', 'year_month'])
    
    # Étape 4 : calculer le rendement mensuel basé sur 'adjusted_close_last'
    last_closes_df['monthly_return'] = last_closes_df.groupby('ticker')['adjusted_close_last'].transform(lambda x: x / x.shift(1))
    
    # Étape 5 : sélectionner les colonnes pertinentes
    final_df = last_closes_df[['ticker','date', 'year_month', 'last_close', 'monthly_return']]
    
    return final_df    
def learning_process_technical(prices,historical_company,index_price,stocks_filter,sector,func_movingaverage,liste_nlong,liste_nshort,liste_nasset,max_persector,final_nmaxasset,list_alpha,list_Temp,mode,param_temp_lvl2,param_alpha_lvl2) : 
    
    parameters = pd.DataFrame([(n_long, n_short, n_asset) 
                           for n_long in liste_nlong 
                           for n_short in liste_nshort 
                           for n_asset in liste_nasset 
                           if n_long > 3 * n_short],
    columns=['n_long', 'n_short', 'n_asset'])
    all_return_monthly = pd.DataFrame()
    all_detaillled_portfolios = pd.DataFrame()
    prices = prices.dropna(subset=['close_vs_sp500']).sort_values(by=['ticker', 'date'])

    print("start backtest level 0")
    for i in tqdm(range(len(parameters)), desc="processing"):
        row = parameters.iloc[i]
        result = learning_monthly(
            data = prices.copy(),
            historical_company = historical_company,
            n_long=row['n_long'],
            n_short=row['n_short'],
            func_movingaverage=func_movingaverage,  
            n_asset=row['n_asset'])
        all_return_monthly = pd.concat([all_return_monthly, result[0]], ignore_index=True)
        all_detaillled_portfolios = pd.concat([all_detaillled_portfolios, result[1]], ignore_index=True)
    
    print("end of  backtest level 0")
    all_return_monthly_afterselection =  (all_return_monthly
                                            .merge(stocks_filter[['year_month','ticker']],
                                                    how = "inner",
                                                    left_on = ['month','ticker'],
                                                    right_on = ['year_month','ticker'])
                                            .merge(sector,on = "ticker",how = "left")
                                            .sort_values('quantile_mTr', ascending=False)
                                            .groupby(['month', 'n_long', 'n_short', 'n_asset', 'sector'], group_keys=False)
                                            .apply(lambda g: g.head(max_persector))
                                            .sort_values('quantile_mTr', ascending=False)
                                            .groupby(['month', 'n_long', 'n_short', 'n_asset'], group_keys=False)
                                            .apply(lambda g: g.head(final_nmaxasset)))
    
    all_return_monthly_afterselection['model'] = all_return_monthly_afterselection['n_long'].astype(str) + "-"+ all_return_monthly_afterselection['n_short'].astype(str) +"-"+all_return_monthly_afterselection['n_asset'].astype(str) 
    all_return_monthly_afterselection_summarised  = (
        all_return_monthly_afterselection
        .groupby(['month', 'model'], as_index=False)
        .agg(
            dr=('dr', 'mean'), #we suppose equiponderation
            n=('dr', 'size')
            )
        .merge(
            index_price[['month', 'dr_sp500']], 
            on='month', 
            how='left'
                )
        .assign(total_return=lambda x: x['dr'] / x['dr_sp500'])
                                            )
    lvl1_bestmodel = []
    print("start learning lvl 1")
    for alpha in tqdm(list_alpha):
        for temp in list_Temp :
            a = bestmodel(
                data = all_return_monthly_afterselection_summarised,
                mode = mode,
                param_temp = temp,
                param_alpha = alpha)
            lvl1_bestmodel.append(a)
    print("end of  learning lvl 1")       
    lvl1_bestmodel = (pd.concat(lvl1_bestmodel, ignore_index=True)
                        .rename(columns = {"model" : "model_lvl0"})
                        .assign(model_lvl1 = lambda x: x['param_alpha'].astype(str)+"-"+x['param_temp'].astype(str))
                        .drop(columns=['score','param_alpha','param_temp'])
                        )
    lvl1_return = (all_return_monthly_afterselection_summarised[['model','month','total_return']]
                    .merge(lvl1_bestmodel,
                            left_on = ["model","month"],
                            right_on = ["model_lvl0","month"],
                            how = "inner")
                    .drop(columns = ['model'])
                    .rename(columns = {"model_lvl1" : "model"})
                    )
    
    lvl2_bestmodel  = (bestmodel(data = lvl1_return,
                            mode = mode,
                            param_temp = param_temp_lvl2,
                            param_alpha = param_alpha_lvl2)[['month','model']]
                        .rename(columns = {"model" : "model_lvl1"})
                        .merge(lvl1_bestmodel,
                            how = "inner",
                            on = ["month","model_lvl1"])
                        .merge(all_return_monthly_afterselection_summarised[['month', 'model', 'dr','dr_sp500','return']],
                            how = "inner",
                            left_on = ['month','model_lvl0'],
                            right_on = ['month','model'])
                        .drop(columns = ['model']))
    
    detail = (lvl2_bestmodel[['month','model_lvl0','model_lvl1']]
                .merge(all_return_monthly_afterselection[['month','ticker','model','sector','dr']],
                        left_on = ['month','model_lvl0'],
                        right_on = ['month','model'])
                .drop(columns = ['model'])
                )
    
    detail[detail['month'] == max(detail['month'])][['month','model_lvl0']].drop_duplicates()
    all_historical_component =  (detail[detail['month'] == max(detail['month'])][['month','model_lvl0']].drop_duplicates()
                                 .merge(all_detaillled_portfolios
                                        .assign(model_lvl0 = lambda x : x['n_long'].astype(str) + "-"+ x['n_short'].astype(str) +"-"+x['n_asset'].astype(str)),
                                        on = ['month','model_lvl0'],
                                        how = 'inner')
                                            .merge(stocks_filter[['year_month','ticker']],
                                                   how = "inner",
                                                   left_on = ['month','ticker'],
                                                   right_on = ['year_month','ticker'])
                                            .merge(sector,on = "ticker",how = "left")
                                            .sort_values('quantile_mTr', ascending=False)
                                            .groupby(['month', 'n_long', 'n_short', 'n_asset', 'sector'], group_keys=False)
                                            .apply(lambda g: g.head(max_persector))
                                            .sort_values('quantile_mTr', ascending=False)
                                            .groupby(['month', 'n_long', 'n_short', 'n_asset'], group_keys=False)
                                            .apply(lambda g: g.head(final_nmaxasset)))
    
    
    return lvl2_bestmodel  , detail,all_historical_component [['date','ticker','n_long', 'n_short', 'n_asset','mTr','sector']]
def learning_fundamental(balance,
                         cashflow,
                         income,
                         earnings,
                         general,
                         monthly_return,
                         historical_company,
                         col_learning,
                         earning_choice,
                         list_date_to_maximise_earning_choice,
                         tresh,
                         n_max_sector,
                         list_kpi_toinvert = ['pe'],
                         list_kpi_toincrease = ['totalrevenue_rolling', 'grossprofit_rolling', 'operatingincome_rolling', 'incomebeforeTax_rolling', 'netincome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freecashflow_rolling', 'epsactual_rolling'],
                         list_ratios_toincrease = ['roic', 'netmargin'],
                         list_kpi_toaccelerate = ['epsactual_rolling'],
                         list_lag_increase = [1,4,4*5],
                         list_ratios_to_augment = ["roic_lag4", "roic_lag1", "netmargin_lag4"],
                         list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']) : 
    
    

    
    ratios = calculate_fundamental_ratios(balance = balance,
                                          cashflow = cashflow,
                                          income = income,
                                          earnings = earnings,
                                          list_kpi_toincrease = list_kpi_toincrease,
                                          list_ratios_toincrease = list_ratios_toincrease,
                                          list_kpi_toaccelerate = list_kpi_toaccelerate,
                                          list_lag_increase = list_lag_increase,
                                          list_ratios_to_augment = list_ratios_to_augment,
                                          list_date_to_maximise = list_date_to_maximise)
    pe = calculate_pe_ratios(balance = balance, 
                             earnings = earnings, 
                             cashflow = cashflow,
                             income = income, 
                             earning_choice = earning_choice,
                             monthly_return = monthly_return,
                             list_date_to_maximise = list_date_to_maximise_earning_choice)
    ratios['year_month'] = ratios['date'].dt.to_Period('m')
    final_merged = []
    list_date_loop = sorted(ratios[ratios['year_month'] >= '2000-01'].dropna(subset = ['year_month'])['year_month'].unique())
    list_date_loop.append(max(list_date_loop) + 1)
    for date_loop in tqdm(list_date_loop) : 
      if not(pd.isna(date_loop)) :     
        historical_company_loop = historical_company[historical_company ['month'] < date_loop]
        historical_company_loop  = historical_company_loop[historical_company_loop['month'] == historical_company_loop['month'].max()]['ticker'].unique()
        
        ratios_loop = ratios[ratios['year_month'] < date_loop]
        ratios_loop = ratios_loop[(ratios_loop['date'] == ratios_loop.groupby('ticker')['date'].transform('max')) & 
                                  (ratios_loop['ticker'].isin(historical_company_loop))]
        ratios_loop['date_diff_ratios'] = ((date_loop-1).to_timestamp(how='end') - pd.to_datetime(ratios_loop['date']) ).dt.days
      
        pe_loop = pe[pe['year_month'] < date_loop]
        pe_loop = pe_loop[(pe_loop['year_month'] == pe_loop.groupby('ticker')['year_month'].transform('max')) & 
                                  (pe_loop['ticker'].isin(historical_company_loop))]
        pe_loop['date_diff_pe'] = ((date_loop-1).to_timestamp(how='end') - pd.to_datetime(pe_loop['year_month'].dt.to_timestamp(how='end'))).dt.days
        
        merge_loop = pe_loop.merge(ratios_loop,on = 'ticker')
        for kpi_toinvert in list_kpi_toinvert : 
            merge_loop[f"{kpi_toinvert }_inverted"] = 1/(merge_loop[kpi_toinvert]+0.00001)

        merge_loop = merge_loop[['ticker']+col_learning]
        for c in col_learning :
                merge_loop[f"{c}_quantile"] = ranking(merge_loop[c],tresh)
        merge_loop['rank'] = merge_loop.filter(regex='_quantile$').prod(axis=1)
        merge_loop = (
            merge_loop.merge(general,on = 'ticker')
            .sort_values(by='rank', ascending=False)
            .assign(one=1)
            .assign(one=lambda x: x.groupby('sector')['one'].cumsum())
            .loc[lambda x: x['one'] <= n_max_sector]
            .drop(columns='one')
            .loc[lambda x: x['rank'] > 0]  
            .assign(year_month = date_loop)
            )
        final_merged.append(merge_loop)

    # concaténer tous les résultats de merge_loop
    final_result = pd.concat(final_merged, ignore_index=True)
    
    return_model = final_result.merge(monthly_return,
                       how = 'left',
                       on = ['ticker','year_month'])
    
    result_summarised = return_model.groupby(['year_month']).agg(
                            monthly_return=('monthly_return', 'mean'),
                            n =('monthly_return', 'count')).reset_index()
    
    result_summarised_yearly = result_summarised .dropna(subset= ['monthly_return'])
    result_summarised_yearly['year'] = result_summarised_yearly['year_month'].dt.year

    result = pd.DataFrame()
    for year in result_summarised_yearly['year'].unique() : 
        full_monthly_return_loop = (result_summarised_yearly[result_summarised_yearly['year']>= year]
                                    .agg(total_return=('monthly_return', lambda x: np.prod(x)**(12/len(x))-1),
                                         worst=('monthly_return', 'min'),
                                         vol=('monthly_return', 'std'),
                                         n_min = ('n','min'),
                                         n_max = ('n','max'),
                                         n_mean = ('n','mean'))
                                .reset_index()
                                .assign(year = year))
        result = pd.concat([result, full_monthly_return_loop])
    result = result.pivot(index = 'year',columns = 'index',values = ['monthly_return','n']).dropna(axis = 1).reset_index()
    

    return return_model , result_summarised ,result,return_model[return_model['year_month'] == max(return_model['year_month'])]       
def return_benchmark(prices,historical_company,index_price,stocks_filter,sector) : 
    
    prices_dr = (prices
                   .sort_values('month')  # Trier par date
                   .groupby(['ticker'])
                   .apply(lambda x: x.assign(dr=x['close'] / x['close'].shift(1)), include_groups=False)
                   .reset_index()
                   )
    benchmark_base = (prices_dr
              .merge(historical_company,
                     how="inner",
                     on=['month', 'ticker'])
              .groupby(['ticker', 'month'])
              .agg({'dr': 'prod'})
              .reset_index()
              .groupby(['month'])
              .agg({'dr': 'mean'})
              .reset_index()
              .assign(model = 'base')
             )
    
    benchmark_after_selection =  (prices_dr
                                          .merge(stocks_filter[['year_month','ticker']],
                                                 how = "inner",
                                                 left_on = ['month','ticker'],
                                                 right_on = ['year_month','ticker'])
                                          .merge(sector,on = "ticker",how = "left")
                                          .groupby(['ticker', 'month'])
                                          .agg({'dr': 'prod'})
                
                                          .reset_index()
                                          .groupby(['month'])
                                          .agg({'dr': 'mean'})
                                          .reset_index()
                                          .assign(model = 'after_selection')
                                         )
    bench = pd.concat([benchmark_base,
                       benchmark_after_selection,
                       (index_price
                        .rename(columns = {"dr_sp500" : "dr"})
                        .assign(model = 'index'))  ])

    return bench
def scoping_fundamental(balance,
                        cashflow,
                        income,
                        earnings,
                        list_kpi_toincrease = ['totalrevenue_rolling', 'grossprofit_rolling', 'operatingincome_rolling', 'incomebeforeTax_rolling', 'netincome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freecashflow_rolling', 'epsactual_rolling'],
                                 list_ratios_toincrease = ['roic', 'netmargin'],
                                 list_kpi_toaccelerate = ['epsactual_rolling'],
                                 list_lag_increase = [1,4,4*5],
                                 list_ratios_to_augment = ["roic_lag4", "roic_lag1", "netmargin_lag4"],
                                 list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']) :



    # 1. Traitement du bilan comptable
    balance_clean = (
        balance[['ticker', 'date', 'filing_date', 'commonstocksharesoutstanding','totalstockholderequity','netdebt']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
            filing_date_balance =lambda x: pd.to_datetime(x['filing_date'])
        )
        .sort_values('filing_date_balance')
        .groupby(['ticker', 'quarter_end'])
        .last()  # prendre la dernière version du rapport pour chaque trimestre
        .reset_index()
        .drop(columns = ['filing_date'])
    )
    for columns in ['totalstockholderequity','netdebt','commonstocksharesoutstanding'] : 
        balance_clean [f"{columns}_rolling"] =  balance_clean.sort_values(['ticker','filing_date_balance']).groupby('ticker')[columns].transform(lambda x: custom_sma(x, n=4))

    # 2. Traitement des résultats
    earnings_clean = (
        earnings[['ticker', 'date', 'reportdate', 'epsactual']]
        .assign(
            quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
            filing_date_earning =lambda x: pd.to_datetime(x['reportdate'])
        )
        .sort_values('filing_date_earning')
        .groupby(['ticker', 'quarter_end'])
        .last()  # prendre la dernière révision des résultats
        .reset_index()
        .drop(columns = ['reportdate'])
        .dropna(subset = ['epsactual'])
    )
    earnings_clean['epsactual_rolling'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsactual'] \
                                            .transform(lambda x: 4 * custom_sma(x, n=4))
    "earnings_clean['rolling_epsactual'] = earnings_clean.sort_values(['ticker','filing_date_earning']).groupby('ticker')['epsactual'] \
                                            .transform(lambda x: 4 * custom_sma(x, n=4))"
     # 3. income
    columns_to_annualise_income = ['totalrevenue', 'grossprofit', 'operatingincome', 
                       'incomebeforeTax', 'netincome', 'ebit', 'ebitda']
     
    income_clean = (
         income[['ticker', 'date', 'filing_date', 'totalrevenue','grossprofit','operatingincome','incomebeforeTax','netincome','ebit','ebitda']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
             filing_date_income=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_income')
         .groupby(['ticker', 'quarter_end'])
         .last()  # prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    for columns in columns_to_annualise_income : 
        income_clean[f"{columns}_rolling"] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_sma(x, n=4))
        #income_clean[columns] =  income_clean.sort_values(['ticker','filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * custom_sma(x, n=4))
                                          
    cash_clean = (
         cashflow[['ticker', 'date', 'filing_date', 'freecashflow']]
         .assign(
             quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_Period('q'),
             filing_date_cash=lambda x: pd.to_datetime(x['filing_date']))
         .sort_values('filing_date_cash')
         .groupby(['ticker', 'quarter_end'])
         .last()  # prendre la dernière révision des résultats
         .reset_index()
         .drop(columns = ['filing_date'])
     )
    
    for columns in ['freecashflow'] : 
        cash_clean[f"{columns}_rolling"] =  cash_clean.sort_values(['filing_date_cash']).groupby('ticker')[columns].transform(lambda x: 4 * custom_sma(x, n=4))                                        

    funda = (funda
            .drop(columns = ['date','date_x','date_y'])
            .assign(date=funda[list_date_to_maximise].max(axis=1))
            )
    funda = (income_clean
             .merge(cash_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(balance_clean,on = ['ticker','quarter_end'],how = 'outer')
             .merge(earnings_clean[['ticker', 'quarter_end','filing_date_earning', 'epsactual','epsactual_rolling']],on = ['ticker','quarter_end'],how = 'outer')
             .assign(netmargin = lambda x : x['ebit_rolling']/x['totalrevenue_rolling'])
             .assign(roic = lambda x : x['ebit_rolling']/(x['totalstockholderequity_rolling']+x['netdebt_rolling'].fillna(0)))
             .assign(ebitpershare_rolling = lambda x : x['ebit_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
             .assign(ebitdapershare_rolling = lambda x : x['ebitda_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
             .assign(netincomepershare_rolling = lambda x : x['netincome_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
             .assign(fcfpershare_rolling = lambda x : x['freecashflow_rolling']/(x['commonstocksharesoutstanding_rolling'].fillna(0)))
             )
    view = funda[['date','ticker','epsactual_rolling',
                  'ebitpershare_rolling','ebitdapershare_rolling',
                  'netincomepershare_rolling','fcfpershare_rolling']]
    

def compare_models(models_data, start_year=None, end_year=None, risk_free_rate=0.02):
    """
    compare performance metrics for multiple investment models.
    
    parameters:
    -----------
    models_data : dict
        dictionary with model names as keys and DataFrames as values.
        each DataFrame must have a 'year_month' column and a column with returns.
    start_year : int, optional
        starting year for analysis. if None, uses earliest available data.
    end_year : int, optional
        ending year for analysis. if None, uses latest available data.
    risk_free_rate : float, optional
        annual risk-free rate for sharpe ratio calculation. default is 2%.
        
    returns:
    --------
    tuple
        (performance_metrics_df, cumulative_returns_df, correlation_matrix, worst_Periods_df, figures)
        - DataFrame with performance metrics
        - DataFrame with cumulative returns over time
        - correlation matrix of returns
        - DataFrame with worst Periods
        - dictionary of matplotlib figures
    """
    # process and align data
    processed_data = {}
    min_date = pd.Period('2100-01', freq='m')
    max_date = pd.Period('1900-01', freq='m')
    
    for model_name, df in models_data.items():
        # make a copy to avoid modifying the original
        df_copy = df.copy()
        
        # find the returns column - assume it's the column with 'return' in the name or the second column
        return_cols = [col for col in df_copy.columns if 'return' in col.lower()]
        if return_cols:
            return_col = return_cols[0]
        else:
            # if no column has 'return' in name, try to identify it
            if 'monthly_return' in df_copy.columns:
                return_col = 'monthly_return'
            elif len(df_copy.columns) > 1:
                # assume the second column is returns (after year_month)
                return_col = df_copy.columns[1]
            else:
                raise valueerror(f"could not identify returns column for model {model_name}")
        
        # ensure year_month is a Period object
        if not isinstance(df_copy['year_month'].iloc[0], pd.Period):
            try:
                df_copy['year_month'] = df_copy['year_month'].dt.to_Period('m')
            except:
                # if it's not a datetime, try to convert it
                df_copy['year_month'] = pd.to_datetime(df_copy['year_month']).dt.to_Period('m')
        
        # filter by start and end years if provided
        if start_year:
            df_copy = df_copy[df_copy['year_month'].dt.year >= start_year]
        if end_year:
            df_copy = df_copy[df_copy['year_month'].dt.year <= end_year]
            
        if df_copy.empty:
            print(f"warning: no data available for {model_name} in selected time Period")
            continue
            
        # update min and max dates
        min_date = min(min_date, df_copy['year_month'].min())
        max_date = max(max_date, df_copy['year_month'].max())
        
        # create a clean Series with Period index and returns
        returns_Series = df_copy.set_index('year_month')[return_col]
        processed_data[model_name] = returns_Series
    
    # combine all returns into a single DataFrame
    all_returns = pd.DataFrame(processed_data)
    
    # calculate correlation matrix
    correlation_matrix = all_returns.corr()
    
    # convert Period index to timestamp for easier plotting
    all_returns_ts = all_returns.copy()
    all_returns_ts.index = all_returns_ts.index.to_timestamp()
    all_returns_ts.sort_index(inplace=True)
    
    # calculate cumulative returns (starting with $1)
    cumulative_returns = (all_returns_ts + 1).cumprod()
    
    # calculate worst Periods
    worst_Periods = {}
    for model in all_returns.columns:
        # add year and month columns
        model_returns = all_returns[model].reset_index()
        model_returns['year'] = model_returns['year_month'].dt.year
        model_returns['month'] = model_returns['year_month'].dt.month
        
        # worst month
        worst_month_idx = model_returns[model].idxmin()
        worst_month = model_returns.loc[worst_month_idx]
        worst_month_date = worst_month['year_month']
        worst_month_return = worst_month[model]
        
        # calculate annual returns
        annual_returns = model_returns.groupby('year')[model].apply(
            lambda x: np.prod(1 + x) - 1
        )
        
        # worst year
        worst_year_idx = annual_returns.idxmin()
        worst_year_return = annual_returns.loc[worst_year_idx]
        
        worst_Periods[model] = {
            'worst month': f"{worst_month_date.strftime('%y-%m')}: {worst_month_return:.2%}",
            'worst year': f"{worst_year_idx}: {worst_year_return:.2%}"
        }
    
    worst_Periods_df = pd.DataFrame(worst_Periods).T
    
    # calculate annual returns for each model
    annual_returns_data = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].reset_index()
        model_returns['year'] = model_returns['year_month'].dt.year
        annual_returns = model_returns.groupby('year')[model].apply(
            lambda x: np.prod(1 + x) - 1
        )
        annual_returns_data[model] = annual_returns
    
    annual_returns_df = pd.DataFrame(annual_returns_data)
    
    # calculate performance metrics
    metrics = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].dropna()
        model_returns_ts = all_returns_ts[model].dropna()
        
        # skip if insufficient data
        if len(model_returns) < 12:
            print(f"warning: insufficient data for {model}")
            continue
            
        # calculate metrics
        total_months = len(model_returns)
        total_years = total_months / 12
        
        # returns
        total_return = cumulative_returns[model].iloc[-1] - 1
        annualized_return = (1 + total_return) ** (1 / total_years) - 1
        monthly_mean = model_returns.mean()
        
        # risk metrics
        monthly_std = model_returns.std()
        annualized_vol = monthly_std * np.sqrt(12)
        sharpe = (annualized_return - risk_free_rate) / annualized_vol
        
        # drawdown calculation
        rolling_max = cumulative_returns[model].cummax()
        drawdown = (cumulative_returns[model] / rolling_max - 1)
        max_drawdown = drawdown.min()
        
        # positive months
        positive_months = (model_returns > 0).sum() / total_months
        
        # calculate cagr for different time Periods
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
        
        # store metrics
        metrics[model] = {
            'start date': min_date.strftime('%y-%m'),
            'end date': max_date.strftime('%y-%m'),
            'Total return': total_return,
            'cagr': annualized_return,
            'cagr (3y)': cagr_3yr,
            'cagr (5y)': cagr_5yr,
            'cagr (10y)': cagr_10yr,
            'monthly mean': monthly_mean,
            'monthly volatility': monthly_std,
            'annualized volatility': annualized_vol,
            'sharpe ratio': sharpe,
            'max drawdown': max_drawdown,
            'positive months %': positive_months,
            'number of stocks (avg)': models_data[model]['n'].mean() if 'n' in models_data[model].columns else None
        }
    
    # convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics).T
    
    # format percentages and ratios
    for col in ['Total return', 'cagr', 'cagr (3y)', 'cagr (5y)', 'cagr (10y)', 
                'monthly mean', 'monthly volatility', 'annualized volatility', 
                'max drawdown', 'positive months %']:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "n/a")
    
    if 'sharpe ratio' in metrics_df.columns:
        metrics_df['sharpe ratio'] = metrics_df['sharpe ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "n/a")
    
    # calculate cagr since each year
    cagr_by_year = calculate_cagr_by_year(all_returns)
    
    # create combined visualization
    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # plot 1: cumulative returns (top left)
    ax1 = plt.subplot(2, 2, 1)
    cumulative_returns.plot(ax=ax1)
    ax1.set_title('cumulative returns', fontsize=16)
    ax1.set_ylabel('value of $1 investment', fontsize=12)
    ax1.grid(True)
    ax1.set_yscale('log')  # log scale for better visualization
    ax1.legend(fontsize=10)
    
    # plot 2: correlation matrix (top right)
    ax2 = plt.subplot(2, 2, 2)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                mask=mask, vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('return correlation matrix', fontsize=16)
    
    # plot 3: cagr by start year (bottom left)
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(cagr_by_year, annot=True, fmt=".1%", cmap="rdylgn", ax=ax3)
    ax3.set_title('cagr by start year', fontsize=16)
    ax3.set_ylabel('start year', fontsize=12)
    ax3.set_xlabel('model', fontsize=12)
    
    # plot 4: annual returns (bottom right)
    ax4 = plt.subplot(2, 2, 4)
    sns.heatmap(annual_returns_df, annot=True, fmt=".1%", cmap="rdylgn", ax=ax4)
    ax4.set_title('annual returns by year', fontsize=16)
    ax4.set_ylabel('year', fontsize=12)
    ax4.set_xlabel('model', fontsize=12)
    
    plt.tight_layout()
    """"""
    # also create individual heatmaps for each model
    individual_heatmaps = {}
    for model in all_returns.columns:
        fig_heatmap = plot_monthly_returns_heatmap(all_returns[model], model)
        individual_heatmaps[model] = fig_heatmap
    
    figures = {
        'main_figure': fig,
        'monthly_heatmaps': individual_heatmaps
    }
    """"""
    return metrics_df, cumulative_returns, correlation_matrix, worst_Periods_df, figures
def plot_monthly_returns_heatmap(returns_Series, model_name):
    """
    create a heatmap of monthly returns for a single model.
    
    parameters:
    -----------
    returns_Series : Series
        Series with Period index and monthly returns as values.
    model_name : str
        name of the model for the plot title.
        
    returns:
    --------
    matplotlib.figure.figure
        figure containing the heatmap
    """
    # make a copy and reset index
    df = returns_Series.reset_index()
    
    # extract year and month
    df['year'] = df['year_month'].dt.year
    df['month'] = df['year_month'].dt.month
    
    # pivot the data for the heatmap
    heatmap_data = df.pivot(index='year', columns='month', values=model_name)
    
    # create a figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # create a custom colormap centered at 0
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    # create the heatmap
    sns.heatmap(heatmap_data, 
                cmap=cmap, 
                center=0,
                annot=True, 
                fmt='.1%', 
                linewidths=.5, 
                ax=ax,
                cbar_kws={'label': 'monthly return'})
    
    # set month names
    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    ax.set_xticklabels(month_names)
    
    # set title
    ax.set_title(f'monthly returns heatmap - {model_name}', fontsize=16)
    
    plt.tight_layout()
    return fig
def calculate_cagr_by_year(returns_df):
    """
    calculate cagr starting from each year for each model.
    
    parameters:
    -----------
    returns_df : DataFrame
        DataFrame with Period index and monthly returns for each model as columns.
        
    returns:
    --------
    DataFrame
        DataFrame with start years as index and models as columns, values are cagr.
    """
    # get unique years
    years = sorted(set(returns_df.index.year))
    
    # initialize results DataFrame
    cagr_results = {}
    
    # for each model
    for model in returns_df.columns:
        model_cagr = {}
        
        # for each start year
        for start_year in years:
            # filter data starting from this year
            filtered_returns = returns_df.loc[returns_df.index.year >= start_year, model]
            
            # skip if less than 12 months of data
            if len(filtered_returns) < 12:
                model_cagr[start_year] = np.nan
                continue
            
            # calculate total return
            total_return = (1 + filtered_returns).prod() - 1
            
            # calculate years
            years_count = len(filtered_returns) / 12
            
            # calculate cagr
            cagr = (1 + total_return) ** (1 / years_count) - 1
            model_cagr[start_year] = cagr
        
        cagr_results[model] = model_cagr
    
    # convert to DataFrame
    cagr_df = pd.DataFrame(cagr_results)
    
    # drop rows where all values are nan
    cagr_df = cagr_df.dropna(how='all')
    
    return cagr_df
