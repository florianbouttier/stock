# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 15:43:16 2024

@author: flbouttier
"""


# %%
import os
import numpy as np
import scipy.stats as stats
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
import re 
import seaborn as sns
from scipy.stats import rankdata
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
#sys.path.append(os.path.dirname(os.getcwd()))

env_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'stock')
os.chdir(env_dir)
import src.functions_backtest as fb

#env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
env_dir = os.path.join(os.path.expanduser('~'), 'Documents', 'stock')
#env_dir = os.getcwd()
data_dir = os.path.join(env_dir, 'data')
os.chdir(data_dir)
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

def technical_data(data,n_long,n_short,func_movingaverage = fb.ema_moving_average) : 
    df = data

    for n, col in [(n_short, 'Short_VS_SP500'), (n_long, 'Long_VS_SP500')]:
        df[col] = df.groupby('ticker')['Close_VS_SP500'].transform(lambda x: func_movingaverage(x, n=n))
    
    df = df.groupby('ticker').apply(lambda x: x.iloc[n_long:],include_groups = False).reset_index(drop=False)

    df = df.dropna(subset = ['Short_VS_SP500','Long_VS_SP500'])
    pd.options.mode.chained_assignment = None  # default='warn'
    df.loc[:, 'MTR'] = df['Short_VS_SP500'] / df['Long_VS_SP500']
    return df
def quantile(values,method):
    
    if method == "rank" : 
        quantiles = rankdata(values.fillna(values.min() - 1), method="average") / len(values)
        quantiles = np.where(quantiles == 0, -np.inf, quantiles)
    if method != "rank" : 
        m = values.mean()
        std = values.std()
        quantiles =stats.norm.cdf(values, m, std)
    
    return quantiles
funda = fb.calculate_fundamental_ratios(balance = Balance_Sheet,
                                 cashflow = Cash_Flow,
                                 income = Income_Statement,
                                 earnings = Earnings,
                                 list_kpi_toincrease = ['totalRevenue_rolling', 'grossProfit_rolling', 'operatingIncome_rolling', 'incomeBeforeTax_rolling', 'netIncome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freeCashFlow_rolling', 'epsActual_rolling'],
                                 list_ratios_toincrease = ['ROIC', 'NetMargin'],
                                 list_kpi_toaccelerate = ['epsActual_rolling'],
                                 list_lag_increase = [1,4,4*5],
                                 list_ratios_to_augment = ["ROIC_lag4", "ROIC_lag1", "NetMargin_lag4"],
                                 list_date_to_maximise = ['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning'])

technical_df = technical_data(
    data = fb.Price_VS_Index(Index = SP500Price.copy(),Prices = Finalprice.copy()),
    n_long = 150,
    n_short = 10,
    func_movingaverage = fb.ema_moving_average)


technical_df['year_month'] =  pd.to_datetime(technical_df['date']).dt.to_period('M')
funda['year_month'] =  pd.to_datetime(funda['date']).dt.to_period('M')
month_list = funda[funda['year_month'] >= '2000-01']['year_month'].unique()
full_db = pd.DataFrame()
columns_learning =['NetMargin','ROIC', 
          'eps_fcf','eps_netincome', 'totalRevenue_rolling_lag1',
        'totalRevenue_rolling_lag4', 'totalRevenue_rolling_lag20',
       'grossProfit_rolling_lag1', 'grossProfit_rolling_lag4',
       'grossProfit_rolling_lag20', 'operatingIncome_rolling_lag1',
       'operatingIncome_rolling_lag4', 'operatingIncome_rolling_lag20',
       'incomeBeforeTax_rolling_lag1', 'incomeBeforeTax_rolling_lag4',
       'incomeBeforeTax_rolling_lag20', 'netIncome_rolling_lag1',
       'netIncome_rolling_lag4', 'netIncome_rolling_lag20',
       'ebit_rolling_lag1', 'ebit_rolling_lag4', 'ebit_rolling_lag20',
       'ebitda_rolling_lag1', 'ebitda_rolling_lag4', 'ebitda_rolling_lag20',
       'freeCashFlow_rolling_lag1', 'freeCashFlow_rolling_lag4',
       'freeCashFlow_rolling_lag20', 'epsActual_rolling_lag1',
       'epsActual_rolling_lag4', 'epsActual_rolling_lag20', 'ROIC_lag1',
       'ROIC_lag4', 'ROIC_lag20', 'NetMargin_lag1', 'NetMargin_lag4',
       'NetMargin_lag20', 'epsActual_rolling_lag1_lag1',
       'epsActual_rolling_lag4_lag1', 'epsActual_rolling_lag20_lag1', 
       'ROIC_lag4_days_increase', 'ROIC_lag1_days_increase',
       'NetMargin_lag4_days_increase','MTR']
method = "notrank"
returns = fb.calculate_monthly_returns(Finalprice)
for month in tqdm(sorted(month_list)[10:]) : 
    
    ticker = US_historical_company[US_historical_company['Month'] == month]['ticker'].unique()
    funda_loop = funda[(funda['year_month'] < month)]
    funda_loop = funda_loop[funda_loop['ticker'].isin(ticker)]
    funda_loop = funda_loop[(funda_loop['date'] == funda_loop.groupby('ticker')['date'].transform('max'))]
    funda_loop = funda_loop.copy()
    funda_loop['year_month'] = month
   
    technical_df_loop = technical_df[(technical_df['year_month'] < month) & (technical_df['ticker'].isin(ticker))]
    technical_df_loop = technical_df_loop[(technical_df_loop['date'] == technical_df_loop.groupby('ticker')['date'].transform('max'))]
    technical_df_loop = technical_df_loop.copy()
    technical_df_loop['year_month'] = month
    
    learning_db_loop = funda_loop.merge(technical_df_loop[['ticker','MTR']],on = 'ticker',how = 'left')
    learning_db_loop = learning_db_loop[['ticker']+columns_learning]
    for c in columns_learning : 
        learning_db_loop[c] = quantile(learning_db_loop[c],method = method)
    
    learning_db_loop = learning_db_loop.merge(General[['ticker','Sector']],on = "ticker",how = "left")
    learning_db_loop = pd.get_dummies(learning_db_loop, columns=['Sector'], prefix='Sector')
    
    for col in learning_db_loop.columns:
        if learning_db_loop[col].dtype == 'bool':
            learning_db_loop[col] = learning_db_loop[col].astype(int)

    returns_loop = returns[returns['year_month'] == month]
    returns_loop = returns_loop[returns_loop['ticker'].isin(ticker)].copy()
    returns_loop['monthly_return'] = returns_loop['monthly_return']-returns_loop['monthly_return'].mean()
    min_val = returns_loop['monthly_return'].min()
    max_val = returns_loop['monthly_return'].max()
    returns_loop['score'] = (returns_loop['monthly_return'] - min_val) / (max_val - min_val)
    returns_loop['year_month'] = month
    full_db = pd.concat([full_db,
                         learning_db_loop.merge(returns_loop[['ticker','score','year_month']],on = 'ticker',how = "left")])
    


# Fonction de préparation des données pour LSTM
def reshape_for_lstm(df, features, label, sequence_length=6):
    df = df.dropna(subset=features + [label])
    df = df.sort_values(['ticker', 'year_month'])
    
    X, y,lab = [], [],[]
    for ticker, group in df.groupby('ticker') :
        for i in range(len(group) - sequence_length):
            seq_x = group.iloc[i:i+sequence_length][features].values
            seq_y = group.iloc[i+sequence_length][label]
            seq_label = group.iloc[i:i+sequence_length][['ticker','year_month']].values
            X.append(seq_x)
            y.append(seq_y)
            lab.append(seq_label)
    return np.array(X), np.array(y),np.array(lab)

# Entraînement du modèle LSTM
def train_lstm(X_train, y_train):
    model = Sequential([
        LSTM(100, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.8),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history  = model.fit(X_train, y_train, epochs=200, batch_size=32)
    return model
def predict_and_backtest_recursive(df, features, returns, sequence_length=6, top_k=10):
    full_predictions = []
    results = []
    
    # Itérer sur chaque mois en commençant à partir du mois "sequence_length" pour avoir assez de données
    for i, month in enumerate(sorted(df['year_month'].unique())[sequence_length:]):
        # Sélectionner les données jusqu'au mois précédent
        ticker = df[df['year_month'] == str(month)]['ticker']
        df_train = df[(df['year_month'] <= month)]
        df_train = df_train[df_train['ticker'].isin(ticker)]
        
        
        X_train, y_train,label = reshape_for_lstm(df_train, features, label='score', sequence_length=sequence_length)
        
        # Créer et entraîner le modèle sur les données passées
        label_last_month = []
        for i in range(len(label)) : 
            print(label[i][-1][1])
            label_loop = label[i][-1][1] == month
            label_last_month.append(label[i][-1][1])
            
        X_train[label_last_month]
        model = train_lstm(X_train, y_train)  # Train LSTM on past data
        
        # Sélectionner les données pour le mois actuel
        month_data = df[df['year_month'] <= month]
        X_test, y_test = reshape_for_lstm(month_data, features, label='score', sequence_length=sequence_length)
        
        # Faire des prédictions avec le modèle entraîné
        preds = model.predict(X_test)
        month_data['predicted_score'] = preds
        
        # Sélectionner les top_k actions par score prédit
        top_actions = month_data.nlargest(top_k, 'predicted_score')
        
        # Calculer le rendement pour les top_k actions
        returns_month = returns[returns['year_month'] == month]
        top_returns = returns_month[returns_month['ticker'].isin(top_actions['ticker'])]
        
        # Calculer le rendement moyen du portefeuille
        portfolio_return = top_returns['monthly_return'].mean()
        
        # Enregistrer les prédictions et le rendement
        full_predictions.append(top_actions)
        results.append(portfolio_return)
        
        # À chaque itération, réentraîner le modèle sur les mois passés jusqu'au mois actuel pour la prochaine itération
        # (le modèle pour le mois actuel est déjà prêt pour la prochaine itération)
    
    # Retourner les résultats du backtest
    return full_predictions, results
# Exemple d'utilisation
features = [col for col in full_db.columns if col not in ['ticker', 'year_month', 'score']]

top_actions_per_month, portfolio_returns = predict_and_backtest(full_db, features, returns, sequence_length=6, top_k=10)

# Résultats du backtest (rendements du portefeuille)
print("Rendement du portefeuille pour chaque mois : ", portfolio_returns)
# %%
