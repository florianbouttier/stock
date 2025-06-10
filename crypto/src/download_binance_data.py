# %%
from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# %% function
client = Client(api_key='', api_secret='')

def download_binance_data(
    tickers=["ETHUSDT", "BTCUSDT", "ETHBTC"],
    interval=Client.KLINE_INTERVAL_8HOUR,
    start_str="1 Jan, 2017",
    end_str=None):
    """
    Télécharge les données OHLCV pour une liste de tickers depuis Binance.

    Params:
        tickers (list): Liste de symboles (ex: ["ETHUSDT", "BTCUSDT"])
        interval (str): Intervalle Binance (ex: Client.KLINE_INTERVAL_1HOUR)
        start_str (str): Date de début ("1 Jan, 2022")
        end_str (str): Date de fin (None = maintenant)

    Returns:
        dict: {ticker: DataFrame}
    """
    data = {}

    for ticker in tickers:
        print(f"Téléchargement de {ticker} ({interval})...")
        klines = client.get_historical_klines(ticker, interval, start_str, end_str)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit='ms')
        df.set_index("timestamp", inplace=True)
        df = df.astype(float, errors='ignore')
        data[ticker] = df[["open", "high", "low", "close", "volume"]]

    return data

def download_binance_data_concat(
    tickers=["ETHUSDT", "BTCUSDT", "ETHBTC"],
    interval=Client.KLINE_INTERVAL_8HOUR,
    start_str="1 Jan, 2017",
    end_str=None) -> pd.DataFrame:
    """
    Télécharge et concatène les données OHLCV pour plusieurs tickers depuis Binance.

    Returns:
        DataFrame: Multi-index DataFrame (timestamp, ticker)
    """
    raw_data = download_binance_data(tickers, interval, start_str, end_str)
    df_list = []
    for ticker, df in raw_data.items():
        df["ticker"] = ticker
        df_list.append(df)

    full_df = pd.concat(df_list)
    full_df.reset_index(inplace=True)  # remet timestamp en colonne
    return full_df

def validate_and_fix_timestamps(df, freq='8H') -> pd.DataFrame:
    """
    Valide que chaque ticker a des timestamps complets toutes les `freq` heures,
    et insère des lignes manquantes si nécessaire (avec NaN).
    Calcul du Daily Return (DR) pour chaque crypto (ticker).
    
    Params:
        df (DataFrame): contient au moins les colonnes ['timestamp', 'ticker']
        freq (str): fréquence d'échantillonnage ('8H' par défaut)
    
    Returns:
        DataFrame: avec tous les timestamps requis, NaN pour les données manquantes, et DR calculé
    """
    all_fixed = []
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    tickers = df['ticker'].unique()

    for ticker in tickers:
        sub_df = df[df['ticker'] == ticker].copy()
        sub_df.set_index('timestamp', inplace=True)

        sub_df = sub_df.sort_index()

        start, end = sub_df.index.min(), sub_df.index.max()
        full_range = pd.date_range(start=start, end=end, freq=freq)
        
        sub_df = sub_df.reindex(full_range)
        sub_df['ticker'] = ticker
        all_fixed.append(sub_df)

        # Calcul du Daily Return (DR) : (close_today - close_yesterday) / close_yesterday
        sub_df['DR'] = sub_df['close'].pct_change()  # pct_change() calcule la variation par rapport à la clôture précédente

    fixed_df = pd.concat(all_fixed)
    fixed_df.reset_index(inplace=True)
    fixed_df.rename(columns={'index': 'timestamp'}, inplace=True)

    # Remplir les valeurs manquantes pour open, high, low, close par ticker
    columns_to_fill = ['open', 'high', 'low', 'close']
    fixed_df[columns_to_fill] = fixed_df.groupby('ticker')[columns_to_fill].transform(lambda group: group.ffill())
    
    return fixed_df

def func_training_1d(prices, fee, List_N, func_MovingAverage):
    
    if func_MovingAverage is None:
            func_MovingAverage = lambda x, n: x.ewm(span=n, adjust=False).mean()

    def Learning(N, func_MovingAverage):
        
        df = prices.copy()
        df = df.sort_values(by=['ticker', 'timestamp'])
        
        # Calcul de la courbe EMA
        df['EMA_Curve'] = df.groupby(['ticker'])['close'].transform(lambda x: func_MovingAverage(x, N))
        df = df.dropna(subset=['EMA_Curve'])
        df_output = pd.DataFrame()
        for ticker in df['ticker'].unique():
            sub_df = df[df['ticker'] == ticker].copy()
            sub_df = sub_df.sort_values(by='timestamp')  

            sub_df['EMA_Curve'] = func_MovingAverage(sub_df['close'], N)

            sub_df['Buying_Decision'] = (sub_df['EMA_Curve'] < sub_df['close']).astype(int)
            sub_df['Selling_Signal'] = (-1 * (sub_df['low'] < sub_df['EMA_Curve'].shift(1))).astype(int)

            sub_df['DR_strategy'] = 1 + sub_df['Buying_Decision'].shift(1) * sub_df['DR']
            sub_df['Buying_Movement_1'] = ((sub_df['Buying_Decision'].shift(1) == 0) & (sub_df['Buying_Decision'] == 1)).astype(int)
            sub_df['Sell_Movement_1'] = ((sub_df['Selling_Signal'] == -1) & (sub_df['Buying_Decision'].shift(1) == 1)).astype(int)
            sub_df['DR_fees'] = (1 - fee * (sub_df['Buying_Movement_1'] + sub_df['Sell_Movement_1']))
            sub_df['DR_strategy'] *= sub_df['DR_fees']
 
            sub_df['N_EMA'] = N
            sub_df['DR'] = 1 + sub_df['DR']
            df_output = pd.concat([df_output,sub_df])
        return df_output

    # Appliquer la fonction Learning sur chaque valeur de N dans List_N
    df_final = pd.concat([Learning(N, func_MovingAverage) for N in List_N])
    return df_final

def extract_symbol_and_ref(df, ref_cur=['USDC', 'BTC']):
   
    
    def extract_symbol_and_ref(ticker, ref_currencies=['USDC', 'BTC']):
        for ref in ref_currencies:
            if ticker.endswith(ref):
                symbol = ticker.replace(ref, '')  # Retire la devise de référence
                return symbol, ref
        return None, None 
    df['symbol'], df['ref_cur'] = zip(*df['ticker'].apply(lambda x: extract_symbol_and_ref(x, ref_cur)))
    df['temporality'] = df['timestamp'].dt.to_period('Q')
    return df

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

def score(x,alpha) : 
    x = x-1
    sco = np.log(1+alpha*x)
    sco = np.where(np.isnan(sco), -np.inf, sco)
    return sco

def best_model_determination(perf_data,mode,list_paramtemp,list_alpha,grouping_columns = ['N_EMA','symbol','ref_cur'],index = ['symbol','ref_cur']) : 
    
    def best_model_at_date(perf_data,temp, mode,param_alpha, param_temp):
        
        filtered_data = perf_data[perf_data['temporality'] < temp]

        summarized_data  = (filtered_data
                .sort_values(by='temporality', ascending=False) 
                .groupby(grouping_columns, group_keys=False)
                .apply(lambda x: decreasing_sum(x['score'], halfperiod=param_temp, mode=mode),include_groups = False)
                .rename('score')  # Renomme la colonne résultante en 'Score'
                .reset_index(name='score') 
                )
       
        best_model = (summarized_data[summarized_data['score'] == summarized_data.groupby(index)['score'].transform('max')])
        best_model = best_model.groupby(index, as_index=False).head(1)
        best_model = best_model.assign(
                              temporality=temp,
                              param_alpha=param_alpha,
                              param_temp=param_temp
                              )

        return best_model
    def best_model_at_alpha(perf_data,mode,alpha, list_temp) : 
        perf_data['score_strategy'] = score(perf_data['DR_strategy'],
                                            alpha = alpha)
        perf_data['score_relativestrategy'] = score(perf_data['DR_strategy']/perf_data['DR'],
                                                    alpha = alpha)
        perf_data['score'] = perf_data[['score_strategy', 'score_relativestrategy']].min(axis=1).fillna(-np.inf)

        results = []
        list_temp = perf_data['temporality'].unique()
        list_temp = list_temp[4:]
        for temp in list_temp :
            for paramtemp in list_paramtemp : 
                results_loop = best_model_at_date(perf_data,
                                                  temp, 
                                                  mode,
                                                  param_alpha=alpha, 
                                                  param_temp = paramtemp)
        
                results.append(results_loop )
        return pd.concat(results, ignore_index=True) 

    result = []
    for alpha in tqdm(list_alpha) : 
       r  =  best_model_at_alpha(perf_data = perf_data,
                                 mode = mode,
                                 alpha = alpha, 
                                 list_temp = list_paramtemp)
       result.append(r)
    return pd.concat(result, ignore_index=True) 

def func_trading_alt(training, ref="BTC",ref_USD = 'USDC', fee=0.075):
    # Calcul des symboles contre USD
    training = training.copy()
    #ETHVSUSD
    training_symbol_USD = training[(training['ref_cur'] == ref_USD) & (training['symbol'] != ref)].copy()
    training_symbol_USD = training_symbol_USD.rename(columns={"Buying_Decision": "buy_symbol_vs_usd","DR_strategy": "DR_symbol_vs_usd"})
    training_symbol_USD = training_symbol_USD[['symbol','timestamp','buy_symbol_vs_usd','DR_symbol_vs_usd']]
     
    #BTCVSUSD
    training_refcur_usd = training[(training['ref_cur'] == ref_USD) & (training['symbol'] == ref)].copy()
    training_refcur_usd = training_refcur_usd.rename(columns={"Buying_Decision": "buy_refcur_vs_usd","DR_strategy": "DR_refcur_vs_usd"})
    training_refcur_usd = training_refcur_usd[['timestamp','buy_refcur_vs_usd','DR_refcur_vs_usd']]
    
    #ETHVSBTC
    training_symbol_refcur = training[(training['ref_cur'] == ref) & (training['symbol'] != ref)].copy()
    training_symbol_refcur = training_symbol_refcur.rename(columns={"Buying_Decision": "buy_symbol_VS_refcur"})
    training_symbol_refcur = training_symbol_refcur[['symbol','timestamp','buy_symbol_VS_refcur']]
    
    #merge
    training_merge = training_symbol_USD.merge(training_refcur_usd, on="timestamp", how="left")
    training_merge = training_merge.merge(training_symbol_refcur, on=["timestamp", "symbol"], how="left")
    
    training_merge['DR_strategy_2d'] = np.where(training_merge['buy_symbol_VS_refcur'].shift(1) == 0, 
                                                 training_merge['DR_refcur_vs_usd'], 
                                                 training_merge['DR_symbol_vs_usd'])
    training_merge['DR_new_fees'] = 1 - fee * np.where((training_merge['buy_symbol_VS_refcur'].shift(1) != training_merge['buy_symbol_VS_refcur']) & (training_merge['DR_strategy_2d'] != 1) & (training_merge['DR_strategy_2d'].shift(1) != 1), 1, 0)
    training_merge['DR_strategy_2d'] *= training_merge['DR_new_fees']
    
    training_merge = (training_merge.merge(training[(training['ref_cur'] == ref_USD) & 
                                                   (training['symbol'] != ref)][['timestamp',
                                                                                 'symbol',
                                                                                 'DR']], 
                                          on=["timestamp", "symbol"], how="left").
                      rename(columns = {'DR' : 'DR_symbol'}))
    
    training_merge = (training_merge.merge(training[(training['ref_cur'] == ref_USD) & 
                                                   (training['symbol'] == ref)][['timestamp']], 
                                          on=["timestamp"], how="left").
                      rename(columns = {'DR' : 'DR_ref'}))
    #training_merge = Test_.merge(Test[(Test['ref_cur'] == "USD") & (Test['symbol'] != ref)], on=["time_close", "symbol"], how="left")
    #training_merge = Test_.merge(Test[(Test['ref_cur'] == "USD") & (Test['symbol'] == ref)], on="time_close", how="left")
    
    # Joindre avec les DR bruts pour symboles et référence
    #Test_ = Test_.merge(Test[(Test['ref_cur'] == "USD") & (Test['symbol'] != ref)], on=["time_close", "symbol"], how="left")
    #Test_ = Test_.merge(Test[(Test['ref_cur'] == "USD") & (Test['symbol'] == ref)], on="time_close", how="left")
    
    return training_merge

# %% DL and retreat data
data_df_USDT = extract_symbol_and_ref(
    download_binance_data_concat(
        tickers=["ETHUSDT", "BTCUSDT", "ETHBTC"],
        start_str="1 Jan, 2017"),
    ref_cur=['USDT', 'BTC']
)

data_df_USDC = extract_symbol_and_ref(
    download_binance_data_concat(
        tickers=["ETHUSDC", "BTCUSDC", "ETHBTC"],
        start_str="1 Jan, 2017"),
    ref_cur=['USDC', 'BTC']
)

# Agrégation avec priorité USDC > USDT
def merge_usdc_usdt(usdc_df, usdt_df):
    
    usdc_df['source'] = 1
    usdt_df['source'] = 0
    
    merged = pd.concat([usdc_df,
                        usdt_df])
    merged['ref_cur'] =merged['ref_cur'].str[:3]
    merged['ticker'] = merged['symbol']+merged['ref_cur']

    merged_ = merged.groupby(['ticker','timestamp']).transform('max').reset_index()
    merge_cols = ['timestamp', 'symbol', 'ref_cur']

    # Garder seulement les colonnes communes
    common_cols = list(set(usdc_df.columns).intersection(set(usdt_df.columns)))

    # Fusion outer puis priorité à USDC
    merged = pd.concat([usdc_df[common_cols], usdt_df[common_cols]])
    merged = merged.sort_values(by=['timestamp', 'symbol', 'ref_cur', 'source'], ascending=[True, True, True, True])
    merged = merged.drop_duplicates(subset=['timestamp', 'symbol', 'ref_cur'], keep='first')
    merged['ref_cur'] =merged['ref_cur'].str[:3]
    merged['ticker'] = merged['symbol']+merged['ref_cur']
    merged = merged[['timestamp','ticker','open','high', 'low','close','volume','temporality',
                    'symbol','ref_cur']]
    
    
    usdc_df['source'] = 'USDC'
    usdt_df['source'] = 'USDT'

    # Colonnes clés pour fusion
    merge_cols = ['timestamp', 'symbol', 'ref_cur']

    # Garder seulement les colonnes communes
    common_cols = list(set(usdc_df.columns).intersection(set(usdt_df.columns)))

    # Fusion outer puis priorité à USDC
    merged = pd.concat([usdc_df[common_cols], usdt_df[common_cols]])
    merged = merged.sort_values(by=['timestamp', 'symbol', 'ref_cur', 'source'], ascending=[True, True, True, True])
    merged = merged.drop_duplicates(subset=['timestamp', 'symbol', 'ref_cur'], keep='first')
    merged['ref_cur'] =merged['ref_cur'].str[:3]
    merged['ticker'] = merged['symbol']+merged['ref_cur']
    merged = merged[['timestamp','ticker','open','high', 'low','close','volume','temporality',
                    'symbol','ref_cur']]
    return merged


data_df_USDC = validate_and_fix_timestamps(data_df_USDC)
data_df_USDT = validate_and_fix_timestamps(data_df_USDT)
#data_df = merge_usdc_usdt(data_df_USDC, data_df_USDT)
data_df = data_df_USDT
# %%
mult = 3
training = func_training_1d(prices=data_df,
                            fee = 0.075/100,
                            List_N= [mult*(2+2*i) for i in range(40)],
                            func_MovingAverage = None)
training = extract_symbol_and_ref(training,ref_cur=['USDT', 'BTC'])
perf_training = (
    training.groupby(['temporality','symbol','ref_cur','N_EMA']).agg({'DR': 'prod', 'DR_strategy': 'prod'}).reset_index()
)
best_model = best_model_determination(
    perf_data = perf_training,
    mode = "exponential",
    list_paramtemp = [4,6,8,10,12,14,16,18,20,22,24],
    list_alpha= [1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3]
    ).reset_index()



perf_per_model = pd.merge(perf_training.reset_index(),
                          best_model.reset_index() ,
                          how = "inner",
                          on = ['temporality','N_EMA','symbol','ref_cur'])

print(np.max(perf_per_model.groupby(['param_alpha','param_temp','temporality','symbol','ref_cur'])['N_EMA'].count()))

perf_per_model['model'] = perf_per_model['param_alpha'].astype('str') +'-'+ perf_per_model['param_temp'].astype('str')               
best_model_lvl2 = best_model_determination(
    perf_data = perf_per_model,
    mode = "exponential",
    list_paramtemp = [12],
    list_alpha= [1.6],
    grouping_columns = ['model','symbol','ref_cur'],
    index = ['symbol','ref_cur']
    )

final_model = pd.merge(best_model_lvl2,
                       perf_per_model[['temporality','model','symbol','ref_cur','N_EMA']],
                       how = "inner",
                       on = ['temporality','model','symbol','ref_cur'])

training_ = pd.merge(training,
                     final_model,
                     how = "inner",
                     on = ['temporality','N_EMA','symbol','ref_cur'])

print(np.max(training_.groupby(['timestamp','symbol','ref_cur'])['DR'].count()))
results_2d = func_trading_alt(training_, ref="BTC",ref_USD = 'USDT', fee=0.075/100)
print(np.max(A.groupby(['timestamp'])['symbol'].count()))

# %%

def calculate_returns_and_plot(df, period='year'):
    """
    Calcule le retour par période (année ou trimestre) pour chaque modèle et génère un heatmap.
    
    Args:
        df (pandas.DataFrame): DataFrame avec les colonnes 'timestamp' et plusieurs colonnes de modèles
                              commençant par 'DR_'.
        period (str): 'year' pour agrégation annuelle, 'quarter' pour agrégation trimestrielle.
    
    Returns:
        pandas.DataFrame: DataFrame des retours calculés.
    """
    # Vérification que le dataframe contient les colonnes nécessaires
    if 'timestamp' not in df.columns:
        raise ValueError("Le dataframe doit contenir une colonne 'timestamp'")
    
    # Conversion de timestamp en datetime si ce n'est pas déjà fait
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extraction de l'année et du trimestre
    df['year'] = df['timestamp'].dt.year
    if period == 'quarter':
        df['quarter'] = df['timestamp'].dt.quarter
        df['period'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)
    else:
        df['period'] = df['year'].astype(str)
    
    # Identification des colonnes de modèles (commençant par 'DR_')
    model_columns = [col for col in df.columns if col.startswith('DR_')]
    
    if not model_columns:
        raise ValueError("Aucune colonne commençant par 'DR_' n'a été trouvée")
    
    # Initialisation du DataFrame pour stocker les résultats
    results = []
    
    # Calcul des retours pour chaque modèle et chaque période
    for period_name in sorted(df['period'].unique()):
        period_data = df[df['period'] == period_name]
        
        for model in model_columns:
            # Calcul du retour cumulatif pour la période
            # Pour les modèles DR_, nous avons des valeurs comme 1.2 (qui signifie +20%)
            # Nous calculons le produit cumulatif puis soustrayons 1
            returns = period_data[model].dropna().prod() - 1
            
            results.append({
                'period': period_name,
                'model': model,
                'return': returns
            })
    
    # Conversion en DataFrame
    results_df = pd.DataFrame(results)
    
    # Pivot du DataFrame pour l'affichage du heatmap
    pivot_df = results_df.pivot(index='model', columns='period', values='return')
    
    # Création du heatmap
    plt.figure(figsize=(14, 8))
    
    # Utilisation d'une palette divergente pour mieux visualiser les retours positifs et négatifs
    cmap = sns.diverging_palette(10, 133, as_cmap=True)
    
    # Création du heatmap avec annotations
    heatmap = sns.heatmap(pivot_df, annot=True, fmt=".2%", cmap=cmap, center=0,
                         linewidths=.5, cbar_kws={"shrink": .75})
    
    # Configuration du titre et des étiquettes
    period_type = "Trimestre" if period == 'quarter' else "Année"
    plt.title(f'Retours par {period_type} pour chaque Modèle', fontsize=16)
    plt.xlabel(period_type, fontsize=12)
    plt.ylabel('Modèle', fontsize=12)
    
    # Rotation des étiquettes pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')
    
    # Ajustement de la mise en page
    plt.tight_layout()
    
    # Affichage du graphique
    plt.show()
    
    return results_df

rs = calculate_returns_and_plot(results_2d,period = 'year')
print(final_model.sort_values(ascending = False,by = 'temporality'))
# %%
