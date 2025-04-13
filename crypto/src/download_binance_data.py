from binance.client import Client
import pandas as pd
import numpy as np
from datetime import datetime

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

        # Trier les données par timestamp avant de procéder
        sub_df = sub_df.sort_index()

        start, end = sub_df.index.min(), sub_df.index.max()
        full_range = pd.date_range(start=start, end=end, freq=freq)
        
        # Reindexer pour avoir toutes les heures nécessaires
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
    return df

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
    training_merge['DR_New_Fees'] = 1 - fee * np.where((training_merge['buy_symbol_VS_refcur'].shift(1) != training_merge['buy_symbol_VS_refcur']) & (training_merge['DR_strategy_2d'] != 1) & (training_merge['DR_strategy_2d'].shift(1) != 1), 1, 0)
    training_merge['DR_Wallet_Final'] *= training_merge['DR_new_fees']
    
    # Joindre avec les DR bruts pour symboles et référence
    #Test_ = Test_.merge(Test[(Test['ref_cur'] == "USD") & (Test['symbol'] != ref)], on=["time_close", "symbol"], how="left")
    #Test_ = Test_.merge(Test[(Test['ref_cur'] == "USD") & (Test['symbol'] == ref)], on="time_close", how="left")
    
    return training_merge

# %% DL and retreat data
data_df = download_binance_data_concat(
    tickers=["ETHUSDC", "BTCUSDC", "ETHBTC"],
    start_str="1 Jan, 2017"
)
data_df = validate_and_fix_timestamps(data_df)
# %%
training = func_training_1d(prices=data_df,
                            fee = 0.075/100,
                            List_N=[5+5*i for i in range(10)],
                            func_MovingAverage = None)
training = extract_symbol_and_ref(training,ref_cur=['USDC', 'BTC'])

# %%
