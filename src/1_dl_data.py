# %%

from importlib import reload
import download_data
download_data = reload(download_data)
from download_data import *
# %%
API_KEY = get_api_key()
service = EODHDDataService(API_KEY)
fundamental_data = FundamentalData(API_KEY)
price = PriceData()

historical_company_sp500 = service.get_sp500_historical_composition()
ticker_from_exchange = service.get_ticker_list_from_exchange(exchange_code= 'US')[['Code', 'Type']]

historical_company = (historical_company_sp500
                      .merge(ticker_from_exchange,
                             left_on=["Ticker"],
                             right_on=["Code"], how="left"))
historical_company = historical_company[(historical_company["Type"] == "Common Stock") | pd.isna(historical_company["Type"])]

list_ticker = historical_company.dropna(subset=['Ticker'])['Ticker'].unique()
list_ticker = [str(ticker).replace('.', '-') + ".US" for ticker in list_ticker]
funda = service.get_fundamental_data(list_ticker)
general = fundamental_data.process_fundamental_data(funda,list_ticker,"general")
income_statement = fundamental_data.process_fundamental_data(funda,list_ticker,"Income_Statement")
balance_sheet = fundamental_data.process_fundamental_data(funda,list_ticker,"Balance_Sheet")
cash_flow = fundamental_data.process_fundamental_data(funda,list_ticker,"Cash_Flow")
earnings = fundamental_data.process_fundamental_data(funda,list_ticker,"Earnings")
outstanding_shares = fundamental_data.process_fundamental_data(funda,list_ticker,"outstandingShares")
prices = price.get_price_data_for_tickers(tickers=list_ticker)
SP500Price = price.get_price_data_for_tickers(tickers="SPY")
Selected_Exchange = 'US'
os.chdir('../data')

prices.to_parquet(os.path.join(f'{Selected_Exchange}_Finalprice.parquet'))
general.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_General.parquet'))
income_statement.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Income_statement.parquet'))
balance_sheet.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Balance_sheet.parquet'))
cash_flow.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Cash_flow.parquet'))
earnings.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_Earnings.parquet'))
outstanding_shares.to_parquet(os.path.join(data_dir, f'{Selected_Exchange}_share.parquet'))
SP500Price.to_parquet(os.path.join(data_dir, "SP500Price.parquet"))


exchange = 'PA'

list_ticker = service.get_ticker_list_from_exchange(exchange_code=exchange)
list_ticker = list(list_ticker[list_ticker['Type'] == "Common Stock"]['Code']) 
list_ticker = [str(ticker).replace('.', '-') + ".PA" for ticker in list_ticker]

funda = service.get_fundamental_data(list_ticker)
general = fundamental_data.process_fundamental_data(funda,list_ticker,"general")
prices = price.get_price_data_for_tickers(tickers=list_ticker)
os.chdir('../data')
technical_price.to_parquet(os.path.join(f'{exchange}_Finalprice.parquet'))
general.to_parquet(os.path.join(data_dir, f'{exchange}_General.parquet'))
# %%
