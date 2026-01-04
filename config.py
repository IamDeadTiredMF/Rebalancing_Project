from pathlib import Path

#data for the etf used in rebalancing and the time periods
start_date="2010-01-01"
end_date="2025-01-01" #last day is excluded in yfinance download, so set to first day of next year to include full 2024 data (31.12.2024)
train_date="2017-12-31" #date to split train and test data instead of using % split. Test data will be from 2018-01-01 to end_date

#etf tickers
stock_ticker_etf="IWM" #iShares Russell 2000 ETF
bond_ticker_etf="TLT" #iShares 20+ Year Treasury Bond ETF

#investment parameters
initial_wealth=100000 #initial investment amount in USD
stock_weight=0.6
bond_weight=0.4
rebalance_frequencies = ['monthly', 'quarterly', 'yearly']  #options for rebalancing frequency

#rebalance_costs
thresholds_options=[0.03,0.05,0.07] #3%,5%,7% thresholds
transaction_cost=[0,0.0005,0.001,0.0015] #0, 5, 10, 15 bps

#data params
trading_days_per_year = 252
volatility_past_period=20
corr_past_period=60

#label + decision params
horizon=126 #days ahead to check if rebalancing was beneficial
trade_lag_days=1
decision_frequency=1

#ml params
random_seed=42
prob_quantile=0.90
ml_thr_window=252
cv_folds=5

#random forest params
rf_n_estimators=100
rf_max_depth=10
rf_sample_split=50
rf_samples_leaf=10
rf_class_weight="balanced_subsample"  #"balanced" or "balanced_subsample"

#paths
root = Path(__file__).resolve().parent
data_direction = root / "data"
output_direction = root / "outputs"
raw_data = data_direction / "raw"
processed_info = data_direction / "processed"
results = output_direction / "results"
figures_direction = output_direction / "figures"
table_direction = output_direction / "tables"
models_direction = output_direction / "models"
def make_dirs():
    raw_data.mkdir(parents=True, exist_ok=True)
    processed_info.mkdir(parents=True, exist_ok=True)
    figures_direction.mkdir(parents=True, exist_ok=True)
    table_direction.mkdir(parents=True, exist_ok=True)
    models_direction.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
def get_price_file_path():
    return raw_data / f"prices_{stock_ticker_etf}_{bond_ticker_etf}.csv"
def get_processed_file_path():
    return processed_info / f"processed_{stock_ticker_etf}_{bond_ticker_etf}.csv"
def get_labels_file_path():
    return processed_info / f"labels_{stock_ticker_etf}_{bond_ticker_etf}.csv"
def get_model_file_path():
    return models_direction / f"model__{stock_ticker_etf}_{bond_ticker_etf}.joblib"
def get_backtest_file_path():
    return results / f"backtest_{stock_ticker_etf}_{bond_ticker_etf}.csv"
