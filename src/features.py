import numpy as np
import pandas as pd
import config
from src.data_processing import download_prices, clean_prices

cols = ["drift", "v_stock", "v_bond", "s_b_corr", "ma_cross", "v_ratio", "rsi", "b_upper", "dom", "is_tom", "v_trend"]

def add_features(df_prices, s_qty, b_qty):
    df = df_prices.copy()
    
    r_s = df["stock"].pct_change()
    r_b = df["bond"].pct_change()
    
    win = config.volatility_past_period 
    ann = config.trading_days_per_year 

    df["v_stock"] = r_s.rolling(win).std() * np.sqrt(ann)
    df["v_bond"]  = r_b.rolling(win).std() * np.sqrt(ann)
    df["s_b_corr"] = r_s.rolling(config.corr_past_period).corr(r_b)
    df['v_ratio'] = df['stock'].pct_change().rolling(20).std() / df['bond'].pct_change().rolling(20).std()
    df["v_trend"] = df["v_stock"] / (df["v_stock"].shift(21) + 1e-9)

    diff = df["stock"].diff()
    g = (diff.where(diff > 0, 0)).rolling(14).mean()
    l = (-diff.where(diff < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + (g / (l + 1e-9))))
    
    mu = df["stock"].rolling(20).mean()
    sigma = df["stock"].rolling(20).std()
    df["b_upper"] = (df["stock"] - mu) / (2 * sigma + 1e-9)

    s_val = s_qty * df["stock"]
    b_val = b_qty * df["bond"]
    w_s = s_val / (s_val + b_val + 1e-9)
    df["drift"] = abs(w_s - config.stock_weight)

    df["ma_cross"] = (df["stock"].rolling(20).mean() > df["stock"].rolling(60).mean()).astype(int)

    df["dom"] = df.index.day
    df["is_tom"] = ((df["dom"] <= 2) | (df["dom"] >= 28)).astype(int)

    return df.dropna().copy()

def get_feature_columns():
    return cols

def save_features(df, path=None):
    p = path or config.get_processed_file_path()
    df.to_csv(p)

def load_features(path=None):
    p = path or config.get_processed_file_path()
    return pd.read_csv(p, index_col=0, parse_dates=True)

if __name__ == "__main__":
    config.make_dirs()
    prices = clean_prices(download_prices(refresh=False))
    initial_stock_value = config.initial_wealth * config.stock_weight
    initial_bond_value  = config.initial_wealth * config.bond_weight
    stock_shares = initial_stock_value / prices["stock"].iloc[0]
    bond_shares  = initial_bond_value  / prices["bond"].iloc[0]
    features = add_features(prices, stock_shares, bond_shares)
    save_features(features)



