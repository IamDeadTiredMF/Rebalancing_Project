import numpy as np
import pandas as pd
import config
from src.data_processing import download_prices, clean_prices

cols=["drift","v_stock","v_bond","s_b_corr","v_ratio","v_trend","vol_regime","stock_bond_divergence"]
def add_features(df_prices, s_qty, b_qty):
    df=df_prices.copy()
    r_s=df["stock"].pct_change(); r_b=df["bond"].pct_change(); win,ann=config.volatility_past_period,config.trading_days_per_year
    df["v_stock"]=r_s.rolling(win).std()*np.sqrt(ann); df["v_bond"]=r_b.rolling(win).std()*np.sqrt(ann)
    df["s_b_corr"]=r_s.rolling(config.corr_past_period).corr(r_b)
    df["v_ratio"]=df["stock"].pct_change().rolling(20).std()/(df["bond"].pct_change().rolling(20).std()+1e-9)
    df["v_trend"]=df["v_stock"]/(df["v_stock"].shift(21)+1e-9)
    s_val=s_qty*df["stock"]; b_val=b_qty*df["bond"]; w_s=s_val/(s_val+b_val+1e-9); df["drift"]=abs(w_s-config.stock_weight)
    df["vol_regime"]=(df["v_stock"]>df["v_stock"].rolling(126).quantile(0.75)).astype(int)
    df["stock_bond_divergence"]=(df["stock"].pct_change(20)-df["bond"].pct_change(20)).abs()
    return df[cols].dropna().copy()

def get_feature_columns():
    return cols

def save_features(df, path=None):
    p=path or config.get_processed_file_path(); 
    df.to_csv(p)

def load_features(path=None):
    p=path or config.get_processed_file_path(); 
    return pd.read_csv(p, index_col=0, parse_dates=True)

if __name__=="__main__":
    config.make_dirs()
    px=clean_prices(download_prices(refresh=False))
    s0,b0=px["stock"].iloc[0],px["bond"].iloc[0]
    sq,bq=(config.initial_wealth*config.stock_weight)/s0,(config.initial_wealth*config.bond_weight)/b0
    feats=add_features(px, s_qty=sq, b_qty=bq)
    save_features(feats)
    print(feats.head())






