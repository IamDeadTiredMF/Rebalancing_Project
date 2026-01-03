import pandas as pd
import numpy as np
import config
from src.portfolio import Portfolio
from src.data_processing import download_prices, clean_prices

def get_sortino(r):
    if len(r) < 2: return -1.0
    neg = r[r < 0]
    sd = np.std(neg) if len(neg) > 0 else 1e-6
    return np.mean(r) / sd

def get_val(sq, bq, sp, bp, s, e):
    return sq * sp[s:e+1] + bq * bp[s:e+1]

def create_labels(prices, tc=0.001, h=126, lag=1, min_d=0.02):
    sp = prices["stock"].values
    bp = prices["bond"].values
    t = prices.index
    
    p = Portfolio(config.initial_wealth, config.stock_weight, config.bond_weight)
    p.initialize_holdings(sp[0], bp[0])
    
    res = {}
    m = 0.02 
    
    for i in range(len(prices) - lag - h - 1):
        dt = t[i]
        drift = p.get_drift(float(sp[i]), float(bp[i]))

        if drift < min_d:
            res[dt] = 0
            continue
            
        ie, ifn = i + lag, i + lag + h
        ist = ie - 1
        
        vh = get_val(p.stock_shares, p.bond_shares, sp, bp, ist, ifn)
        rh = vh[1:] / vh[:-1] - 1.0
        sh = get_sortino(rh)

        cur_v = p.get_value(sp[ie], bp[ie])
        ns = (cur_v * config.stock_weight * (1 - tc)) / sp[ie]
        nb = (cur_v * config.bond_weight * (1 - tc)) / bp[ie]
        
        v_pre = get_val(p.stock_shares, p.bond_shares, sp, bp, ist, ie-1)
        v_aft = get_val(ns, nb, sp, bp, ie, ifn)
        vr = np.append(v_pre, v_aft)
        
        sr = get_sortino(vr[1:] / vr[:-1] - 1.0)
        res[dt] = 1 if sr > (sh + m) else 0

    return pd.Series(res, name="label")

def save_labels(s):
    s.to_csv(config.get_labels_file_path(), header=True)

def load_labels():
    df = pd.read_csv(config.get_labels_file_path(), index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(int)

if __name__ == "__main__":
    config.make_dirs()
    px = clean_prices(download_prices(refresh=False))

    lbls = create_labels(px, tc=0.001, h=config.horizon, lag=config.trade_lag_days, min_d=0.03)

    print(f"cnt: {len(lbls)}")
    print(f"pos: {int(lbls.sum())} | rate: {lbls.mean():.4f}")
    print(lbls.head(10))
    save_labels(lbls)