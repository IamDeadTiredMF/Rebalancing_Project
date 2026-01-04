import pandas as pd
import numpy as np
import config
from src.portfolio import Portfolio
from src.data_processing import download_prices, clean_prices


def get_sortino(r):
    if len(r) < 2:
        return -1.0
    neg = r[r < 0]
    sd = np.std(neg) if len(neg) > 0 else 1e-6
    return float(np.mean(r) / sd)


def get_val(sq, bq, sp, bp, s, e):
    return sq * sp[s:e + 1] + bq * bp[s:e + 1]


def create_labels(prices, tc=0.001, h=126, lag=1, min_d=0.02, margin=0.02):
    sp, bp, t = prices["stock"].values, prices["bond"].values, prices.index

    p = Portfolio(config.initial_wealth, config.stock_weight, config.bond_weight)
    p.initialize_holdings(sp[0], bp[0])

    res = {}

    for i in range(len(prices) - lag - h - 1):
        dt = t[i]
        drift = p.get_drift(float(sp[i]), float(bp[i]))

        if drift < min_d:
            res[dt] = 0
            continue

        ie, ifn, ist = i + lag, i + lag + h, i + lag - 1

        vh = get_val(p.stock_shares, p.bond_shares, sp, bp, ist, ifn)
        rh = vh[1:] / vh[:-1] - 1.0
        sh = get_sortino(rh)

        cur_stock_val, cur_bond_val = p.stock_shares * sp[ie], p.bond_shares * bp[ie]
        total_value = cur_stock_val + cur_bond_val

        t_stock, t_bond = total_value * config.stock_weight, total_value * config.bond_weight
        turnover = abs(t_stock - cur_stock_val) + abs(t_bond - cur_bond_val)
        cost = turnover * float(tc)
        net_wealth = total_value - cost

        ns, nb = (net_wealth * config.stock_weight) / sp[ie], (net_wealth * config.bond_weight) / bp[ie]

        v_pre = get_val(p.stock_shares, p.bond_shares, sp, bp, ist, ie - 1)
        v_aft = get_val(ns, nb, sp, bp, ie, ifn)
        vr = np.append(v_pre, v_aft)

        sr = get_sortino(vr[1:] / vr[:-1] - 1.0)
        res[dt] = 1 if sr > (sh + margin) else 0

    return pd.Series(res, name="label")


def save_labels(s):
    s.to_csv(config.get_labels_file_path(), header=True)


def load_labels():
    df = pd.read_csv(config.get_labels_file_path(), index_col=0, parse_dates=True)
    return df.iloc[:, 0].astype(int)


if __name__ == "__main__":
    config.make_dirs()
    px = clean_prices(download_prices(refresh=False))
    lbls=create_labels(px,tc=config.transaction_cost[1],h=config.horizon,lag=config.trade_lag_days,min_d=min(config.thresholds_options))
    print(f"cnt: {len(lbls)}")
    print(f"pos: {int(lbls.sum())} | rate: {lbls.mean():.4f}")
    print(lbls.head(10))
    save_labels(lbls)
