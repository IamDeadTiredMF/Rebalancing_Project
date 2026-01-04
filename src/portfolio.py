import pandas as pd 
import numpy as np
import config


def _period_end_trading_days(index, freq):
    s = pd.Series(index=index, data=1)
    out = s.groupby(pd.Grouper(freq=freq)).apply(lambda x: x.index[-1])
    return out.sort_values().unique()


class Portfolio:
    def __init__(self, initial_wealth, stock_weight, bond_weight):
        self.initial_wealth = float(initial_wealth)
        self.target_stock_weight = float(stock_weight)
        self.target_bond_weight = float(bond_weight)
        self.stock_shares, self.bond_shares = 0.0, 0.0
        self.rebalance_dates, self.costs_paid = [], []
        self.total_turnover, self.total_costs = 0.0, 0.0

    def initialize_holdings(self, stock_price, bond_price):
        stock_value, bond_value = self.initial_wealth * self.target_stock_weight, self.initial_wealth * self.target_bond_weight
        self.stock_shares, self.bond_shares = stock_value / stock_price, bond_value / bond_price

    def get_value(self, stock_price, bond_price):
        return self.stock_shares * stock_price + self.bond_shares * bond_price

    def get_weights(self, stock_price, bond_price):
        total = self.get_value(stock_price, bond_price)
        if total <= 0:
            return 0.0, 0.0
        return (self.stock_shares * stock_price) / total, (self.bond_shares * bond_price) / total

    def get_drift(self, stock_price, bond_price):
        sw, _ = self.get_weights(stock_price, bond_price)
        return abs(sw - self.target_stock_weight)

    def rebalance(self, stock_price, bond_price, transaction_cost, date):
        tc = float(transaction_cost)

        total_value = self.get_value(stock_price, bond_price)
        current_stock_val, current_bond_val = self.stock_shares * stock_price, self.bond_shares * bond_price

        target_stock_gross, target_bond_gross = total_value * self.target_stock_weight, total_value * self.target_bond_weight
        turnover = abs(target_stock_gross - current_stock_val) + abs(target_bond_gross - current_bond_val)
        cost = turnover * tc

        net_wealth = total_value - cost
        self.stock_shares, self.bond_shares = (net_wealth * self.target_stock_weight) / stock_price, (net_wealth * self.target_bond_weight) / bond_price

        self.total_turnover, self.total_costs = self.total_turnover + turnover, self.total_costs + cost
        self.rebalance_dates.append(date)
        self.costs_paid.append(cost)
        return turnover, cost


def simulate_buy_hold(prices, initial_wealth=None):
    iw = initial_wealth or config.initial_wealth
    p = Portfolio(iw, config.stock_weight, config.bond_weight)
    p.initialize_holdings(prices["stock"].iloc[0], prices["bond"].iloc[0])

    rows = []
    for date, row in prices.iterrows():
        v = p.get_value(row["stock"], row["bond"])
        sw, bw = p.get_weights(row["stock"], row["bond"])
        rows.append({"date": date, "value": v, "stock_weight": sw, "bond_weight": bw, "drift": p.get_drift(row["stock"], row["bond"]), "rebalanced": False, "turnover": 0.0, "cost": 0.0})
    return pd.DataFrame(rows).set_index("date")


def simulate_calendar_rebalancing(prices, frequency="monthly", transaction_cost=None, initial_wealth=None):
    iw = initial_wealth or config.initial_wealth
    tc = transaction_cost if transaction_cost is not None else config.transaction_cost[1]

    p = Portfolio(iw, config.stock_weight, config.bond_weight)
    p.initialize_holdings(prices["stock"].iloc[0], prices["bond"].iloc[0])

    freq_map = {"monthly": "M", "quarterly": "Q", "yearly": "Y"}
    decision_days = set(_period_end_trading_days(prices.index, freq_map[frequency]))

    rows, lag, pending = [], int(config.trade_lag_days), None

    for i, (date, row) in enumerate(prices.iterrows()):
        exec_today, t_today, c_today = False, 0.0, 0.0

        if pending is not None:
            pending -= 1
            if pending == 0:
                t_today, c_today = p.rebalance(row["stock"], row["bond"], tc, date)
                exec_today, pending = True, None

        if pending is None and date in decision_days and i < len(prices) - lag:
            pending = lag

        sw, bw = p.get_weights(row["stock"], row["bond"])
        rows.append({"date": date, "value": p.get_value(row["stock"], row["bond"]), "stock_weight": sw, "bond_weight": bw, "drift": p.get_drift(row["stock"], row["bond"]), "rebalanced": exec_today, "turnover": t_today, "cost": c_today})

    return pd.DataFrame(rows).set_index("date")


def simulate_threshold_rebalancing(prices, threshold=None, transaction_cost=None, initial_wealth=None):
    iw = initial_wealth or config.initial_wealth
    tc = transaction_cost if transaction_cost is not None else config.transaction_cost[1]
    thr = threshold if threshold is not None else config.thresholds_options[1]

    p = Portfolio(iw, config.stock_weight, config.bond_weight)
    p.initialize_holdings(prices["stock"].iloc[0], prices["bond"].iloc[0])

    rows, lag, pending = [], int(config.trade_lag_days), None
    step = int(config.decision_frequency)

    for i, (date, row) in enumerate(prices.iterrows()):
        exec_today, t_today, c_today = False, 0.0, 0.0

        if pending is not None:
            pending -= 1
            if pending == 0:
                t_today, c_today = p.rebalance(row["stock"], row["bond"], tc, date)
                exec_today, pending = True, None

        drift = p.get_drift(row["stock"], row["bond"])

        if pending is None and i % step == 0 and drift > thr and i < len(prices) - lag:
            pending = lag

        sw, bw = p.get_weights(row["stock"], row["bond"])
        rows.append({"date": date, "value": p.get_value(row["stock"], row["bond"]), "stock_weight": sw, "bond_weight": bw, "drift": drift, "rebalanced": exec_today, "turnover": t_today, "cost": c_today})

    return pd.DataFrame(rows).set_index("date")


def simulate_ml_rebalancing(prices, ml_predictions, prob_threshold=0.0, transaction_cost=None, initial_wealth=None):
    iw = initial_wealth or config.initial_wealth
    tc = transaction_cost if transaction_cost is not None else config.transaction_cost[1]
    pt = float(prob_threshold)
    min_d = min(config.thresholds_options)

    p = Portfolio(iw, config.stock_weight, config.bond_weight)
    p.initialize_holdings(prices["stock"].iloc[0], prices["bond"].iloc[0])

    rows, lag, pending = [], int(config.trade_lag_days), None
    step = int(config.decision_frequency)

    for i, (date, row) in enumerate(prices.iterrows()):
        exec_today, t_today, c_today = False, 0.0, 0.0

        if pending is not None:
            pending -= 1
            if pending == 0:
                t_today, c_today = p.rebalance(row["stock"], row["bond"], tc, date)
                exec_today, pending = True, None

        prob = float(ml_predictions.get(date, 0.0))
        drift = p.get_drift(row["stock"], row["bond"])

        if pending is None and i % step == 0 and drift > min_d and prob > pt and i < len(prices) - lag:
            pending = lag

        sw, bw = p.get_weights(row["stock"], row["bond"])
        rows.append({"date": date, "value": p.get_value(row["stock"], row["bond"]), "stock_weight": sw, "bond_weight": bw, "drift": drift, "rebalanced": exec_today, "turnover": t_today, "cost": c_today})

    return pd.DataFrame(rows).set_index("date")


def simulate_ml_rebalancing_model(prices, model, features_df, feature_cols, invert_proba=False, prob_threshold=0.0, transaction_cost=None, initial_wealth=None, thr_window=None, prob_quantile=None):
    iw = initial_wealth or config.initial_wealth
    tc = transaction_cost if transaction_cost is not None else config.transaction_cost[1]
    pt = float(prob_threshold)
    min_d = min(config.thresholds_options)
    step = int(config.decision_frequency)

    win = int(thr_window or config.ml_thr_window)
    q = float(prob_quantile or config.prob_quantile)

    classes = list(getattr(model, "classes_", [0, 1]))
    pos_idx = classes.index(1) if 1 in classes else 1

    p = Portfolio(iw, config.stock_weight, config.bond_weight)
    p.initialize_holdings(prices["stock"].iloc[0], prices["bond"].iloc[0])

    rows, lag, pending = [], int(config.trade_lag_days), None
    prob_hist = []

    for i, (date, row) in enumerate(prices.iterrows()):
        exec_today, t_today, c_today = False, 0.0, 0.0

        if pending is not None:
            pending -= 1
            if pending == 0:
                t_today, c_today = p.rebalance(row["stock"], row["bond"], tc, date)
                exec_today, pending = True, None

        drift = p.get_drift(row["stock"], row["bond"])

        prob = 0.0
        score = -1e9

        if i % step == 0 and date in features_df.index:
            x = features_df.loc[date, feature_cols].copy()
            if "drift" in x.index:
                x["drift"] = drift

            x = x.reindex(feature_cols)
            x1 = pd.DataFrame([x.values], columns=feature_cols)
            prob = float(model.predict_proba(x1)[0][pos_idx])

            if invert_proba:
                prob = 1.0 - prob

            prob_hist.append(prob)
            if len(prob_hist) >= win:
                thr = float(np.quantile(prob_hist[-win:], q))
                score = prob - thr

        if pending is None and i % step == 0 and drift > min_d and score > pt and i < len(prices) - lag:
            pending = lag

        sw, bw = p.get_weights(row["stock"], row["bond"])
        rows.append({"date": date, "value": p.get_value(row["stock"], row["bond"]), "stock_weight": sw, "bond_weight": bw, "drift": drift, "rebalanced": exec_today, "turnover": t_today, "cost": c_today, "ml_prob": prob, "ml_score": score})

    return pd.DataFrame(rows).set_index("date")


def calculate_performance_metrics(results,initial_wealth=None):
    iw=initial_wealth or config.initial_wealth
    v=results["value"]

    total_ret=(v.iloc[-1]-iw)/iw
    n_years=len(v)/config.trading_days_per_year
    ann_ret=(1+total_ret)**(1/n_years)-1 if n_years>0 else 0.0

    daily_ret=v.pct_change().dropna()
    ann_vol=daily_ret.std(ddof=0)*(config.trading_days_per_year**0.5) if len(daily_ret)>1 else 0.0

    neg=daily_ret[daily_ret<0]
    down_std=neg.std(ddof=0) if len(neg)>1 else 1e-9
    ann_down=down_std*(config.trading_days_per_year**0.5)
    sortino=(daily_ret.mean()/down_std)*(config.trading_days_per_year**0.5) if len(daily_ret)>1 else 0.0

    dd=(v-v.expanding().max())/v.expanding().max()

    return {"total_return":total_ret,"annualized_return":ann_ret,"annualized_volatility":ann_vol,"annualized_downside_volatility":ann_down,"sortino_ratio":sortino,"max_drawdown":dd.min(),"n_rebalances":int(results["rebalanced"].sum()),"avg_drift":float(results["drift"].mean()),"max_drift":float(results["drift"].max()),"final_value":float(v.iloc[-1]),"total_costs":float(results["cost"].sum()),"total_turnover":float(results["turnover"].sum())}
