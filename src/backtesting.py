import pandas as pd
import joblib
import config
from src.data_processing import download_prices, clean_prices
from src.features import add_features
from src.portfolio import *


def run_backtest(show_table=True):
    px = clean_prices(download_prices(refresh=False))
    cutoff = pd.to_datetime(config.train_date)
    px_test = px[px.index > cutoff].copy()

    s0, b0 = px["stock"].iloc[0], px["bond"].iloc[0]
    sq = (config.initial_wealth * config.stock_weight) / s0
    bq = (config.initial_wealth * config.bond_weight) / b0

    model, f_cols, f_df = None, None, None
    try:
        m_data = joblib.load(config.get_model_file_path()); model, f_cols = m_data["model"], m_data["cols"]; f_df = add_features(px, s_qty=sq, b_qty=bq)
    except Exception:
        model, f_cols, f_df = None, None, None

    paths = {"buy_hold": simulate_buy_hold(px_test, initial_wealth=config.initial_wealth), **{f"calendar_{f}": simulate_calendar_rebalancing(px_test, frequency=f, initial_wealth=config.initial_wealth) for f in config.rebalance_frequencies}, **{f"threshold_{int(t*100)}pct": simulate_threshold_rebalancing(px_test, threshold=t, initial_wealth=config.initial_wealth) for t in config.thresholds_options}}

    if model is not None and f_cols is not None and f_df is not None:
        tag = f"ml_model_rollq_{int(config.prob_quantile*100)}"
        paths[tag] = simulate_ml_rebalancing_model(px_test, model, f_df, f_cols, invert_proba=False, prob_threshold=0.0, initial_wealth=config.initial_wealth, thr_window=config.ml_thr_window, prob_quantile=config.prob_quantile)

    res = pd.DataFrame({k: calculate_performance_metrics(v, initial_wealth=config.initial_wealth) for k, v in paths.items()}).T
    res = res.sort_values("sortino_ratio", ascending=False)

    out_path = config.get_backtest_file_path()
    res.to_csv(out_path)

    if show_table:
        display_cols = ["sortino_ratio", "annualized_return", "annualized_downside_volatility", "annualized_volatility", "n_rebalances", "total_costs", "final_value"]
        print("BACKTEST RESULTS (TEST ONLY, sorted by sortino ratio)")
        print(res[display_cols].to_string())
        print(f"results saved to: {out_path}")

    return res


if __name__ == "__main__":
    config.make_dirs()
    run_backtest()








