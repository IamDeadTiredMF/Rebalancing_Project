import pandas as pd
import joblib
import config
from src.portfolio import simulate_buy_hold, simulate_calendar_rebalancing, simulate_threshold_rebalancing, simulate_ml_rebalancing, calculate_performance_metrics
from src.data_processing import download_prices, clean_prices
from src.features import add_features


def run_backtest():
    px = clean_prices(download_prices(refresh=False))
    cutoff = pd.to_datetime(config.train_date)
    px_test = px[px.index > cutoff].copy()

    s0, b0 = px["stock"].iloc[0], px["bond"].iloc[0]
    sq, bq = (config.initial_wealth * config.stock_weight) / s0, (config.initial_wealth * config.bond_weight) / b0

    ml_map_test = None
    try:
        m_data = joblib.load(config.get_model_file_path())
        model, f_cols = m_data["model"], m_data["cols"]

        f_df = add_features(px, s_qty=sq, b_qty=bq)
        p = pd.Series(model.predict_proba(f_df[f_cols])[:, 1], index=f_df.index)
        p = p.reindex(px.index).fillna(0.0)

        thr = p.rolling(config.ml_thr_window).quantile(config.prob_quantile)
        score = (p - thr).fillna(-1.0)

        ml_map_test = {k: float(score.loc[k]) for k in px_test.index if k in score.index}

        print("ml model loaded")
    except Exception:
        print("no ml model found, skipping ml strategy")

    paths = {}
    paths["buy_hold"] = simulate_buy_hold(px_test, initial_wealth=config.initial_wealth)

    for freq in config.rebalance_frequencies:
        tag = f"calendar_{freq}"
        print(f"running {tag}...")
        paths[tag] = simulate_calendar_rebalancing(px_test, frequency=freq, initial_wealth=config.initial_wealth)

    for thr0 in config.thresholds_options:
        tag = f"threshold_{int(thr0*100)}pct"
        print(f"running {tag}...")
        paths[tag] = simulate_threshold_rebalancing(px_test, threshold=thr0, initial_wealth=config.initial_wealth)

    if ml_map_test is not None:
        tag = f"ml_rollq_{int(config.prob_quantile*100)}"
        print(f"running {tag}...")
        paths[tag] = simulate_ml_rebalancing(px_test, ml_map_test, prob_threshold=0.0, initial_wealth=config.initial_wealth)

    stats = {k: calculate_performance_metrics(v, initial_wealth=config.initial_wealth) for k, v in paths.items()}
    res = pd.DataFrame(stats).T.sort_values("sharpe_ratio", ascending=False)

    print("BACKTEST RESULTS (TEST ONLY, sorted by sharpe ratio)")
    display_cols = ["sharpe_ratio", "annualized_return", "annualized_volatility", "n_rebalances", "total_costs", "final_value"]
    print(res[display_cols].to_string())

    out_path = config.get_backtest_file_path()
    res.to_csv(out_path)
    print(f"\nresults saved to: {out_path}")

    return res


if __name__ == "__main__":
    config.make_dirs()
    run_backtest()






