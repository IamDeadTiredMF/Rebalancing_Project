import pandas as pd
import joblib
import config
from src.portfolio import *
from src.data_processing import *
from src.features import *

def run_backtest():
    
    px = clean_prices(download_prices(refresh=False))
    s0, b0 = px["stock"].iloc[0], px["bond"].iloc[0]
    init_s_qty = (config.initial_wealth * config.stock_weight) / s0
    init_b_qty = (config.initial_wealth * config.bond_weight) / b0

    try:
        m_data = joblib.load(config.get_model_file_path())
        model, f_cols = m_data["model"], m_data["cols"]
        f_df = add_features(px, s_qty=init_s_qty, b_qty=init_b_qty)
        probs = model.predict_proba(f_df[f_cols])[:, 1]
        ml_map = dict(zip(f_df.index, probs))
        print("ml model loaded")
    except FileNotFoundError:
        ml_map = None
        print("no ml model found, skipping ml strategies")

    stats = {}
    
    stats["buy_hold"] = calculate_performance_metrics(simulate_buy_hold(px))
    
    for freq in config.rebalance_frequencies:
        tag = f"calendar_{freq}"
        print(f"running {tag}...")
        path = simulate_calendar_rebalancing(px, frequency=freq)
        stats[tag] = calculate_performance_metrics(path)
    
    for thr in config.thresholds_options:
        tag = f"threshold_{int(thr*100)}pct"
        print(f"running {tag}...")
        path = simulate_threshold_rebalancing(px, threshold=thr)
        stats[tag] = calculate_performance_metrics(path)
    
    if ml_map is not None:
        test_thresholds = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
        for pt in test_thresholds:
            tag = f"ml_prob_{int(pt*100)}"
            print(f"running {tag}...")
            path = simulate_ml_rebalancing(px, ml_map, prob_threshold=pt)
            stats[tag] = calculate_performance_metrics(path)

    res = pd.DataFrame(stats).T.sort_values("sharpe_ratio", ascending=False)
    
    print("BACKTEST RESULTS (sorted by sharpe ratio)")
    
    display_cols = ["sharpe_ratio", "annualized_return", "annualized_volatility", "n_rebalances", "total_costs", "final_value"]
    print(res[display_cols].to_string())
    
    res.to_csv(config.get_backtest_file_path())
    print(f"\nfull results saved to: {config.get_backtest_file_path()}")
    
    return res

if __name__ == "__main__":
    config.make_dirs()
    run_backtest()
