import pandas as pd
import matplotlib.pyplot as plt
import joblib
import config
from src.data_processing import download_prices, clean_prices
from src.features import add_features
from src.portfolio import *


def plot_comparison():
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

    paths = {"Buy & Hold": simulate_buy_hold(px_test, initial_wealth=config.initial_wealth), "Quarterly Rebalance": simulate_calendar_rebalancing(px_test, "quarterly", initial_wealth=config.initial_wealth), "Threshold (5% Drift)": simulate_threshold_rebalancing(px_test, threshold=0.05, initial_wealth=config.initial_wealth)}

    if model is not None and f_cols is not None and f_df is not None:
        paths[f"ML Model Rolling Q{int(config.prob_quantile*100)}"] = simulate_ml_rebalancing_model(px_test, model, f_df, f_cols, invert_proba=False, prob_threshold=0.0, initial_wealth=config.initial_wealth, thr_window=config.ml_thr_window, prob_quantile=config.prob_quantile)

    plt.figure(figsize=(12, 6))
    for name, df in paths.items():
        plt.plot(df.index, df["value"], label=name, linewidth=2)
    plt.title(f"Test Period Wealth: {config.stock_ticker_etf} & {config.bond_ticker_etf}", fontsize=14, fontweight="bold")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.figures_direction / "value_curves_test.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    for name, df in paths.items():
        v = df["value"]; dd = (v - v.cummax()) / v.cummax() * 100; plt.plot(df.index, dd, label=name, linewidth=1.5)
    plt.title("Test Period Drawdowns", fontsize=14, fontweight="bold")
    plt.ylabel("Drawdown %")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(config.figures_direction / "drawdowns_test.png")
    plt.close()

    plt.figure(figsize=(12, 6))
    target_pct = config.stock_weight * 100
    for name, df in paths.items():
        plt.plot(df.index, df["stock_weight"] * 100, label=name, alpha=0.8)
    plt.axhline(target_pct, color="red", linestyle="--", label=f"Target ({target_pct}%)")
    plt.title("Test Period Equity Weight", fontsize=14, fontweight="bold")
    plt.ylabel("Stock Weight %")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(config.figures_direction / "stock_weight_test.png")
    plt.close()

    print(f"Visualizations generated in {config.figures_direction}")


if __name__ == "__main__":
    config.make_dirs()
    plot_comparison()




