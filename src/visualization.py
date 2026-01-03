import pandas as pd
import matplotlib.pyplot as plt
import joblib
import config
from src.portfolio import simulate_buy_hold, simulate_calendar_rebalancing, simulate_threshold_rebalancing, simulate_ml_rebalancing
from src.data_processing import download_prices, clean_prices
from src.features import add_features


def plot_comparison():
    px = clean_prices(download_prices(refresh=False))
    cutoff = pd.to_datetime(config.train_date)
    px_test = px[px.index > cutoff].copy()

    s0, b0 = px["stock"].iloc[0], px["bond"].iloc[0]
    sq, bq = (config.initial_wealth * config.stock_weight) / s0, (config.initial_wealth * config.bond_weight) / b0

    has_ml, ml_map_test = False, None
    try:
        m_data = joblib.load(config.get_model_file_path())
        model, f_cols = m_data["model"], m_data["cols"]

        f_df = add_features(px, s_qty=sq, b_qty=bq)
        p = pd.Series(model.predict_proba(f_df[f_cols])[:, 1], index=f_df.index)
        p = p.reindex(px.index).fillna(0.0)

        thr = p.rolling(config.ml_thr_window).quantile(config.prob_quantile)
        score = (p - thr).fillna(-1.0)

        ml_map_test = {k: float(score.loc[k]) for k in px_test.index if k in score.index}
        has_ml = True
    except Exception as e:
        print(f"ML skipped in plots: {e}")

    paths = {}
    paths["Buy & Hold"] = simulate_buy_hold(px_test, initial_wealth=config.initial_wealth)
    paths["Quarterly Rebalance"] = simulate_calendar_rebalancing(px_test, "quarterly", initial_wealth=config.initial_wealth)
    paths["Threshold (5% Drift)"] = simulate_threshold_rebalancing(px_test, threshold=0.05, initial_wealth=config.initial_wealth)

    if has_ml and ml_map_test is not None:
        paths[f"ML Rolling Q{int(config.prob_quantile*100)}"] = simulate_ml_rebalancing(px_test, ml_map_test, prob_threshold=0.0, initial_wealth=config.initial_wealth)

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
        v = df["value"]
        dd = (v - v.cummax()) / v.cummax() * 100
        plt.plot(df.index, dd, label=name, linewidth=1.5)
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


