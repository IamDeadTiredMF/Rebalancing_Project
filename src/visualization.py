import pandas as pd
import matplotlib.pyplot as plt
import joblib
import config
from src.portfolio import *
from src.data_processing import download_prices, clean_prices

def plot_comparison():
    # Load data and initial quantities
    px = clean_prices(download_prices(refresh=False))
    s0, b0 = px["stock"].iloc[0], px["bond"].iloc[0]
    init_s = (config.initial_wealth * config.stock_weight) / s0
    init_b = (config.initial_wealth * config.bond_weight) / b0
    
    # Load model for tactical predictions
    try:
        m_data = joblib.load(config.get_model_file_path())
        from src.features import add_features
        feats = add_features(px, s_qty=init_s, b_qty=init_b)
        probs = m_data["model"].predict_proba(feats[m_data["cols"]])[:, 1]
        ml_map = dict(zip(feats.index, probs))
        has_ml = True
    except Exception:
        ml_map = None
        has_ml = False
        print("Model not found - skipping ML strategy in plots.")

    # Run simulations
    paths = {}
    paths["Buy & Hold"] = simulate_buy_hold(px)
    paths["Quarterly Rebalance"] = simulate_calendar_rebalancing(px, "quarterly")
    paths["Threshold (5% Drift)"] = simulate_threshold_rebalancing(px, threshold=0.05)
    
    if has_ml:
        # Using the threshold that gave us the $914k result
        paths["ML Tactical (Dynamic)"] = simulate_ml_rebalancing(px, ml_map, 
                                                               prob_threshold=config.prob_threshold)

    # 1. Growth of Wealth
    plt.figure(figsize=(12, 6))
    for name, df in paths.items():
        plt.plot(df.index, df["value"], label=name, linewidth=2)
    
    plt.title(f"Cumulative Wealth: {config.stock_ticker_etf} & {config.bond_ticker_etf}", 
              fontsize=14, fontweight="bold")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.figures_direction / "value_curves.png")
    plt.close()

    # 2. Drawdown Analysis (Risk)
    plt.figure(figsize=(12, 6))
    for name, df in paths.items():
        v = df["value"]
        dd = (v - v.cummax()) / v.cummax() * 100
        plt.plot(df.index, dd, label=name, linewidth=1.5)
    
    plt.title("Drawdown Profile (Peak-to-Trough)", fontsize=14, fontweight="bold")
    plt.ylabel("Drawdown %")
    plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(config.figures_direction / "drawdowns.png")
    plt.close()

    # 3. Allocation Drift
    plt.figure(figsize=(12, 6))
    target_pct = config.stock_weight * 100
    for name, df in paths.items():
        plt.plot(df.index, df["stock_weight"] * 100, label=name, alpha=0.8)
    
    plt.axhline(target_pct, color="red", linestyle="--", label=f"Target ({target_pct}%)")
    plt.title("Actual Equity Exposure Over Time", fontsize=14, fontweight="bold")
    plt.ylabel("Stock Weight %")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(config.figures_direction / "stock_weight.png")
    plt.close()

    print(f"Visualizations generated in {config.figures_direction}")

if __name__ == "__main__":
    config.make_dirs()
    plot_comparison()