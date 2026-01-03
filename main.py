import config
from src.data_processing import download_prices, clean_prices
from src.features import add_features, save_features
from src.labels import create_labels, save_labels
from src.models import train
from src.backtesting import run_backtest
from src.visualization import plot_comparison
from src.analysis_table import generate_report_table


def main():
    px = clean_prices(download_prices(refresh=False))

    s0, b0 = px["stock"].iloc[0], px["bond"].iloc[0]
    sq, bq = (config.initial_wealth * config.stock_weight) / s0, (config.initial_wealth * config.bond_weight) / b0

    feats = add_features(px, s_qty=sq, b_qty=bq)
    save_features(feats)

    lbls = create_labels(px, tc=config.transaction_cost[1], h=config.horizon, lag=config.trade_lag_days, min_d=min(config.thresholds_options))
    save_labels(lbls)

    train()
    run_backtest()
    plot_comparison()
    generate_report_table()


if __name__ == "__main__":
    config.make_dirs()
    main()