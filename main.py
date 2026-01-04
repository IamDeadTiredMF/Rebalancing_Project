import config
from src.data_processing import download_prices, clean_prices
from src.features import add_features, save_features
from src.labels import create_labels, save_labels
from src.models import train
from src.backtesting import run_backtest
from src.visualization import plot_comparison
from src.analysis_table import generate_report_table

def main():
    prices = clean_prices(download_prices(refresh=False))

    s0, b0 = prices["stock"].iloc[0], prices["bond"].iloc[0]
    s_qty = (config.initial_wealth * config.stock_weight) / s0
    b_qty = (config.initial_wealth * config.bond_weight) / b0

    features = add_features(prices, s_qty=s_qty, b_qty=b_qty)
    save_features(features)

    labels = create_labels(prices,tc=config.transaction_cost[1],h=config.horizon,lag=config.trade_lag_days,min_d=min(config.thresholds_options))
    save_labels(labels)

    train(True, False)

    run_backtest()
    plot_comparison()
    generate_report_table()

if __name__ == "__main__":
    config.make_dirs()
    main()

