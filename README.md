# Rebalancing_Project
**Marius Fortuna**

This project compares classic portfolio rebalancing rules against a machine-learning-based trigger for a two-asset portfolio:

- Stock ETF: **QQQ** (NASDAQ-100)
- Bond ETF: **TLT** (20+ Year U.S. Treasury Bonds)

I start from a standard **60/40** portfolio (60% stocks, 40% bonds). The goal is to decide when rebalancing back to target is actually worth it, while accounting for **transaction costs** and using a **trade execution lag** to avoid look-ahead bias.

---

## What the project does

1. Download data from Yahoo Finance to obtain historical prices  
2. Add features such as drift, volatility, correlation, momentum, etc.  
3. Generate labels (1/0) by simulating two futures over the horizon in `config.py`:
   - Hold current weights (no rebalance)
   - Rebalance (with execution lag to avoid look-ahead bias) and transaction costs (taxes ignored)
4. Train a classifier using **Random Forest** to help determine the outcome in the test series  
5. Backtest all strategies using an out-of-sample test period  
6. Output results as `.csv` and tables as `.md`, plus plots as `.png` in their respective folders  

---

## Project structure

```text
balancing_project/
  config.py
  main.py
  src/
    data_processing.py
    features.py
    labels.py
    models.py
    portfolio.py
    backtesting.py
    visualization.py
    analysis_table.py
  data/
    processed/
    raw/
  outputs/
    figures/
    models/
    results/
    tables/
```
# Installation:
Python 3.11.14(everything else is placed in environment.yml)
run in terminal
```bash
conda env create -f environment.yml
conda activate balancing_project
# how-run the project
python -m main

# this will download, clean the prices, compute the features and lavels, train the model, run backtests on the test period, generate plots and tables.

#Optional: there is also separate tests, present in most files. If you want to run use:

python -m src.data_processing 
python -m src.features
python -m src.models
python -m src.visualization
```
# Outputs:
1. outputs/results/backtest_QQQ_TLT.csv
2. outputs/figures/value_curves_test.png
3. outputs/figures/drawdowns_test.png
4. outputs/tables/summary_table.md
# Configuration
All parameters are set in config.py and can be changed to observe other outcomes:
1. etf tickers, data rangers and the train and test split
2. weights
3. transaction costs as a list
4. rebalancing data for calendar and drift based
5. ML parameters: prob_quantile, ml_thr_window, trade_lag_days
# Strategies
1. buy & hold
2. calendar rebalancing
3. x% drift from set weight rebalancing
4. ML trigger
   - models produces daily probabilities
   - trigger uses rolling quantile to decide when the probability is unusually high relative to recent history.
# Limitations
1. this has only 2 major asset setting.
2. Labels are inherently noisy because future risk-adjusted performance is regime-dependent
3. there is no mention of taxes
4. The ML model may not outperform simple baselines out-of-sample; the main focus is a correct, reproducible pipeline with realistic constraints (transaction costs + execution lag
# Reproductibility:
First run downloads and caches prices in data/raw/…csv; subsequent runs reuse cached data unless refresh=True.
Everything is deterministic given the config settings, but to force refreshes on the dataset, one must set:
```bash
# from
px = clean_prices(download_prices(refresh=False))
# to 
px = clean_prices(download_prices(refresh=True))
```