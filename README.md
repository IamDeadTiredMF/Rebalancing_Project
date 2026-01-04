# Rebalancing_Project
**Author:Marius Fortuna**

This project compares classic portfolio rebalancing rules against a machine-learning-based trigger for a two-asset portfolio:

- Stock ETF: **IWM** (iShares Russell 2000 ETF)
- Bond ETF: **TLT** (20+ Year U.S. Treasury Bonds)

I start from a standard **60/40** portfolio (60% stocks, 40% bonds). The goal is to decide when rebalancing back to target is actually worth it, while accounting for **transaction costs** and using a **trade execution lag** to avoid look-ahead bias.

---

## What the project does
1. Download historical prices from Yahoo Finance (cached locally for reproducibility)
2. Compute features (drift, volatility, correlation, momentum proxies, calendar features)
3. Generate labels (0/1) by simulating two futures over a fixed horizon:
   - **Hold** current weights (no rebalance)
   - **Rebalance** back to 60/40 (executed with a lag and including transaction costs; taxes ignored)
4. Train a **Random Forest** classifier to predict whether rebalancing will improve future risk-adjusted performance
5. Backtest all strategies **out-of-sample** on the test period
6. Save results as `.csv`, a summary table as `.md`, and plots as `.png`

---

## Dataset and split
Configured in `config.py`:

- Date range: **2010-01-01 to 2024-12-31**  
  (implemented via `end_date="2025-01-01"` because yfinance excludes the end date)
- Train/Test split:
  - Train: up to **2017-12-31**
  - Test backtest: **2018-01-01 to 2024-12-31**

---
## Strategies implemented
1. **Buy & Hold** (no rebalancing)
2. **Calendar rebalancing**: monthly / quarterly / yearly
3. **Threshold rebalancing**: rebalance when drift exceeds **3% / 5% / 7%**
4. **ML trigger**:
   - A Random Forest outputs daily probabilities
   - A rolling-quantile trigger (`prob_quantile`, `ml_thr_window`) detects unusually high predicted benefit
   - Trades execute with a **1-day lag** to reduce look-ahead bias

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

# this will download, clean the prices, compute the features and labels, train the model, run backtests on the test period, generate plots and tables.

## Regenerating Outputs

If you want to regenerate all figures and tables from scratch:

1. **Delete existing outputs:**
```bash
rm outputs/results/backtest_IWM_TLT.csv
rm outputs/figures/*.png
rm outputs/tables/summary_table.md
```

2. **Run the pipeline:**
```bash
python -m main
```

**Note:** The pipeline will skip regenerating outputs if they already exist. Delete them first if you want fresh results.

#Optional: there is also separate tests, present in most files. If you want to run use:

python -m src.data_processing 
python -m src.features
python -m src.models
python -m src.visualization
```
# Outputs:
1. outputs/results/backtest_IWM_TLT.csv
2. outputs/figures/value_curves_test.png
3. outputs/figures/drawdowns_test.png
4. outputs/tables/summary_table.md
5. outputs/figures/stock_weight_test.png
# Configuration
All parameters are set in config.py and can be changed to observe other outcomes:
1. etf tickers, data ranges and the train and test split
2. weights
3. transaction costs as a list
4. rebalancing data for calendar and drift based
5. ML parameters: prob_quantile, ml_thr_window, trade_lag_days
# Limitations
1. this has only 2-asset setting.
2. Labels are inherently noisy because future risk-adjusted performance is regime-dependent
3. there is no mention of taxes
4. The ML model may not outperform simple baselines out-of-sample; the main focus is a correct, reproducible pipeline with realistic constraints (transaction costs + execution lag)
# Reproducibility:
First run downloads and caches prices in data/raw/…csv; subsequent runs reuse cached data unless refresh=True.
Everything is deterministic given the config settings, but to force refreshes on the dataset, one must set:
```bash
# from
px = clean_prices(download_prices(refresh=False))
# to 
px = clean_prices(download_prices(refresh=True))
```
