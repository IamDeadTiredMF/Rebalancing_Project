**Author:** Marius Fortuna  
**Date:** November 2025 (Updated following TA feedback)

## Research Question
Can a machine learning classifier learn when rebalancing is beneficial (after costs and execution lags) to reduce unnecessary turnover while maintaining competitive risk-adjusted performance?

## Proposed Methodology
Following feedback from Anna Smirnova to ensure the project meets the "Data Science" requirements of the course, I am reframing the rebalancing decision as a **Classification Problem**.

### 1. Data & Assets
- **Assets:** IWM (Equities) and TLT (Bonds).
- **Source:** Yahoo Finance (`yfinance`).

### 2. Machine Learning Approach
- **Model:** Random Forest Classifier.
- **Features:** Portfolio drift, rolling volatilities, stock-bond correlations, and regime proxies.
- **Labels:** Binary (1 if rebalancing improves the 6-month forward Sortino ratio after costs; 0 otherwise).
- **Validation:** Out-of-sample backtest from 2018 to 2024.

### 3. Comparisons (The "Baseline")
To evaluate the ML model, I will compare its performance against:
- **Calendar Rules:** Monthly, Quarterly, and Yearly rebalancing.
- **Threshold Rules:** Rebalancing when drift exceeds 3%, 5%, or 7%.
- **Passive:** Buy & Hold strategy.

## Expected Contribution
The project will demonstrate that selective, "regime-aware" rebalancing is feasible in a reproducible research pipeline and can materially reduce turnover compared to rigid mechanical rules.