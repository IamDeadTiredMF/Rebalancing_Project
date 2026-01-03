import pandas as pd
import yfinance as yf
import config

def download_prices(refresh: bool = False):
    config.make_dirs()
    path = config.raw_data / f"prices_{config.stock_ticker_etf}_{config.bond_ticker_etf}.csv"

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = yf.download(
        [config.stock_ticker_etf, config.bond_ticker_etf],
        start=config.start_date,
        end=config.end_date,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise ValueError("Downloaded data is empty.")

    prices = df["Close"].copy()
    prices.columns = ["stock", "bond"] 
    prices.to_csv(path)
    return prices

def clean_prices(df: pd.DataFrame):
    if df is None:
        raise ValueError("clean_prices received None.")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna()
    return df

if __name__ == "__main__":
    prices = download_prices(refresh=False)
    prices_clean = clean_prices(prices)
    print(prices_clean.head())