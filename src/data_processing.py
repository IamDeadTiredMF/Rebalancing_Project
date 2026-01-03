import pandas as pd
import yfinance as yf
import config


def download_prices(refresh=False):
    config.make_dirs()
    path = config.get_price_file_path()

    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)

    df = yf.download([config.stock_ticker_etf, config.bond_ticker_etf], start=config.start_date, end=config.end_date, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("Downloaded data is empty.")

    prices = df["Close"].copy()
    prices = prices.rename(columns={config.stock_ticker_etf: "stock", config.bond_ticker_etf: "bond"})
    prices.to_csv(path)
    return prices


def clean_prices(df):
    if df is None:
        raise ValueError("clean_prices received None.")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df = df.dropna()
    return df


if __name__ == "__main__":
    px = clean_prices(download_prices(refresh=False))
    print(px.head())