from pathlib import Path

import yfinance as yf

from src.config import START_DATE, TICKERS


def download_stock_data(tickers=None, start_date=START_DATE):
    """Download daily historical stock data for the selected tickers."""
    tickers = tickers or TICKERS
    data = yf.download(tickers, start=start_date, auto_adjust=False, group_by="ticker")
    return data


def save_raw_stock_data(output_path="data/raw/stock_prices.csv"):
    """Download and save raw stock data to CSV."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = download_stock_data()
    data.to_csv(output_path)
    return output_path

