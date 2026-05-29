import os

import requests


ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def fetch_news_sentiment(ticker, api_key=None, limit=50):
    """Get recent news sentiment data for one ticker."""
    api_key = api_key or os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        raise ValueError("Missing Alpha Vantage API key.")

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": limit,
        "apikey": api_key,
    }
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()
