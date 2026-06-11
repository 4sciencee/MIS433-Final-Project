import json
import os
from pathlib import Path

import altair as alt
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score


BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

TRAINING_DATA_PATH = BASE_DIR / "data" / "processed" / "training_ready_stock_data.csv"
LATEST_ROWS_PATH = BASE_DIR / "data" / "processed" / "latest_prediction_rows.csv"
MODEL_READY_PATH = BASE_DIR / "data" / "processed" / "model_ready_stock_data.csv"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"

FEATURE_COLUMNS = [
    "daily_return",
    "return_7d",
    "return_30d",
    "ma_7d",
    "ma_30d",
    "ma_90d",
    "volatility_30d",
    "volume_change",
    "avg_sentiment_score",
    "article_count",
]

TARGET_COLUMN = "target_up_7d"
PREDICTION_THRESHOLD = 0.45
REFRESH_LIVE_STOCK_DATA = os.getenv("REFRESH_LIVE_STOCK_DATA", "true").lower() == "true"
REFRESH_LIVE_SENTIMENT_DATA = os.getenv("REFRESH_LIVE_SENTIMENT_DATA", "true").lower() == "true"

COLOR_TOKENS = {
    "page": "#171716",
    "panel": "#242521",
    "card": "#2F302D",
    "text": "#F3F1EA",
    "muted": "#C7C3B8",
    "subtle": "#9A968C",
    "border": "rgba(255, 255, 255, .14)",
    "strong": "#1D9E75",
    "strong_text": "#8EEA76",
    "caution": "#EF9F27",
    "caution_text": "#F4B860",
    "negative": "#E24B4A",
    "info": "#378ADD",
    "info_text": "#BFD8FF",
}


st.set_page_config(page_title="AI Investment Signals", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');

    :root {
        --color-page: __PAGE__;
        --color-panel: __PANEL__;
        --color-card: __CARD__;
        --color-text-primary: __TEXT__;
        --color-text-secondary: __MUTED__;
        --color-text-tertiary: __SUBTLE__;
        --color-border: __BORDER__;
        --color-strong: __STRONG__;
        --color-strong-text: __STRONG_TEXT__;
        --color-caution: __CAUTION__;
        --color-caution-text: __CAUTION_TEXT__;
        --color-negative: __NEGATIVE__;
        --color-info: __INFO__;
        --color-info-text: __INFO_TEXT__;
        --color-background-primary: var(--color-card);
        --color-background-secondary: var(--color-page);
        --color-background-success: rgba(29, 158, 117, .16);
        --color-text-success: var(--color-strong-text);
        --color-background-warning: rgba(239, 159, 39, .16);
        --color-text-warning: var(--color-caution-text);
        --color-background-info: rgba(55, 138, 221, .16);
        --color-text-info: var(--color-info-text);
        --color-border-tertiary: var(--color-border);
        --color-border-secondary: rgba(255, 255, 255, .22);
        --color-border-success: rgba(29, 158, 117, .45);
        --color-border-warning: rgba(239, 159, 39, .45);
        --color-border-info: rgba(55, 138, 221, .55);
        --border-radius-md: 8px;
        --border-radius-lg: 12px;
    }

    html, body, [data-testid="stAppViewContainer"] {
        background: var(--color-page);
        color: var(--color-text-primary);
        font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    h1, h2, h3, h4, p {
        letter-spacing: 0;
        font-weight: 500;
    }

    div[data-testid="stVerticalBlock"] {
        gap: .85rem;
    }

    .app-hero {
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-lg);
        background: linear-gradient(180deg, var(--color-card) 0%, var(--color-panel) 100%);
        padding: 20px 22px;
        margin-bottom: 14px;
        box-shadow: 0 14px 34px rgba(0, 0, 0, .22);
    }

    .app-title {
        color: var(--color-text-primary);
        font-size: 28px;
        line-height: 1.15;
        margin: 0 0 7px;
        font-weight: 600;
    }

    .app-subtitle {
        color: var(--color-text-secondary);
        font-size: 14px;
        line-height: 1.5;
        max-width: 760px;
    }

    .disclaimer-pill {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        background: var(--color-background-secondary);
        color: var(--color-text-secondary);
        border: .5px solid var(--color-border-tertiary);
        padding: 5px 10px;
        font-size: 12px;
        margin-top: 12px;
    }

    .page-intro {
        display: flex;
        align-items: flex-start;
        justify-content: space-between;
        gap: 16px;
        margin: 8px 0 12px;
    }

    .page-title {
        color: var(--color-text-primary);
        font-size: 24px;
        line-height: 1.2;
        margin: 0 0 5px;
        font-weight: 600;
    }

    .page-copy {
        color: var(--color-text-secondary);
        font-size: 13px;
        line-height: 1.5;
    }

    .source-pill {
        color: var(--color-text-tertiary);
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 11px;
        white-space: nowrap;
    }

    .kpi-card {
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-lg);
        padding: 15px 16px;
        min-height: 105px;
        box-shadow: 0 10px 22px rgba(0, 0, 0, .14);
    }

    .kpi-label {
        color: var(--color-text-secondary);
        font-size: 12px;
        margin-bottom: 7px;
    }

    .kpi-value {
        color: var(--color-text-primary);
        font-size: 28px;
        line-height: 1;
        font-weight: 600;
        margin-bottom: 9px;
    }

    .kpi-note {
        display: inline-flex;
        border-radius: 999px;
        background: var(--color-background-success);
        color: var(--color-strong-text);
        padding: 3px 8px;
        font-size: 12px;
    }

    .section-label {
        color: var(--color-text-primary);
        font-size: 17px;
        font-weight: 600;
        margin: 12px 0 6px;
    }

    .section-copy {
        color: var(--color-text-tertiary);
        font-size: 12px;
        margin-bottom: 8px;
    }

    .insight-card {
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-lg);
        padding: 14px 15px;
        min-height: 92px;
    }

    .insight-title {
        color: var(--color-text-secondary);
        font-size: 12px;
        margin-bottom: 6px;
    }

    .insight-main {
        color: var(--color-text-primary);
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .insight-detail {
        color: var(--color-text-tertiary);
        font-size: 12px;
        line-height: 1.45;
    }

    .ai-panel {
        background: var(--color-background-info);
        border: .5px solid var(--color-border-info);
        border-radius: var(--border-radius-lg);
        padding: 14px 15px;
        margin-top: 10px;
    }

    .ai-panel-title {
        color: var(--color-text-primary);
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .ai-panel-copy {
        color: var(--color-text-secondary);
        font-size: 12px;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    .result-banner {
        background: var(--color-background-success);
        border: .5px solid var(--color-border-success);
        border-radius: var(--border-radius-lg);
        padding: 13px 15px;
        color: var(--color-text-success);
        font-size: 15px;
        font-weight: 600;
    }

    .small-note {
        color: var(--color-text-tertiary);
        font-size: 12px;
        line-height: 1.5;
    }

    .pill-strong, .pill-caution, .pill-muted {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: var(--border-radius-md);
        padding: 3px 9px;
        font-size: 12px;
        font-weight: 600;
        white-space: nowrap;
    }

    .pill-strong {
        background: var(--color-background-success);
        color: var(--color-strong-text);
        border: .5px solid var(--color-border-success);
    }

    .pill-caution {
        background: var(--color-background-warning);
        color: var(--color-caution-text);
        border: .5px solid var(--color-border-warning);
    }

    .pill-muted {
        background: var(--color-panel);
        color: var(--color-text-tertiary);
        border: .5px solid var(--color-border);
    }

    .metric-row {
        display: grid;
        grid-template-columns: 128px 1fr 1fr;
        gap: 14px;
        align-items: center;
        padding: 11px 0;
        border-top: .5px solid var(--color-border);
    }

    .metric-name {
        color: var(--color-text-secondary);
        font-size: 13px;
    }

    .metric-cell {
        min-width: 0;
    }

    .metric-value {
        color: var(--color-text-primary);
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .metric-value.winner {
        color: var(--color-strong-text);
    }

    .track {
        height: 8px;
        border-radius: 999px;
        background: var(--color-panel);
        overflow: hidden;
    }

    .fill {
        height: 100%;
        border-radius: 999px;
    }

    .filter-caption {
        color: var(--color-text-secondary);
        font-size: 13px;
        margin-top: 3px;
    }

    .frame {
        background: var(--color-background-secondary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-lg);
        overflow: hidden;
        margin-bottom: 1.5rem;
    }

    .bar {
        background: var(--color-background-primary);
        border-bottom: .5px solid var(--color-border-tertiary);
        padding: 9px 13px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 8px;
    }

    .brand {
        color: var(--color-text-primary);
        font-size: 14px;
        font-weight: 500;
    }

    .chip, .nav-chip, .input-chip {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        border-radius: var(--border-radius-md);
        background: var(--color-background-secondary);
        color: var(--color-text-secondary);
        padding: 4px 9px;
        font-size: 12px;
        font-weight: 400;
        line-height: 1.2;
    }

    .foot {
        font-size: 12px;
        color: var(--color-text-secondary);
        padding: 9px 13px;
        border-top: .5px solid var(--color-border-tertiary);
        background: var(--color-background-primary);
        margin-top: .9rem;
    }

    .side-panel {
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-md);
        padding: 12px;
        min-height: 590px;
    }

    .main-panel {
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-md);
        padding: 14px;
        min-height: 590px;
    }

    .side-label {
        font-size: 11px;
        color: var(--color-text-tertiary);
        margin-bottom: 6px;
        font-weight: 400;
    }

    .source-note {
        font-size: 11px;
        color: var(--color-text-tertiary);
        line-height: 1.6;
        margin-top: 14px;
    }

    .screen-label {
        color: var(--color-text-tertiary);
        font-size: 12px;
        margin: 0 0 7px 2px;
    }

    .result-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 10px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }

    .ticker {
        color: var(--color-text-primary);
        font-size: 24px;
        font-weight: 500;
        line-height: 1.1;
    }

    .subtitle {
        color: var(--color-text-secondary);
        font-size: 12px;
        margin-top: 2px;
    }

    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        border-radius: var(--border-radius-md);
        padding: 5px 12px;
        font-size: 13px;
        font-weight: 500;
        white-space: nowrap;
    }

    .pill-up {
        background: var(--color-background-success);
        color: var(--color-text-success);
    }

    .pill-caution {
        background: var(--color-background-secondary);
        color: var(--color-text-secondary);
    }

    .prob-card {
        background: var(--color-background-secondary);
        border-radius: var(--border-radius-md);
        padding: 11px 13px;
        margin: 14px 0;
    }

    .prob-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 1rem;
    }

    .prob-label {
        color: var(--color-text-secondary);
        font-size: 12px;
        font-weight: 400;
    }

    .prob-value {
        color: var(--color-text-primary);
        font-size: 22px;
        font-weight: 500;
    }

    .bar-track {
        height: 7px;
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: 4px;
        margin-top: 8px;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        background: var(--color-text-info);
        border-radius: 4px;
    }

    .bar-labels {
        display: flex;
        justify-content: space-between;
        color: var(--color-text-tertiary);
        font-size: 11px;
        margin-top: 4px;
    }

    .metric-card {
        background: var(--color-background-secondary);
        border-radius: var(--border-radius-md);
        padding: 10px 12px;
        min-height: 72px;
    }

    .metric-label {
        color: var(--color-text-secondary);
        font-size: 12px;
        font-weight: 400;
        margin-bottom: 3px;
    }

    .metric-value {
        color: var(--color-text-primary);
        font-size: 19px;
        font-weight: 500;
    }

    .section-title {
        color: var(--color-text-primary);
        font-size: 13px;
        font-weight: 500;
        margin: 16px 0 7px;
    }

    .chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 7px;
    }

    .caption {
        color: var(--color-text-tertiary);
        font-size: 11px;
        margin-top: 6px;
    }

    .summary-box {
        border: .5px solid var(--color-border-info);
        border-radius: var(--border-radius-md);
        background: var(--color-background-primary);
        padding: 12px 13px;
        color: var(--color-text-secondary);
        font-size: 13px;
        line-height: 1.7;
        margin-top: 14px;
    }

    .summary-title {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: .5rem;
        color: var(--color-text-primary);
        font-size: 13px;
        font-weight: 500;
        margin-bottom: 7px;
    }

    .ai-tag {
        border-radius: var(--border-radius-md);
        background: var(--color-background-info);
        color: var(--color-text-info);
        padding: 3px 8px;
        font-size: 11px;
        font-weight: 400;
        white-space: nowrap;
    }

    .comparison-card {
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-md);
        padding: 12px;
        min-height: 205px;
    }

    .comparison-title {
        color: var(--color-text-primary);
        font-size: 18px;
        font-weight: 500;
        margin-bottom: 8px;
    }

    .comparison-lines {
        color: var(--color-text-secondary);
        font-size: 12px;
        line-height: 2;
        margin-top: 10px;
    }

    .rank-card {
        background: var(--color-background-secondary);
        border-radius: var(--border-radius-md);
        padding: 10px 12px;
    }

    .rank-card .label {
        color: var(--color-text-secondary);
        font-size: 11px;
        font-weight: 400;
    }

    .rank-card .value {
        color: var(--color-text-primary);
        font-size: 14px;
        font-weight: 500;
        margin-top: 2px;
    }

    .stButton > button {
        border-radius: var(--border-radius-md);
        font-weight: 500;
        min-height: 36px;
        background: var(--color-background-info);
        color: var(--color-text-info);
        border: .5px solid var(--color-border-info);
    }

    .stSelectbox label, .stSlider label, .stSegmentedControl label {
        color: var(--color-text-secondary) !important;
        font-size: 12px !important;
        font-weight: 400 !important;
    }

    [data-testid="stDataFrame"] {
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-md);
        overflow: hidden;
    }

    [data-testid="stMetric"] {
        background: var(--color-background-primary);
        border: .5px solid var(--color-border-tertiary);
        border-radius: var(--border-radius-lg);
        padding: 14px 15px;
    }

    [data-testid="stMetricLabel"] {
        color: var(--color-text-secondary);
    }

    [data-testid="stMetricValue"] {
        color: var(--color-text-primary);
        font-size: 24px;
        font-weight: 600;
    }

    [data-testid="stTabs"] [role="tablist"] {
        gap: 8px;
        border-bottom: .5px solid var(--color-border-tertiary);
    }

    [data-testid="stTabs"] [role="tab"] {
        border-radius: var(--border-radius-md) var(--border-radius-md) 0 0;
        padding: 8px 12px;
        color: var(--color-text-secondary);
        font-weight: 500;
    }

    [data-testid="stTabs"] [aria-selected="true"] {
        background: var(--color-background-primary);
        color: var(--color-text-primary);
    }

    div[data-testid="stExpander"] {
        border-color: var(--color-border-tertiary);
        background: var(--color-background-primary);
        border-radius: var(--border-radius-md);
    }

    div[data-testid="stAlert"] {
        border-radius: var(--border-radius-lg);
        border: .5px solid var(--color-border-info);
    }
    </style>
    """.replace("__PAGE__", COLOR_TOKENS["page"])
    .replace("__PANEL__", COLOR_TOKENS["panel"])
    .replace("__CARD__", COLOR_TOKENS["card"])
    .replace("__TEXT__", COLOR_TOKENS["text"])
    .replace("__MUTED__", COLOR_TOKENS["muted"])
    .replace("__SUBTLE__", COLOR_TOKENS["subtle"])
    .replace("__BORDER__", COLOR_TOKENS["border"])
    .replace("__STRONG__", COLOR_TOKENS["strong"])
    .replace("__STRONG_TEXT__", COLOR_TOKENS["strong_text"])
    .replace("__CAUTION__", COLOR_TOKENS["caution"])
    .replace("__CAUTION_TEXT__", COLOR_TOKENS["caution_text"])
    .replace("__NEGATIVE__", COLOR_TOKENS["negative"])
    .replace("__INFO__", COLOR_TOKENS["info"])
    .replace("__INFO_TEXT__", COLOR_TOKENS["info_text"]),
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    training_df = pd.read_csv(TRAINING_DATA_PATH)
    latest_df = pd.read_csv(LATEST_ROWS_PATH)
    model_ready_df = pd.read_csv(MODEL_READY_PATH)

    training_df["date"] = pd.to_datetime(training_df["date"])
    latest_df["date"] = pd.to_datetime(latest_df["date"])
    model_ready_df["date"] = pd.to_datetime(model_ready_df["date"])

    return training_df, latest_df, model_ready_df


def get_secret_value(secret_name, env_name=None):
    try:
        return st.secrets[secret_name]
    except Exception:
        return os.getenv(env_name or secret_name)


def get_alpha_vantage_key():
    return get_secret_value("ALPHA_VANTAGE_API_KEY")


def get_openai_key():
    return (
        get_secret_value("Chatgpt-MIS-433")
        or get_secret_value("OPENAI_API_KEY")
    )


def add_stock_features(stock_df):
    stock_df = stock_df.sort_values(["ticker", "date"]).copy()
    grouped = stock_df.groupby("ticker")

    stock_df["daily_return"] = grouped["close"].pct_change(fill_method=None)
    stock_df["return_7d"] = grouped["close"].pct_change(7, fill_method=None)
    stock_df["return_30d"] = grouped["close"].pct_change(30, fill_method=None)
    stock_df["ma_7d"] = grouped["close"].transform(lambda values: values.rolling(7).mean())
    stock_df["ma_30d"] = grouped["close"].transform(lambda values: values.rolling(30).mean())
    stock_df["ma_90d"] = grouped["close"].transform(lambda values: values.rolling(90).mean())
    stock_df["volatility_30d"] = grouped["daily_return"].transform(lambda values: values.rolling(30).std())
    stock_df["volume_change"] = grouped["volume"].pct_change(fill_method=None)

    return stock_df


@st.cache_data(ttl=3600)
def get_live_sentiment(tickers):
    alpha_key = get_alpha_vantage_key()

    if not REFRESH_LIVE_SENTIMENT_DATA or not alpha_key:
        return pd.DataFrame(), "Saved Alpha Vantage sentiment"

    sentiment_rows = []

    try:
        for ticker in tickers:
            params = {
                "function": "NEWS_SENTIMENT",
                "tickers": ticker,
                "apikey": alpha_key,
                "limit": 50,
            }
            response = requests.get("https://www.alphavantage.co/query", params=params, timeout=20)
            response.raise_for_status()
            data = response.json()

            if "feed" not in data:
                continue

            for article in data.get("feed", []):
                for ticker_info in article.get("ticker_sentiment", []):
                    if ticker_info.get("ticker") == ticker:
                        sentiment_rows.append(
                            {
                                "ticker": ticker,
                                "sentiment_score": float(ticker_info.get("ticker_sentiment_score", 0)),
                                "relevance_score": float(ticker_info.get("relevance_score", 0)),
                            }
                        )

        if not sentiment_rows:
            return pd.DataFrame(), "Saved Alpha Vantage sentiment"

        sentiment_df = pd.DataFrame(sentiment_rows)
        live_sentiment = sentiment_df.groupby("ticker").agg(
            avg_sentiment_score=("sentiment_score", "mean"),
            article_count=("sentiment_score", "count"),
        ).reset_index()

        return live_sentiment, "Live Alpha Vantage sentiment"

    except Exception as error:
        st.warning(f"Live Alpha Vantage sentiment was not available, so saved sentiment is being used. Details: {error}")
        return pd.DataFrame(), "Saved Alpha Vantage sentiment"


@st.cache_data(ttl=900)
def get_app_market_data(latest_df, model_ready_df, live_sentiment_df):
    if not REFRESH_LIVE_STOCK_DATA:
        return latest_df.copy(), model_ready_df.copy(), "Saved project CSV files"

    try:
        import yfinance as yf

        live_data = []
        tickers = sorted(model_ready_df["ticker"].unique())

        for ticker in tickers:
            ticker_df = yf.download(
                ticker,
                period="9mo",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if ticker_df.empty:
                continue

            if isinstance(ticker_df.columns, pd.MultiIndex):
                ticker_df.columns = ticker_df.columns.get_level_values(0)

            ticker_df = ticker_df.reset_index()
            ticker_df.columns = [str(column).lower().replace(" ", "_") for column in ticker_df.columns]
            ticker_df["ticker"] = ticker

            if "adj_close" not in ticker_df.columns:
                ticker_df["adj_close"] = ticker_df["close"]

            live_data.append(
                ticker_df[["date", "open", "high", "low", "close", "adj_close", "volume", "ticker"]]
            )

        if not live_data:
            raise ValueError("No live stock rows returned from yfinance.")

        live_df = pd.concat(live_data, ignore_index=True)
        live_df["date"] = pd.to_datetime(live_df["date"])
        live_df = live_df.dropna(subset=["close"]).copy()
        live_df = add_stock_features(live_df)

        sentiment_columns = ["ticker", "avg_sentiment_score", "article_count"]
        saved_sentiment = latest_df[sentiment_columns].drop_duplicates("ticker")

        if live_sentiment_df.empty:
            latest_sentiment = saved_sentiment
            sentiment_status = "saved Alpha Vantage sentiment"
        else:
            live_sentiment = live_sentiment_df[sentiment_columns].drop_duplicates("ticker")
            latest_sentiment = pd.concat([saved_sentiment, live_sentiment], ignore_index=True)
            latest_sentiment = latest_sentiment.drop_duplicates("ticker", keep="last")
            sentiment_status = "live Alpha Vantage sentiment with saved fallback"

        live_df = live_df.merge(latest_sentiment, on="ticker", how="left")
        live_df["avg_sentiment_score"] = live_df["avg_sentiment_score"].fillna(0)
        live_df["article_count"] = live_df["article_count"].fillna(0)

        live_latest_rows = live_df.sort_values("date").groupby("ticker").tail(1).copy()
        live_latest_rows = live_latest_rows.dropna(subset=FEATURE_COLUMNS)

        if len(live_latest_rows) < len(tickers):
            raise ValueError("Live data did not include enough rows for all model features.")

        return live_latest_rows, live_df, f"Live yfinance refresh with {sentiment_status}"

    except Exception as error:
        st.warning(f"Live stock refresh was not available, so the app is using saved project data. Details: {error}")
        return latest_df.copy(), model_ready_df.copy(), "Saved project CSV files"


@st.cache_resource
def train_model(training_df):
    model_df = training_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    model_df = model_df.sort_values("date")

    split_date = model_df["date"].quantile(0.80)
    train_df = model_df[model_df["date"] <= split_date]
    test_df = model_df[model_df["date"] > split_date]

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )
    model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])

    test_probability = model.predict_proba(test_df[FEATURE_COLUMNS])[:, 1]
    test_prediction = (test_probability >= PREDICTION_THRESHOLD).astype(int)
    balanced_accuracy = balanced_accuracy_score(test_df[TARGET_COLUMN], test_prediction)

    return model, balanced_accuracy, len(train_df), len(test_df)


def format_percent(value):
    return f"{value * 100:.1f}%"


def format_money(value):
    return f"${value:,.2f}"


def signal_label(signal):
    return "Likely up" if signal == "Positive Signal" else "No clear up-signal"


def signal_css(signal):
    return "pill-up" if signal == "Positive Signal" else "pill-caution"


def get_prediction(model, row):
    probability = model.predict_proba(row[FEATURE_COLUMNS])[:, 1][0]
    predicted_up = int(probability >= PREDICTION_THRESHOLD)
    signal = "Positive Signal" if predicted_up == 1 else "Caution"
    return probability, signal


def create_prediction_table(model, latest_df):
    rows = []
    for ticker in sorted(latest_df["ticker"].unique()):
        ticker_row = latest_df[latest_df["ticker"] == ticker].iloc[[0]].copy()
        probability, signal = get_prediction(model, ticker_row)
        data = ticker_row.iloc[0]
        rows.append(
            {
                "Ticker": ticker,
                "Signal": signal,
                "Probability": probability,
                "Latest Close": data["close"],
                "7-Day Return": data["return_7d"],
                "30-Day Volatility": data["volatility_30d"],
                "Sentiment": data["avg_sentiment_score"],
                "Articles": int(data["article_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("Probability", ascending=False)


def get_company_row(prediction_table, ticker):
    return prediction_table[prediction_table["Ticker"] == ticker].iloc[0]


def create_template_summary(row, risk_preference, mode_name):
    return (
        f"{row['Ticker']} shows a {signal_label(row['Signal']).lower()} result for the 7-trading-day window. "
        f"The upward-move probability is {format_percent(row['Probability'])}. Recent return is "
        f"{format_percent(row['7-Day Return'])}, 30-day volatility is {format_percent(row['30-Day Volatility'])}, "
        f"and sentiment is {row['Sentiment']:.3f} across {int(row['Articles'])} article(s). "
        f"Risk preference selected: {risk_preference}. This {mode_name.lower()} is decision support only."
    )


def create_openai_summary(prompt, fallback_summary):
    api_key = get_openai_key()
    if not api_key:
        return fallback_summary

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Write clear and simple investment signal summaries for a business analytics class project. Do not give financial advice.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=230,
        )
        return response.choices[0].message.content
    except Exception:
        return fallback_summary


@st.cache_data(ttl=3600)
def create_summary_maps(prediction_records_json, openai_available, generate_openai):
    prediction_records = json.loads(prediction_records_json)
    briefing_summaries = {}
    comparison_summaries = {}

    for row in prediction_records:
        fallback = (
            f"{row['ticker']} shows a {row['signal'].lower()} for the next 7 trading days, "
            f"with an upward-move probability of {format_percent(row['probability'])}. Recent return is "
            f"{format_percent(row['return7'])}, 30-day volatility is {format_percent(row['volatility30'])}, "
            f"and sentiment is {row['sentiment']:.3f} across {int(row['articles'])} articles. "
            "This is decision support, not financial advice."
        )
        briefing_summaries[row["ticker"]] = fallback

    for row_a in prediction_records:
        for row_b in prediction_records:
            if row_a["ticker"] == row_b["ticker"]:
                continue

            leader = row_a if row_a["probability"] >= row_b["probability"] else row_b
            other = row_b if leader == row_a else row_a
            key = f"{row_a['ticker']}|{row_b['ticker']}"
            fallback = (
                f"{leader['ticker']} has the stronger 7-day upward signal at "
                f"{format_percent(leader['probability'])} compared with {other['ticker']} at "
                f"{format_percent(other['probability'])}. The comparison should also consider volatility, "
                "recent return, and sentiment. Decision support only, not financial advice."
            )
            comparison_summaries[key] = fallback

    sorted_records = sorted(prediction_records, key=lambda item: item["probability"], reverse=True)
    highest = sorted_records[0]
    lowest = sorted_records[-1]
    watchlist_fallback = (
        f"Across the selected companies, {highest['ticker']} ranks highest for a 7-day upward move and "
        f"{lowest['ticker']} ranks lowest. Sentiment, article count, and volatility help explain why the "
        "ranking differs by company. Decision support only, not financial advice."
    )
    watchlist_summary = watchlist_fallback

    if openai_available and generate_openai:
        try:
            from openai import OpenAI

            client = OpenAI(api_key=get_openai_key())
            summary_prompt = (
                "Create short, clear summary text for a beginner business analytics stock signal app. "
                "Do not give financial advice. Return only JSON with these keys: briefing, comparison, watchlist. "
                "briefing should be an object keyed by ticker. comparison should be an object keyed like 'NVDA|AMD'. "
                "watchlist should be one string. Keep each summary 1-2 short sentences. "
                f"Prediction records: {json.dumps(prediction_records)}. "
                f"Comparison keys needed: {json.dumps(list(comparison_summaries.keys()))}."
            )
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Return valid JSON only. Write simple decision-support summaries, not financial advice.",
                    },
                    {"role": "user", "content": summary_prompt},
                ],
                temperature=0.3,
                max_tokens=2400,
                response_format={"type": "json_object"},
            )
            ai_data = json.loads(response.choices[0].message.content)
            briefing_summaries.update(ai_data.get("briefing", {}))
            comparison_summaries.update(ai_data.get("comparison", {}))
            watchlist_summary = ai_data.get("watchlist", watchlist_summary)
            summary_source = "OpenAI"
        except Exception:
            summary_source = "Template fallback"
    elif openai_available:
        summary_source = "Ready for OpenAI"
    else:
        summary_source = "Template fallback"

    return {
        "briefing": briefing_summaries,
        "comparison": comparison_summaries,
        "watchlist": watchlist_summary,
        "source": summary_source,
    }


def metric_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def probability_card(probability):
    width = max(0, min(100, probability * 100))
    st.markdown(
        f"""
        <div class="prob-card">
            <div class="prob-head">
                <span class="prob-label">Probability of upward move</span>
                <span class="prob-value">{format_percent(probability)}</span>
            </div>
            <div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>
            <div class="bar-labels"><span>0%</span><span>100%</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary(title, summary):
    st.markdown(
        f"""
        <div class="summary-box">
            <div class="summary-title">
                <span>{title}</span>
                <span class="ai-tag">AI - OpenAI</span>
            </div>
            <div>{summary}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_table(df):
    table = df.copy()
    if "Probability" in table:
        table["Probability"] = table["Probability"].map(format_percent)
    if "Latest Close" in table:
        table["Latest Close"] = table["Latest Close"].map(format_money)
    if "7-Day Return" in table:
        table["7-Day Return"] = table["7-Day Return"].map(format_percent)
    if "30-Day Volatility" in table:
        table["30-Day Volatility"] = table["30-Day Volatility"].map(format_percent)
    if "Sentiment" in table:
        table["Sentiment"] = table["Sentiment"].map(lambda value: f"{value:.3f}")
    if "Signal" in table:
        table["Signal"] = table["Signal"].map(signal_label)
    return table


def render_trend_chart(model_ready_df, ticker):
    history = model_ready_df[model_ready_df["ticker"] == ticker].sort_values("date").tail(90)
    trend = history.set_index("date")[["close", "ma_7d", "ma_30d", "ma_90d"]]
    trend = trend.rename(
        columns={
            "close": "Close",
            "ma_7d": "MA 7",
            "ma_30d": "MA 30",
            "ma_90d": "MA 90",
        }
    )
    st.line_chart(trend, height=175)


def render_frame_header():
    st.markdown(
        """
        <div class="frame">
            <div class="bar">
                <span class="brand">AI Investment Signal Assistant</span>
                <span class="chip">Decision support - not advice</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        """
        <div class="foot">
            Decision-support tool - not financial advice. Signals are model estimates, not guarantees.
        </div>
        """,
        unsafe_allow_html=True,
    )


def html_select_options(tickers, selected):
    options = []
    for ticker in tickers:
        selected_text = " selected" if ticker == selected else ""
        options.append(f"<option{selected_text}>{ticker}</option>")
    return "".join(options)


def html_signal_pill(signal):
    if signal == "Positive Signal":
        return '<span class="pill"><i class="ti ti-trending-up" aria-hidden="true"></i> Likely up</span>'
    return (
        '<span class="pill neutral"><i class="ti ti-minus" aria-hidden="true"></i> '
        "No clear up-signal</span>"
    )


def create_chart_records(model_ready_df):
    chart_records = {}

    for ticker in sorted(model_ready_df["ticker"].unique()):
        history = model_ready_df[model_ready_df["ticker"] == ticker].sort_values("date").tail(90)
        rows = []

        for _, row in history.iterrows():
            rows.append(
                {
                    "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                    "close": None if pd.isna(row["close"]) else float(row["close"]),
                    "ma7": None if pd.isna(row["ma_7d"]) else float(row["ma_7d"]),
                    "ma30": None if pd.isna(row["ma_30d"]) else float(row["ma_30d"]),
                    "ma90": None if pd.isna(row["ma_90d"]) else float(row["ma_90d"]),
                }
            )

        chart_records[ticker] = rows

    return chart_records


def render_html_dashboard(prediction_table, chart_history_df, data_status, generate_openai):
    tickers = sorted(prediction_table["Ticker"])
    nvda = get_company_row(prediction_table, "NVDA" if "NVDA" in tickers else tickers[0])
    amd = get_company_row(prediction_table, "AMD" if "AMD" in tickers else tickers[-1])
    app_records = []
    for _, row in prediction_table.iterrows():
        app_records.append(
            {
                "ticker": row["Ticker"],
                "signal": signal_label(row["Signal"]),
                "probability": float(row["Probability"]),
                "latestClose": float(row["Latest Close"]),
                "return7": float(row["7-Day Return"]),
                "volatility30": float(row["30-Day Volatility"]),
                "sentiment": float(row["Sentiment"]),
                "articles": int(row["Articles"]),
            }
        )
    app_data = json.dumps(app_records)
    summary_maps = create_summary_maps(app_data, bool(get_openai_key()), generate_openai)
    summary_data = json.dumps(summary_maps)
    chart_data = json.dumps(create_chart_records(chart_history_df))
    if summary_maps["source"] == "OpenAI":
        summary_badge = "AI - OpenAI"
    elif summary_maps["source"] == "Ready for OpenAI":
        summary_badge = "Click to generate"
    else:
        summary_badge = "Template summary"

    html = f"""
    <html>
    <head>
      <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@tabler/icons-webfont@3/dist/tabler-icons.min.css">
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap" rel="stylesheet">
      <style>
      :root {{
        --color-page:#171716;
        --color-background-primary:#2F302D;
        --color-background-secondary:#242521;
        --color-background-tertiary:#20211E;
        --color-text-primary:#F3F1EA;
        --color-text-secondary:#C7C3B8;
        --color-text-tertiary:#9A968C;
        --color-border-tertiary:rgba(255,255,255,.12);
        --color-border-secondary:rgba(255,255,255,.22);
        --color-background-success:#135F13;
        --color-text-success:#8EEA76;
        --color-background-info:#315F99;
        --color-text-info:#BFD8FF;
        --color-border-info:#4A90E2;
        --border-radius-md:8px;
        --border-radius-lg:12px;
      }}
      *{{box-sizing:border-box}}
      body{{margin:0;background:var(--color-page);font-family:'Inter',system-ui,-apple-system,sans-serif;color:var(--color-text-primary);font-size:13px}}
      .wrap{{width:760px;max-width:calc(100vw - 28px);margin:0 auto;padding:6px 0 20px}}
      .screen-page{{display:none}}
      .screen-page.active{{display:block}}
      .sr-only{{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);border:0}}
      .frame{{height:860px;background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-lg);overflow:hidden;margin-bottom:1.5rem;box-shadow:0 12px 28px rgba(0,0,0,.20);display:flex;flex-direction:column}}
      .bar{{background:var(--color-background-primary);border-bottom:0.5px solid var(--color-border-tertiary);padding:9px 13px;display:flex;justify-content:space-between;align-items:center;gap:8px}}
      .body{{display:flex;align-items:stretch;flex:1;min-height:0}}
      .side{{width:180px;flex:none;border-right:0.5px solid var(--color-border-tertiary);padding:12px;background:var(--color-background-primary)}}
      .main{{flex:1;min-width:0;padding:14px;background:var(--color-background-primary);overflow:hidden}}
      .nav{{display:flex;align-items:center;gap:8px;padding:7px 9px;border-radius:var(--border-radius-md);font-size:13px;color:var(--color-text-secondary);margin-bottom:3px;border:0;background:transparent;width:100%;font-family:inherit;text-align:left;cursor:pointer;font-weight:500;line-height:1.2}}
      .nav.on{{background:var(--color-background-secondary);color:var(--color-text-primary)}}
      .lbl{{font-size:12px;color:var(--color-text-secondary);margin:14px 0 5px}}
      .mc{{background:var(--color-background-tertiary);border-radius:var(--border-radius-md);padding:10px 12px}}
      .mc .l{{font-size:12px;color:var(--color-text-secondary)}}
      .mc .v{{font-size:19px;font-weight:500;margin-top:3px}}
      .chip{{display:inline-flex;align-items:center;gap:5px;font-size:12px;padding:4px 9px;border-radius:var(--border-radius-md);background:var(--color-background-secondary);color:var(--color-text-secondary)}}
      .cap{{font-size:12px;color:var(--color-text-tertiary);margin:0 0 7px 2px}}
      .seg{{display:flex;border:0.5px solid var(--color-border-secondary);border-radius:var(--border-radius-md);overflow:hidden}}
      .seg button{{flex:1;text-align:center;font-size:12px;padding:6px 0;color:var(--color-text-secondary);background:transparent;border:0;font-family:inherit;cursor:pointer}}
      .seg button.on{{background:var(--color-background-secondary);color:var(--color-text-primary)}}
      .pill{{display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:500;padding:5px 12px;border-radius:var(--border-radius-md);background:var(--color-background-success);color:var(--color-text-success)}}
      .pill.neutral{{background:var(--color-background-secondary);color:var(--color-text-secondary)}}
      .cta{{background:transparent;color:var(--color-text-primary);border:0.5px solid var(--color-border-secondary);width:100%;justify-content:center;border-radius:var(--border-radius-md);padding:8px 10px;font-size:13px;font-weight:500}}
      .cta:hover,.nav:hover{{background:var(--color-background-secondary)}}
      .aitag{{font-size:11px;padding:3px 8px;border-radius:var(--border-radius-md);background:var(--color-background-info);color:var(--color-text-info);font-weight:500}}
      .foot{{font-size:12px;color:var(--color-text-secondary);display:flex;align-items:center;gap:6px;padding:9px 13px;border-top:0.5px solid var(--color-border-tertiary);background:var(--color-background-primary);min-height:36px}}
      select{{width:100%;height:34px;border:.5px solid var(--color-border-secondary);border-radius:var(--border-radius-md);background:var(--color-background-primary);color:var(--color-text-primary);font-family:inherit;font-size:13px;padding:0 8px;font-weight:500}}
      table{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:12px}}
      th{{font-weight:400;padding:6px 4px;color:var(--color-text-secondary);text-align:left}}
      td{{border-top:.5px solid var(--color-border-tertiary)}}
      .mini-chart{{margin-top:16px;border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-md);padding:12px 13px;background:var(--color-background-primary)}}
      .trend-chart{{height:92px;margin-top:8px}}
      .trend-chart svg{{width:100%;height:92px;display:block}}
      .chart-legend{{display:flex;gap:10px;flex-wrap:wrap;font-size:11px;color:var(--color-text-tertiary);margin-top:4px}}
      .legend-dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}}
      .summary-text{{font-size:12.5px;line-height:1.55;color:var(--color-text-secondary)}}
      .compare-card{{flex:1;min-width:0;border:0.5px solid var(--color-border-tertiary);border-radius:var(--border-radius-md);padding:12px;min-height:185px}}
      .watch-callout{{flex:1;min-width:170px;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:10px 12px}}
      </style>
    </head>
    <body>
    <div class="wrap">
      <h2 class="sr-only">UI mockup of the AI Investment Signal Assistant dashboard showing all three workflows with real model outputs.</h2>

      <div class="screen-page active" id="screen-briefing">
      <div class="frame">
        <div class="bar">
          <span style="display:flex;align-items:center;gap:8px;font-weight:500;font-size:14px"><i class="ti ti-activity-heartbeat"></i> AI Investment Signal Assistant</span>
          <span class="chip"><i class="ti ti-shield-half"></i> Decision support - not advice</span>
        </div>
        <div class="body">
          <div class="side">
            <div style="font-size:11px;color:var(--color-text-tertiary);margin-bottom:6px">Workflow</div>
            <button class="nav on" data-screen="screen-briefing"><i class="ti ti-file-analytics"></i> Company briefing</button>
            <button class="nav" data-screen="screen-comparison"><i class="ti ti-arrows-left-right"></i> Company comparison</button>
            <button class="nav" data-screen="screen-watchlist"><i class="ti ti-list-check"></i> Watchlist review</button>
            <div class="lbl">Company</div>
            <select id="briefing-company">{html_select_options(tickers, nvda["Ticker"])}</select>
            <div class="lbl">Risk preference</div>
            <div class="seg" data-risk-group="briefing"><button data-risk="Low">Low</button><button class="on" data-risk="Medium">Medium</button><button data-risk="High">High</button></div>
            <button class="cta" id="briefing-generate" style="margin-top:14px"><i class="ti ti-player-play"></i> Generate signal</button>
            <div style="font-size:11px;color:var(--color-text-tertiary);margin-top:14px;line-height:1.6">Data: {data_status}<br>Summary text: {summary_maps["source"]}</div>
          </div>
          <div class="main">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;flex-wrap:wrap">
              <div>
                <div id="briefing-ticker" style="font-size:24px;font-weight:500;line-height:1.1">{nvda["Ticker"]}</div>
                <div style="font-size:12px;color:var(--color-text-secondary);margin-top:2px">Selected company - 7-trading-day direction</div>
              </div>
              <div id="briefing-pill" style="text-align:right">{html_signal_pill(nvda["Signal"])}</div>
            </div>
            <div style="margin-top:14px;background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:11px 13px">
              <div style="display:flex;justify-content:space-between;align-items:baseline"><span style="font-size:12px;color:var(--color-text-secondary)">Probability of upward move</span><span id="briefing-probability" style="font-size:22px;font-weight:500">{format_percent(nvda["Probability"])}</span></div>
              <div style="height:7px;border-radius:4px;background:var(--color-background-primary);border:0.5px solid var(--color-border-tertiary);margin-top:8px;overflow:hidden"><div id="briefing-probability-bar" style="height:100%;width:{nvda["Probability"]*100:.1f}%;background:var(--color-text-info)"></div></div>
              <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--color-text-tertiary);margin-top:4px"><span>0%</span><span>100%</span></div>
            </div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:12px;margin-top:14px">
              <div class="mc"><div class="l">Latest close</div><div class="v" id="briefing-close"></div></div>
              <div class="mc"><div class="l">7-day return</div><div class="v" id="briefing-return"></div></div>
              <div class="mc"><div class="l">30-day volatility</div><div class="v" id="briefing-volatility"></div></div>
              <div class="mc"><div class="l">Sentiment score</div><div class="v" id="briefing-sentiment"></div></div>
              <div class="mc"><div class="l">Article count</div><div class="v" id="briefing-articles"></div></div>
            </div>
            <div style="margin-top:16px">
              <div style="font-size:13px;font-weight:500;margin-bottom:7px">Why this signal</div>
              <div style="display:flex;flex-wrap:wrap;gap:7px">
                <span class="chip"><i class="ti ti-chart-line"></i> Price trend (MA 7/30/90)</span>
                <span class="chip"><i class="ti ti-wave-sine"></i> 30-day volatility</span>
                <span class="chip"><i class="ti ti-chart-bar"></i> Trading volume</span>
                <span class="chip"><i class="ti ti-news"></i> News sentiment</span>
              </div>
              <div style="font-size:11px;color:var(--color-text-tertiary);margin-top:6px">Inputs used by the Random Forest model</div>
            </div>
            <div class="mini-chart">
              <div style="font-size:13px;font-weight:500">Price & moving averages - last 90 days</div>
              <div id="trend-chart" class="trend-chart"></div>
              <div class="chart-legend">
                <span><span class="legend-dot" style="background:#F3F1EA"></span>Close</span>
                <span><span class="legend-dot" style="background:#8EEA76"></span>MA 7</span>
                <span><span class="legend-dot" style="background:#BFD8FF"></span>MA 30</span>
                <span><span class="legend-dot" style="background:#F4B860"></span>MA 90</span>
              </div>
            </div>
            <div style="margin-top:16px;border:0.5px solid var(--color-border-info);border-radius:var(--border-radius-md);padding:12px 13px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px"><span style="font-size:13px;font-weight:500">Plain-English summary</span><span class="aitag"><i class="ti ti-sparkles"></i> {summary_badge}</span></div>
              <div id="briefing-summary" class="summary-text"></div>
            </div>
          </div>
        </div>
        <div class="foot"><i class="ti ti-info-circle"></i> Decision-support tool - not financial advice. Signals are model estimates, not guarantees.</div>
      </div>
      </div>

      <div class="screen-page" id="screen-comparison">
      <div class="frame">
        <div class="bar"><span style="display:flex;align-items:center;gap:8px;font-weight:500;font-size:14px"><i class="ti ti-activity-heartbeat"></i> AI Investment Signal Assistant</span><span class="chip"><i class="ti ti-shield-half"></i> Decision support - not advice</span></div>
        <div class="body">
          <div class="side">
            <div style="font-size:11px;color:var(--color-text-tertiary);margin-bottom:6px">Workflow</div>
            <button class="nav" data-screen="screen-briefing"><i class="ti ti-file-analytics"></i> Company briefing</button>
            <button class="nav on" data-screen="screen-comparison"><i class="ti ti-arrows-left-right"></i> Company comparison</button>
            <button class="nav" data-screen="screen-watchlist"><i class="ti ti-list-check"></i> Watchlist review</button>
            <div class="lbl">Company A</div><select id="compare-a-select">{html_select_options(tickers, nvda["Ticker"])}</select>
            <div class="lbl">Company B</div><select id="compare-b-select">{html_select_options(tickers, amd["Ticker"])}</select>
            <div class="lbl">Risk preference</div><div class="seg" data-risk-group="compare"><button data-risk="Low">Low</button><button class="on" data-risk="Medium">Medium</button><button data-risk="High">High</button></div>
            <button class="cta" id="compare-button" style="margin-top:14px"><i class="ti ti-arrows-left-right"></i> Compare</button>
          </div>
          <div class="main">
            <div style="font-size:15px;font-weight:500;margin-bottom:12px">Comparison</div>
            <div style="display:flex;align-items:stretch;gap:10px">
              <div class="compare-card">
                <div id="compare-a-title" style="font-size:18px;font-weight:500"></div><div id="compare-a-pill"></div><div id="compare-a-lines"></div>
              </div>
              <div style="display:flex;align-items:center;font-size:12px;color:var(--color-text-tertiary);font-weight:500">vs</div>
              <div class="compare-card">
                <div id="compare-b-title" style="font-size:18px;font-weight:500"></div><div id="compare-b-pill"></div><div id="compare-b-lines"></div>
              </div>
            </div>
            <div style="margin-top:14px;border:0.5px solid var(--color-border-info);border-radius:var(--border-radius-md);padding:12px 13px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px"><span style="font-size:13px;font-weight:500">AI comparison summary</span><span class="aitag"><i class="ti ti-sparkles"></i> {summary_badge}</span></div>
              <div id="compare-summary" class="summary-text"></div>
            </div>
          </div>
        </div>
        <div class="foot"><i class="ti ti-info-circle"></i> Decision-support tool - not financial advice. Signals are model estimates, not guarantees.</div>
      </div>
      </div>

      <div class="screen-page" id="screen-watchlist">
      <div class="frame">
        <div class="bar"><span style="display:flex;align-items:center;gap:8px;font-weight:500;font-size:14px"><i class="ti ti-activity-heartbeat"></i> AI Investment Signal Assistant</span><span class="chip"><i class="ti ti-shield-half"></i> Decision support - not advice</span></div>
        <div class="body">
          <div class="side">
            <div style="font-size:11px;color:var(--color-text-tertiary);margin-bottom:6px">Workflow</div>
            <button class="nav" data-screen="screen-briefing"><i class="ti ti-file-analytics"></i> Company briefing</button>
            <button class="nav" data-screen="screen-comparison"><i class="ti ti-arrows-left-right"></i> Company comparison</button>
            <button class="nav on" data-screen="screen-watchlist"><i class="ti ti-list-check"></i> Watchlist review</button>
            <div class="lbl">Companies</div><div style="font-size:12px;color:var(--color-text-secondary)">All 6 selected</div>
            <div class="lbl">Filter: probability >=</div><input id="watch-threshold" type="range" min="0" max="100" value="40" style="width:100%">
            <div id="watch-threshold-label" style="font-size:11px;color:var(--color-text-tertiary);margin-top:2px">threshold 40%</div>
            <div class="lbl">Risk preference</div><div class="seg" data-risk-group="watch"><button data-risk="Low">Low</button><button class="on" data-risk="Medium">Medium</button><button data-risk="High">High</button></div>
          </div>
          <div class="main">
            <div style="font-size:15px;font-weight:500;margin-bottom:4px">Watchlist - ranked by probability</div>
            <div style="font-size:11px;color:var(--color-text-tertiary);margin-bottom:10px">Values populated from the current model output</div>
            <table>
              <colgroup><col style="width:34px"><col style="width:62px"><col><col style="width:58px"><col style="width:60px"><col style="width:52px"></colgroup>
              <thead><tr><th>#</th><th>Company</th><th>Signal</th><th>Prob.</th><th>Sent.</th><th>Art.</th></tr></thead>
              <tbody id="watch-table-body"></tbody>
            </table>
            <div style="display:flex;gap:10px;margin-top:12px;flex-wrap:wrap">
              <div class="watch-callout"><div style="font-size:11px;color:var(--color-text-secondary)">Highest-ranked</div><div id="watch-highest" style="font-weight:500;margin-top:2px"></div></div>
              <div class="watch-callout"><div style="font-size:11px;color:var(--color-text-secondary)">Lowest in filter</div><div id="watch-lowest" style="font-weight:500;margin-top:2px"></div></div>
            </div>
            <div style="margin-top:14px;border:0.5px solid var(--color-border-info);border-radius:var(--border-radius-md);padding:12px 13px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:7px"><span style="font-size:13px;font-weight:500">AI watchlist summary</span><span class="aitag"><i class="ti ti-sparkles"></i> {summary_badge}</span></div>
              <div id="watch-summary" class="summary-text"></div>
            </div>
          </div>
        </div>
        <div class="foot"><i class="ti ti-info-circle"></i> Decision-support tool - not financial advice. Signals are model estimates, not guarantees.</div>
      </div>
      </div>
    </div>
    <script>
      const appData = {app_data};
      const chartData = {chart_data};
      const summaryData = {summary_data};
      const navButtons = document.querySelectorAll('.nav[data-screen]');
      const pages = document.querySelectorAll('.screen-page');
      const selectedRisks = {{ briefing: 'Medium', compare: 'Medium', watch: 'Medium' }};

      function money(value) {{
        return '$' + Number(value).toLocaleString(undefined, {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
      }}

      function percent(value) {{
        return (Number(value) * 100).toFixed(1) + '%';
      }}

      function findCompany(ticker) {{
        return appData.find((row) => row.ticker === ticker) || appData[0];
      }}

      function signalPill(row) {{
        if (row.signal === 'Likely up') {{
          return '<span class="pill"><i class="ti ti-trending-up" aria-hidden="true"></i> Likely up</span>';
        }}
        return '<span class="pill neutral"><i class="ti ti-minus" aria-hidden="true"></i> No clear up-signal</span>';
      }}

      function buildLine(points, key, minValue, maxValue) {{
        const usable = points
          .map((point, index) => {{
            const value = point[key];
            if (value === null || value === undefined) return null;
            const x = points.length === 1 ? 0 : (index / (points.length - 1)) * 500;
            const y = 82 - ((value - minValue) / (maxValue - minValue || 1)) * 70;
            return `${{x.toFixed(1)}},${{y.toFixed(1)}}`;
          }})
          .filter(Boolean);
        return usable.join(' ');
      }}

      function renderTrendChart(ticker) {{
        const points = chartData[ticker] || [];
        const chart = document.getElementById('trend-chart');

        if (points.length === 0) {{
          chart.innerHTML = '<div style="font-size:12px;color:var(--color-text-tertiary);padding-top:25px">No chart data available for this company.</div>';
          return;
        }}

        const values = [];
        points.forEach((point) => {{
          ['close', 'ma7', 'ma30', 'ma90'].forEach((key) => {{
            if (point[key] !== null && point[key] !== undefined) values.push(point[key]);
          }});
        }});

        const minValue = Math.min(...values);
        const maxValue = Math.max(...values);

        chart.innerHTML = `
          <svg viewBox="0 0 500 92" preserveAspectRatio="none" role="img" aria-label="Last 90 days of price and moving averages">
            <line x1="0" y1="82" x2="500" y2="82" stroke="rgba(255,255,255,.12)" stroke-width="1" />
            <polyline points="${{buildLine(points, 'close', minValue, maxValue)}}" fill="none" stroke="#F3F1EA" stroke-width="2" />
            <polyline points="${{buildLine(points, 'ma7', minValue, maxValue)}}" fill="none" stroke="#8EEA76" stroke-width="1.6" />
            <polyline points="${{buildLine(points, 'ma30', minValue, maxValue)}}" fill="none" stroke="#BFD8FF" stroke-width="1.6" />
            <polyline points="${{buildLine(points, 'ma90', minValue, maxValue)}}" fill="none" stroke="#F4B860" stroke-width="1.6" />
          </svg>`;
      }}

      function companyLines(row) {{
        return `
          <div style="margin-top:10px;font-size:12px;line-height:2;color:var(--color-text-secondary)">
            <div style="display:flex;justify-content:space-between">Probability <span>${{percent(row.probability)}}</span></div>
            <div style="display:flex;justify-content:space-between">Sentiment <span>${{row.sentiment.toFixed(3)}}</span></div>
            <div style="display:flex;justify-content:space-between">7-day return <span>${{percent(row.return7)}}</span></div>
            <div style="display:flex;justify-content:space-between">30-day volatility <span>${{percent(row.volatility30)}}</span></div>
          </div>`;
      }}

      function briefingSummary(row, risk) {{
        const aiSummary = summaryData.briefing[row.ticker];
        if (aiSummary) {{
          return `${{aiSummary}} Risk preference selected: ${{risk}}.`;
        }}
        const riskNote = risk === 'Low'
          ? 'Because the risk preference is low, volatility should be weighted heavily before acting on this signal.'
          : risk === 'High'
            ? 'Because the risk preference is high, the user may be more willing to review stocks with stronger movement and higher volatility.'
            : 'For a medium risk preference, the signal should be compared with volatility, sentiment, and recent return.';
        return `${{row.ticker}} shows a ${{row.signal.toLowerCase()}} direction for the next 7 trading days, with an upward-move probability of ${{percent(row.probability)}}. Recent return is ${{percent(row.return7)}}, 30-day volatility is ${{percent(row.volatility30)}}, and sentiment is ${{row.sentiment.toFixed(3)}} across ${{row.articles}} articles. ${{riskNote}} This is decision support, not financial advice.`;
      }}

      function updateBriefing() {{
        const row = findCompany(document.getElementById('briefing-company').value);
        document.getElementById('briefing-ticker').textContent = row.ticker;
        document.getElementById('briefing-pill').innerHTML = signalPill(row);
        document.getElementById('briefing-probability').textContent = percent(row.probability);
        document.getElementById('briefing-probability-bar').style.width = `${{row.probability * 100}}%`;
        document.getElementById('briefing-close').textContent = money(row.latestClose);
        document.getElementById('briefing-return').textContent = percent(row.return7);
        document.getElementById('briefing-volatility').textContent = percent(row.volatility30);
        document.getElementById('briefing-sentiment').textContent = row.sentiment.toFixed(3);
        document.getElementById('briefing-articles').textContent = row.articles;
        document.getElementById('briefing-summary').textContent = briefingSummary(row, selectedRisks.briefing);
        renderTrendChart(row.ticker);
      }}

      function updateComparison() {{
        const rowA = findCompany(document.getElementById('compare-a-select').value);
        const rowB = findCompany(document.getElementById('compare-b-select').value);
        document.getElementById('compare-a-title').textContent = rowA.ticker;
        document.getElementById('compare-b-title').textContent = rowB.ticker;
        document.getElementById('compare-a-pill').innerHTML = signalPill(rowA);
        document.getElementById('compare-b-pill').innerHTML = signalPill(rowB);
        document.getElementById('compare-a-lines').innerHTML = companyLines(rowA);
        document.getElementById('compare-b-lines').innerHTML = companyLines(rowB);
        const leader = rowA.probability >= rowB.probability ? rowA : rowB;
        const other = leader === rowA ? rowB : rowA;
        const compareKey = `${{rowA.ticker}}|${{rowB.ticker}}`;
        const aiSummary = summaryData.comparison[compareKey];
        document.getElementById('compare-summary').textContent = aiSummary
          ? `${{aiSummary}} Risk preference selected: ${{selectedRisks.compare}}.`
          : `${{leader.ticker}} has the stronger 7-day upward signal at ${{percent(leader.probability)}} compared with ${{other.ticker}} at ${{percent(other.probability)}}. The comparison should also consider volatility, recent return, and sentiment. Risk preference selected: ${{selectedRisks.compare}}. Decision support only, not financial advice.`;
      }}

      function updateWatchlist() {{
        const threshold = Number(document.getElementById('watch-threshold').value) / 100;
        document.getElementById('watch-threshold-label').textContent = `threshold ${{Math.round(threshold * 100)}}%`;
        let rows = appData
          .filter((row) => row.probability >= threshold)
          .sort((a, b) => b.probability - a.probability);
        if (rows.length === 0) rows = [...appData].sort((a, b) => b.probability - a.probability);
        const tableRows = rows.map((row, index) => {{
          const isTop = index === 0;
          const isLast = index === rows.length - 1;
          const rowStyle = isTop
            ? 'background:var(--color-background-success);border-left:2px solid var(--color-text-success)'
            : isLast
              ? 'border-left:2px solid var(--color-border-secondary)'
              : '';
          const suffix = isTop ? ' - top' : isLast ? ' - low in filter' : '';
          return `<tr style="${{rowStyle}}">
            <td style="padding:7px 4px">${{index + 1}}</td>
            <td style="padding:7px 4px;font-weight:500">${{row.ticker}}</td>
            <td style="padding:7px 4px">${{row.signal}}<span style="font-size:10px">${{suffix}}</span></td>
            <td style="padding:7px 4px">${{percent(row.probability)}}</td>
            <td style="padding:7px 4px">${{row.sentiment.toFixed(3)}}</td>
            <td style="padding:7px 4px">${{row.articles}}</td>
          </tr>`;
        }}).join('');
        const strongest = rows[0];
        const weakest = rows[rows.length - 1];
        document.getElementById('watch-table-body').innerHTML = tableRows;
        document.getElementById('watch-highest').textContent = `${{strongest.ticker}} - ${{percent(strongest.probability)}}`;
        document.getElementById('watch-lowest').textContent = `${{weakest.ticker}} - ${{percent(weakest.probability)}}`;
        document.getElementById('watch-summary').textContent = summaryData.watchlist
          ? `${{summaryData.watchlist}} Risk preference selected: ${{selectedRisks.watch}}.`
          : `Across the selected companies, ${{strongest.ticker}} ranks highest for a 7-day upward move and ${{weakest.ticker}} ranks lowest within the current filter. Sentiment, article count, and volatility help explain why the ranking differs by company. Risk preference selected: ${{selectedRisks.watch}}. Decision support only, not financial advice.`;
      }}

      function showScreen(screenId) {{
        pages.forEach((page) => {{
          page.classList.toggle('active', page.id === screenId);
        }});
        navButtons.forEach((button) => {{
          button.classList.toggle('on', button.dataset.screen === screenId);
        }});
      }}

      navButtons.forEach((button) => {{
        button.addEventListener('click', () => showScreen(button.dataset.screen));
      }});

      document.querySelectorAll('[data-risk-group]').forEach((group) => {{
        group.querySelectorAll('button').forEach((button) => {{
          button.addEventListener('click', () => {{
            group.querySelectorAll('button').forEach((item) => item.classList.remove('on'));
            button.classList.add('on');
            selectedRisks[group.dataset.riskGroup] = button.dataset.risk;
            updateBriefing();
            updateComparison();
            updateWatchlist();
          }});
        }});
      }});

      document.getElementById('briefing-company').addEventListener('change', updateBriefing);
      document.getElementById('briefing-generate').addEventListener('click', updateBriefing);
      document.getElementById('compare-a-select').addEventListener('change', updateComparison);
      document.getElementById('compare-b-select').addEventListener('change', updateComparison);
      document.getElementById('compare-button').addEventListener('click', updateComparison);
      document.getElementById('watch-threshold').addEventListener('input', updateWatchlist);

      updateBriefing();
      updateComparison();
      updateWatchlist();
    </script>
    </body>
    </html>
    """
    return html


def probability_color(probability):
    return COLOR_TOKENS["strong"] if probability >= 0.50 else COLOR_TOKENS["caution"]


def return_color(value):
    if value < 0:
        return COLOR_TOKENS["negative"]
    return COLOR_TOKENS["strong"]


def signal_display(probability):
    return "Stronger" if probability >= 0.50 else "Caution"


def render_kpi_card(label, value, note, color_key="strong"):
    color_map = {
        "strong": ("var(--color-background-success)", "var(--color-strong-text)"),
        "caution": ("var(--color-background-warning)", "var(--color-caution-text)"),
        "negative": ("rgba(226, 75, 74, .16)", "var(--color-negative)"),
        "info": ("var(--color-background-info)", "var(--color-info-text)"),
        "muted": ("var(--color-panel)", "var(--color-text-secondary)"),
    }
    note_background, note_color = color_map[color_key]
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-note" style="background:{note_background}; color:{note_color};">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_intro(title, copy, source_text=None):
    source_html = f'<div class="source-pill">{source_text}</div>' if source_text else ""
    st.markdown(
        f"""
        <div class="page-intro">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-copy">{copy}</div>
            </div>
            {source_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(title, copy=None):
    copy_html = f'<div class="section-copy">{copy}</div>' if copy else ""
    st.markdown(
        f"""
        <div class="section-label">{title}</div>
        {copy_html}
        """,
        unsafe_allow_html=True,
    )


def style_negative_returns(table):
    def color_return(value):
        if value < 0:
            return f"color: {COLOR_TOKENS['negative']}; font-weight: 600;"
        return f"color: {COLOR_TOKENS['strong']}; font-weight: 600;"

    styles = pd.DataFrame("", index=table.index, columns=table.columns)
    if "7-Day Return" in table.columns:
        styles["7-Day Return"] = table["7-Day Return"].map(color_return)
    if "30-Day Volatility" in table.columns:
        high_volatility = table["30-Day Volatility"] == table["30-Day Volatility"].max()
        styles.loc[high_volatility, "30-Day Volatility"] = (
            f"color: {COLOR_TOKENS['negative']}; font-weight: 600;"
        )
    return table.style.apply(lambda _: styles, axis=None)


def signal_column_config():
    return {
        "Ticker": st.column_config.TextColumn("Ticker", width="small"),
        "Signal": st.column_config.TextColumn("Signal", width="medium"),
        "Probability": st.column_config.NumberColumn("Prob.", format="%.1f%%"),
        "Latest Close": st.column_config.NumberColumn("Close", format="$%.2f"),
        "7-Day Return": st.column_config.NumberColumn("7d", format="%.1f%%"),
        "30-Day Volatility": st.column_config.NumberColumn("Vol.", format="%.1f%%"),
        "Sentiment": st.column_config.ProgressColumn(
            "Sentiment",
            min_value=0,
            max_value=0.50,
            format="%.3f",
        ),
        "Articles": st.column_config.NumberColumn("Art.", format="%d"),
        "Review Score": st.column_config.ProgressColumn(
            "Ranker score",
            min_value=0,
            max_value=1,
            format="%.3f",
        ),
    }


def display_table(table, height=250):
    table_to_show = table.copy()
    if "Probability" in table_to_show.columns:
        table_to_show["Probability"] = table_to_show["Probability"] * 100
    if "7-Day Return" in table_to_show.columns:
        table_to_show["7-Day Return"] = table_to_show["7-Day Return"] * 100
    if "30-Day Volatility" in table_to_show.columns:
        table_to_show["30-Day Volatility"] = table_to_show["30-Day Volatility"] * 100

    st.dataframe(
        style_negative_returns(table_to_show),
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config=signal_column_config(),
    )


def make_sentiment_bar_chart(data, height=260):
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=COLOR_TOKENS["info"])
        .encode(
            x=alt.X("Ticker:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Sentiment:Q", title=None),
            tooltip=["Ticker", alt.Tooltip("Sentiment:Q", format=".3f")],
        )
        .properties(height=height)
        .configure_axis(
            labelColor=COLOR_TOKENS["muted"],
            titleColor=COLOR_TOKENS["muted"],
            gridColor="rgba(255,255,255,.10)",
        )
        .configure_view(strokeWidth=0)
    )


def make_probability_dot_plot(data, height=260):
    chart_df = data[["Ticker", "Probability"]].copy()
    chart_df["Probability Percent"] = chart_df["Probability"] * 100
    chart_df["Signal Group"] = chart_df["Probability"].apply(
        lambda value: "Above 50%" if value >= 0.50 else "Below 50%"
    )
    min_probability = max(0, chart_df["Probability Percent"].min() - 2)
    max_probability = min(100, chart_df["Probability Percent"].max() + 2)

    points = (
        alt.Chart(chart_df)
        .mark_point(filled=True, size=115)
        .encode(
            y=alt.Y("Ticker:N", sort="-x", title=None),
            x=alt.X(
                "Probability Percent:Q",
                title="Upward-move probability",
                scale=alt.Scale(domain=[min_probability, max_probability]),
                axis=alt.Axis(format=".0f"),
            ),
            color=alt.Color(
                "Signal Group:N",
                scale=alt.Scale(
                    domain=["Above 50%", "Below 50%"],
                    range=[COLOR_TOKENS["strong"], COLOR_TOKENS["caution"]],
                ),
                legend=None,
            ),
            tooltip=["Ticker", alt.Tooltip("Probability Percent:Q", title="Probability", format=".1f")],
        )
    )
    labels = points.mark_text(
        align="left",
        baseline="middle",
        dx=9,
        color=COLOR_TOKENS["muted"],
        fontSize=12,
    ).encode(text=alt.Text("Probability Percent:Q", format=".1f"))

    rule = alt.Chart(pd.DataFrame({"x": [50]})).mark_rule(
        color=COLOR_TOKENS["subtle"],
        strokeDash=[5, 5],
    ).encode(x="x:Q")

    return (
        (rule + points + labels)
        .properties(height=height)
        .configure_axis(
            labelColor=COLOR_TOKENS["muted"],
            titleColor=COLOR_TOKENS["muted"],
            gridColor="rgba(255,255,255,.10)",
        )
        .configure_view(strokeWidth=0)
    )


def make_watchlist_score_chart(data, height=280):
    chart_df = data[["Ticker", "Review Score", "Passes Filter"]].copy()
    chart_df["Filter Status"] = chart_df["Passes Filter"].map({True: "Pass", False: "Below filter"})
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Ticker:N", sort="-y", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Review Score:Q", title="Ranker score"),
            color=alt.Color(
                "Filter Status:N",
                scale=alt.Scale(
                    domain=["Pass", "Below filter"],
                    range=[COLOR_TOKENS["strong"], COLOR_TOKENS["subtle"]],
                ),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            opacity=alt.condition("datum['Passes Filter']", alt.value(1), alt.value(0.42)),
            tooltip=["Ticker", alt.Tooltip("Review Score:Q", format=".3f"), "Filter Status"],
        )
        .properties(height=height)
        .configure_axis(
            labelColor=COLOR_TOKENS["muted"],
            titleColor=COLOR_TOKENS["muted"],
            gridColor="rgba(255,255,255,.10)",
        )
        .configure_legend(labelColor=COLOR_TOKENS["muted"])
        .configure_view(strokeWidth=0)
    )


def make_return_bar_chart(data, height=180):
    return (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=COLOR_TOKENS["negative"])
        .encode(
            x=alt.X("Company:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("7-Day Return:Q", title="7-day return", axis=alt.Axis(format=".1%")),
            tooltip=["Company", alt.Tooltip("7-Day Return:Q", format=".1%")],
        )
        .properties(height=height)
        .configure_axis(
            labelColor=COLOR_TOKENS["muted"],
            titleColor=COLOR_TOKENS["muted"],
            gridColor="rgba(255,255,255,.10)",
        )
        .configure_view(strokeWidth=0)
    )


def make_indexed_price_chart(data, height=300):
    indexed = data.copy()
    indexed = indexed.sort_values(["Company", "date"])
    indexed["Indexed Close"] = indexed.groupby("Company")["Close"].transform(
        lambda values: values / values.iloc[0] * 100
    )

    base = alt.Chart(indexed).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("Indexed Close:Q", title="Indexed close"),
        color=alt.Color(
            "Company:N",
            scale=alt.Scale(range=[COLOR_TOKENS["info"], COLOR_TOKENS["strong"]]),
            legend=alt.Legend(orient="bottom", title=None),
        ),
        strokeDash=alt.StrokeDash("Company:N", legend=None),
        tooltip=[
            "Company",
            alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("Indexed Close:Q", format=".1f"),
        ],
    )
    line = base.mark_line(strokeWidth=2)
    points = base.mark_point(filled=True, size=35, opacity=0.35)
    reference = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(
        color=COLOR_TOKENS["subtle"],
        strokeDash=[5, 5],
    ).encode(y="y:Q")

    return (
        (reference + line + points)
        .properties(height=height)
        .configure_axis(
            labelColor=COLOR_TOKENS["muted"],
            titleColor=COLOR_TOKENS["muted"],
            gridColor="rgba(255,255,255,.10)",
        )
        .configure_legend(labelColor=COLOR_TOKENS["muted"])
        .configure_view(strokeWidth=0)
    )


def render_metric_pair(metric_name, company_a, value_a, company_b, value_b, higher_is_better=True, formatter=None):
    if formatter is None:
        formatter = lambda value: f"{value:.3f}"
    if higher_is_better:
        a_wins = value_a >= value_b
    else:
        a_wins = value_a <= value_b
    b_wins = not a_wins
    max_value = max(abs(value_a), abs(value_b), 0.0001)
    width_a = min(100, abs(value_a) / max_value * 100)
    width_b = min(100, abs(value_b) / max_value * 100)
    color = COLOR_TOKENS["negative"] if metric_name == "7-day return" else COLOR_TOKENS["info"]
    if metric_name == "Probability":
        color = COLOR_TOKENS["strong"]
    if metric_name == "Volatility":
        color = COLOR_TOKENS["caution"]

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-name">{metric_name}</div>
            <div class="metric-cell">
                <div class="metric-value {'winner' if a_wins else ''}">{'✓ ' if a_wins else ''}{company_a}: {formatter(value_a)}</div>
                <div class="track"><div class="fill" style="width:{width_a:.1f}%;background:{color};"></div></div>
            </div>
            <div class="metric-cell">
                <div class="metric-value {'winner' if b_wins else ''}">{'✓ ' if b_wins else ''}{company_b}: {formatter(value_b)}</div>
                <div class="track"><div class="fill" style="width:{width_b:.1f}%;background:{color};"></div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_signal_pill(probability):
    label = "Stronger" if probability >= 0.50 else "Caution"
    class_name = "pill-strong" if probability >= 0.50 else "pill-caution"
    return f'<span class="{class_name}">{label}</span>'


def render_secondary_ai_action(title, copy, button_label, key, callback):
    st.markdown(
        f"""
        <div class="ai-panel-title">{title}</div>
        <div class="ai-panel-copy">{copy}</div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(button_label, key=key, type="secondary"):
        st.info(callback())


def make_legacy_line_chart(data, height=300):
    # Kept only for any older local screenshots that may still call the previous helper.
    return make_indexed_price_chart(data.rename(columns={"close": "Close"}), height)


def make_grouped_bar_chart(data, height=260):
    # Kept only for compatibility with older local page code while the tabs are redesigned.
    return make_watchlist_score_chart(
        data.rename(columns={"Value": "Review Score"}).assign(**{"Passes Filter": True}),
        height,
    )


def make_line_chart(data, height=300):
    return make_indexed_price_chart(data, height)


def make_bar_chart(data, x_column, y_column, color=None, height=260):
    chart_df = data[[x_column, y_column]].copy()
    chart_df = chart_df.rename(columns={x_column: "Ticker", y_column: "Value"})
    return (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color or COLOR_TOKENS["info"])
        .encode(
            x=alt.X("Ticker:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title=None),
            tooltip=["Ticker", alt.Tooltip("Value:Q", format=".3f")],
        )
        .properties(height=height)
        .configure_axis(
            labelColor=COLOR_TOKENS["muted"],
            titleColor=COLOR_TOKENS["muted"],
            gridColor="rgba(255,255,255,.10)",
        )
        .configure_view(strokeWidth=0)
    )


def get_overview_summary(prediction_table):
    ranked = prediction_table.sort_values("Probability", ascending=False)
    top = ranked.iloc[0]
    bottom = ranked.iloc[-1]
    avg_probability = ranked["Probability"].mean()
    fallback = (
        f"{top['Ticker']} has the highest upward-move probability at {format_percent(top['Probability'])}, "
        f"while {bottom['Ticker']} has the lowest at {format_percent(bottom['Probability'])}. "
        f"The average probability across the AI company group is {format_percent(avg_probability)}. "
        "This summary is decision support only, not financial advice."
    )
    prompt = (
        "Write a short market overview for a stock signal dashboard. "
        "Explain which AI-related company has the strongest signal, which has the weakest signal, "
        "and mention probability, recent return, volatility, and sentiment in simple language. "
        f"Data: {ranked.to_dict(orient='records')}. Do not give financial advice."
    )
    return create_openai_summary(prompt, fallback)


def get_comparison_summary(company_a, company_b, risk_preference):
    leader = company_a if company_a["Probability"] >= company_b["Probability"] else company_b
    other = company_b if leader is company_a else company_a
    fallback = (
        f"{leader['Ticker']} has the stronger short-term signal at {format_percent(leader['Probability'])}, "
        f"compared with {other['Ticker']} at {format_percent(other['Probability'])}. "
        f"The comparison should also consider volatility, recent return, and sentiment. "
        f"Risk preference selected: {risk_preference}. Decision support only, not financial advice."
    )
    prompt = (
        "Write a short comparison summary for two AI-related stocks. "
        f"Company A: {company_a.to_dict()}. Company B: {company_b.to_dict()}. "
        f"Risk preference: {risk_preference}. Explain which company has the stronger short-term signal. "
        "Do not give financial advice."
    )
    return create_openai_summary(prompt, fallback)


def get_watchlist_summary(watchlist_df, risk_preference):
    top = watchlist_df.iloc[0]
    bottom = watchlist_df.iloc[-1]
    fallback = (
        f"{top['Ticker']} ranks highest in the watchlist based on the selected risk preference. "
        f"{bottom['Ticker']} ranks lowest within the current filter. "
        "The ranking uses model probability, volatility, sentiment, and recent return. "
        "This is decision support only, not financial advice."
    )
    prompt = (
        "Write a short watchlist summary for a decision-support stock dashboard. "
        f"Risk preference: {risk_preference}. Watchlist data: {watchlist_df.to_dict(orient='records')}. "
        "Explain the highest-ranked and lowest-ranked companies in simple language. "
        "Do not give financial advice."
    )
    return create_openai_summary(prompt, fallback)


def make_watchlist_table(prediction_table, risk_preference, probability_threshold):
    watchlist_df = prediction_table.copy()
    watchlist_df = watchlist_df[watchlist_df["Probability"] >= probability_threshold].copy()

    if watchlist_df.empty:
        watchlist_df = prediction_table.copy()

    if risk_preference == "Low":
        watchlist_df["Review Score"] = (
            watchlist_df["Probability"]
            - watchlist_df["30-Day Volatility"]
            + (watchlist_df["Sentiment"] * 0.05)
        )
    elif risk_preference == "High":
        watchlist_df["Review Score"] = (
            watchlist_df["Probability"]
            + (watchlist_df["7-Day Return"] * 0.10)
            + (watchlist_df["Sentiment"] * 0.05)
        )
    else:
        watchlist_df["Review Score"] = (
            watchlist_df["Probability"]
            + (watchlist_df["Sentiment"] * 0.05)
            - (watchlist_df["30-Day Volatility"] * 0.05)
        )

    return watchlist_df.sort_values("Review Score", ascending=False)


def render_streamlit_dashboard(prediction_table, chart_history_df, data_status):
    st.markdown(
        """
        <div class="app-hero">
            <div class="app-title">AI Investment Signal Assistant</div>
            <div class="app-subtitle">
                Decision-support dashboard for comparing AI-related companies. Signals are model estimates, not financial advice.
            </div>
            <div class="disclaimer-pill">Live stock data + news sentiment + Random Forest model + optional AI summaries</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["Overview Dashboard", "Compare Companies", "Watchlist Review"])
    ranked = prediction_table.sort_values("Probability", ascending=False).copy()

    with tabs[0]:
        render_page_intro(
            "Overview Dashboard",
            "Scan the full AI company group first. This page highlights the strongest signal, weakest signal, sentiment leader, and highest-risk name before the user drills into details.",
            data_status,
        )

        top_company = ranked.iloc[0]
        weakest_company = ranked.iloc[-1]
        highest_sentiment = ranked.sort_values("Sentiment", ascending=False).iloc[0]
        highest_volatility = ranked.sort_values("30-Day Volatility", ascending=False).iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Strongest signal", top_company["Ticker"], format_percent(top_company["Probability"]))
        with col2:
            render_kpi_card("Weakest signal", weakest_company["Ticker"], format_percent(weakest_company["Probability"]))
        with col3:
            render_kpi_card("Best news sentiment", highest_sentiment["Ticker"], f"{highest_sentiment['Sentiment']:.3f}")
        with col4:
            render_kpi_card("Highest volatility", highest_volatility["Ticker"], format_percent(highest_volatility["30-Day Volatility"]))

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            render_section_title("Upward-Move Probability", "Higher values mean stronger conditions for a 7-trading-day upward move.")
            probability_chart_df = ranked[["Ticker", "Probability"]].copy()
            st.altair_chart(
                make_bar_chart(probability_chart_df, "Ticker", "Probability", "#7EC4F7", 260),
                use_container_width=True,
            )
        with chart_col2:
            render_section_title("News Sentiment", "Alpha Vantage sentiment compares recent news tone by company.")
            sentiment_chart_df = ranked[["Ticker", "Sentiment"]].copy()
            st.altair_chart(
                make_bar_chart(sentiment_chart_df, "Ticker", "Sentiment", "#8EEA76", 260),
                use_container_width=True,
            )

        render_section_title("Company Signal Table", "Detailed values used to compare the six AI-related companies.")
        st.dataframe(
            prepare_display_table(
                ranked[
                    [
                        "Ticker",
                        "Signal",
                        "Probability",
                        "Latest Close",
                        "7-Day Return",
                        "30-Day Volatility",
                        "Sentiment",
                        "Articles",
                    ]
                ]
            ),
            use_container_width=True,
            hide_index=True,
            height=250,
        )

        with st.expander("What this overview shows"):
            st.write(
                "The overview combines model probability, recent return, volatility, trading activity, "
                "and Alpha Vantage sentiment so the user can compare all six companies before choosing one."
            )

        st.markdown(
            """
            <div class="ai-panel-title">AI overview summary</div>
            <div class="ai-panel-copy">Generate a short explanation only after reviewing the dashboard. This keeps the app intentional and avoids unnecessary API calls.</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate overview summary", use_container_width=True, key="overview_summary_button"):
            st.info(get_overview_summary(ranked))

    with tabs[1]:
        render_page_intro(
            "Compare Companies",
            "Choose two companies to compare the short-term outlook, recent performance, risk, price trend, and news sentiment side by side.",
            "User-selected comparison",
        )

        compare_col1, compare_col2, compare_col3 = st.columns([1, 1, 1.25])
        ticker_a = compare_col1.selectbox("Company A", ranked["Ticker"], index=0, key="compare_company_a")
        ticker_b = compare_col2.selectbox("Company B", ranked["Ticker"], index=min(1, len(ranked) - 1), key="compare_company_b")
        risk_preference = compare_col3.radio(
            "Risk preference",
            ["Low", "Medium", "High"],
            horizontal=True,
            index=1,
            key="compare_risk_preference",
        )

        company_a = get_company_row(ranked, ticker_a)
        company_b = get_company_row(ranked, ticker_b)
        leader = company_a if company_a["Probability"] >= company_b["Probability"] else company_b

        st.markdown(
            f"""
            <div class="result-banner">Stronger short-term signal: {leader['Ticker']} at {format_percent(leader['Probability'])}</div>
            """,
            unsafe_allow_html=True,
        )

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            render_section_title(company_a["Ticker"], "Company A")
            a1, a2 = st.columns(2)
            a1.metric("Probability", format_percent(company_a["Probability"]))
            a2.metric("7-day return", format_percent(company_a["7-Day Return"]))
            a3, a4 = st.columns(2)
            a3.metric("Volatility", format_percent(company_a["30-Day Volatility"]))
            a4.metric("Sentiment", f"{company_a['Sentiment']:.3f}")
        with metric_col2:
            render_section_title(company_b["Ticker"], "Company B")
            b1, b2 = st.columns(2)
            b1.metric("Probability", format_percent(company_b["Probability"]))
            b2.metric("7-day return", format_percent(company_b["7-Day Return"]))
            b3, b4 = st.columns(2)
            b3.metric("Volatility", format_percent(company_b["30-Day Volatility"]))
            b4.metric("Sentiment", f"{company_b['Sentiment']:.3f}")

        comparison_chart = pd.DataFrame(
            [
                {
                    "Company": company_a["Ticker"],
                    "Probability": company_a["Probability"],
                    "Volatility": company_a["30-Day Volatility"],
                    "Sentiment": company_a["Sentiment"],
                },
                {
                    "Company": company_b["Ticker"],
                    "Probability": company_b["Probability"],
                    "Volatility": company_b["30-Day Volatility"],
                    "Sentiment": company_b["Sentiment"],
                },
            ]
        )
        comparison_chart_long = comparison_chart.melt(
            id_vars="Company",
            var_name="Metric",
            value_name="Value",
        )
        return_chart = pd.DataFrame(
            {
                "7-Day Return": [
                    company_a["7-Day Return"],
                    company_b["7-Day Return"],
                ]
            },
            index=[company_a["Ticker"], company_b["Ticker"]],
        )
        render_section_title("Side-by-Side Metrics", "Compares probability, volatility, and sentiment by company.")
        st.altair_chart(make_grouped_bar_chart(comparison_chart_long, 250), use_container_width=True)
        render_section_title("Recent Return", "Shows recent 7-day price movement separately because returns can be negative.")
        return_chart_df = return_chart.reset_index().rename(columns={"index": "Company"})
        st.altair_chart(
            make_bar_chart(return_chart_df, "Company", "7-Day Return", "#F4B860", 180),
            use_container_width=True,
        )

        history_a = chart_history_df[chart_history_df["ticker"] == company_a["Ticker"]].sort_values("date").tail(90)
        history_b = chart_history_df[chart_history_df["ticker"] == company_b["Ticker"]].sort_values("date").tail(90)
        price_chart = pd.concat(
            [
                history_a[["date", "close"]].assign(Company=company_a["Ticker"]),
                history_b[["date", "close"]].assign(Company=company_b["Ticker"]),
            ],
            ignore_index=True,
        ).rename(columns={"close": "Close"})
        render_section_title("Closing Price Trend - Last 90 Days", "Shows recent price movement for both selected companies.")
        st.altair_chart(make_line_chart(price_chart, 300), use_container_width=True)

        st.markdown(
            """
            <div class="ai-panel-title">AI comparison summary</div>
            <div class="ai-panel-copy">Generate a plain-English comparison after choosing the two companies and risk preference.</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate comparison summary", use_container_width=True, key="comparison_summary_button"):
            st.info(get_comparison_summary(company_a, company_b, risk_preference))

    with tabs[2]:
        render_page_intro(
            "Watchlist Review",
            "Filter the AI company group and rank the names using probability, volatility, sentiment, and recent performance.",
            "Risk-adjusted review",
        )

        filter_col1, filter_col2 = st.columns([1, 2])
        watch_risk = filter_col1.radio(
            "Risk preference",
            ["Low", "Medium", "High"],
            horizontal=True,
            index=1,
            key="watchlist_risk_preference",
        )
        threshold = filter_col2.slider(
            "Minimum upward-move probability",
            0,
            100,
            40,
            5,
            key="watchlist_probability_threshold",
        ) / 100

        watchlist_df = make_watchlist_table(ranked, watch_risk, threshold)
        strongest = watchlist_df.iloc[0]
        lowest = watchlist_df.iloc[-1]
        highest_risk = watchlist_df.sort_values("30-Day Volatility", ascending=False).iloc[0]
        best_sentiment = watchlist_df.sort_values("Sentiment", ascending=False).iloc[0]

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_kpi_card("Top watchlist name", strongest["Ticker"], format_percent(strongest["Probability"]))
        with col2:
            render_kpi_card("Lowest in filter", lowest["Ticker"], format_percent(lowest["Probability"]))
        with col3:
            render_kpi_card("Highest volatility", highest_risk["Ticker"], format_percent(highest_risk["30-Day Volatility"]))
        with col4:
            render_kpi_card("Best sentiment", best_sentiment["Ticker"], f"{best_sentiment['Sentiment']:.3f}")

        render_section_title("Ranked Watchlist", "Review score changes based on the selected risk preference.")
        st.dataframe(
            prepare_display_table(
                watchlist_df[
                    [
                        "Ticker",
                        "Signal",
                        "Probability",
                        "7-Day Return",
                        "30-Day Volatility",
                        "Sentiment",
                        "Articles",
                        "Review Score",
                    ]
                ].assign(**{"Review Score": watchlist_df["Review Score"].map(lambda value: f"{value:.3f}")})
            ),
            use_container_width=True,
            hide_index=True,
            height=250,
        )

        render_section_title("Watchlist Ranking", "Higher review score means the company is more relevant for the selected filter and risk setting.")
        watchlist_chart_df = watchlist_df[["Ticker", "Review Score"]].copy()
        st.altair_chart(
            make_bar_chart(watchlist_chart_df, "Ticker", "Review Score", "#7EC4F7", 280),
            use_container_width=True,
        )

        with st.expander("How the watchlist ranking works"):
            st.write(
                "The model probability is still the main signal. The risk preference adjusts the review score: "
                "low risk gives more weight to lower volatility, medium risk stays closest to the model signal, "
                "and high risk allows more recent movement."
            )

        st.markdown(
            """
            <div class="ai-panel-title">AI watchlist summary</div>
            <div class="ai-panel-copy">Generate a summary after setting the risk preference and probability filter.</div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Generate watchlist summary", use_container_width=True, key="watchlist_summary_button"):
            st.info(get_watchlist_summary(watchlist_df, watch_risk))

    st.markdown(
        f"""
        <div class="foot">Data source status: {data_status}. Decision-support tool - not financial advice.</div>
        """,
        unsafe_allow_html=True,
    )


training_df, latest_df, model_ready_df = load_data()
model, balanced_accuracy, train_rows, test_rows = train_model(training_df)
tickers = sorted(model_ready_df["ticker"].unique())
live_sentiment_df, sentiment_status = get_live_sentiment(tickers)
app_latest_df, app_history_df, data_status = get_app_market_data(latest_df, model_ready_df, live_sentiment_df)
prediction_table = create_prediction_table(model, app_latest_df)

render_streamlit_dashboard(prediction_table, app_history_df, data_status)
