# MIS 433 AI Investment Signals

This project uses stock price trends, trading volume, volatility, and recent news sentiment to create short-term investment signals for AI-related companies.

The companies used in this project are:

- NVDA
- MSFT
- GOOGL
- AMZN
- AMD
- AVGO

The goal is not to predict the exact future stock price. The goal is to predict whether a stock may move up over the next 7 trading days.

## Data Sources

- `yfinance`: historical stock price data from Yahoo Finance
- Alpha Vantage API: recent news sentiment data
- OpenAI API: plain-English summaries for the final application feature

## AI Investment Signals Overview

```text
notebooks/AI_Investment_Signals.ipynb
```

This notebook is organized as a full modeling process from top to bottom:

1. Set up folders, imports, and project paths
2. Load stock data
3. Clean and format the stock data
4. Create basic charts and summary statistics
5. Add calculated features
6. Add Alpha Vantage sentiment data
7. Create the target variable
8. Create exploratory data analysis charts
9. Train and test Logistic Regression, Decision Tree, and Random Forest models
10. Compare model results
11. Create current prediction outputs
12. Run the notebook application features

The notebook uses saved CSV files by default so it runs quickly and does not repeatedly call APIs. Fresh data can still be pulled by changing the refresh options at the top of the notebook.

## Modeling Summary

The model predicts stock direction, not exact price.

The current best model setup is:

```text
Model: Random Forest
Prediction window: 7 trading days
Target: stock rises more than 1%
Best metric used: balanced accuracy
```

The notebook compares:

- Logistic Regression
- Decision Tree
- Random Forest

Balanced accuracy is used to evaluate both target classes.

## Current Results

The best current model result is:

```text
Model: Random Forest
Settings: 100 trees, max depth 6
Prediction threshold: 0.50
Accuracy: 61.8%
Balanced accuracy: 61.7%
```

The latest prediction output gives positive signals for AMD and MSFT. It gives caution signals for AMZN, GOOGL, NVDA, and AVGO.

These results compare companies and produce a model signal. They are not guaranteed stock predictions.

## Main Data Files

### `data/processed/model_ready_stock_data.csv`

Full processed dataset created by the notebook.

This file includes:

- cleaned stock price data
- calculated stock features
- sentiment scores
- future return columns
- target columns

This is the full dataset used to build the training and prediction files.

### `data/processed/stock_prices_clean.csv`

Clean stock price dataset used near the start of the notebook.

This file lets the notebook start from saved stock data instead of pulling fresh Yahoo Finance data every time.

### `data/processed/training_ready_stock_data.csv`

Rows used for model training and testing.

These rows already have a known future result. For example, the 7-day target can be calculated because the notebook can look 7 trading days ahead in the historical data.

### `data/processed/latest_prediction_rows.csv`

Newest row for each ticker.

These rows are used for current predictions. They are not used for training because the future 7-day result is not known yet.

### `data/external/daily_sentiment_scores.csv`

Daily Alpha Vantage sentiment scores by ticker.

The sentiment data is merged into the stock dataset so the model can use news sentiment as one of the inputs.

## Output Files

### `outputs/model_results/model_comparison.csv`

Model testing results.

This file shows the model type, accuracy, and balanced accuracy.

### `outputs/model_results/latest_direction_predictions.csv`

Current prediction output.

This file gives one prediction for each company using the newest available row. The prediction is a direction signal, not a future price.

## Chart Outputs

Charts are saved in:

```text
outputs/charts/
```

Current charts include:

- `normalized_stock_performance.png`: compares stock growth from the same starting point
- `risk_return_scatter.png`: compares average daily return and volatility by company
- `latest_sentiment_by_company.png`: compares recent average sentiment by company
- `target_distribution_7d.png`: shows how many rows are in each target class
- `model_comparison_balanced_accuracy.png`: compares the strongest model tests by balanced accuracy

## Variable Guide

### Stock Price Variables

- `date`: trading date
- `ticker`: stock symbol
- `open`: price at the start of the trading day
- `high`: highest price during the trading day
- `low`: lowest price during the trading day
- `close`: price at the end of the trading day
- `adj_close`: closing price adjusted for stock splits or dividends
- `volume`: number of shares traded

### Calculated Features

- `daily_return`: percent change in close price from the previous trading day
- `return_7d`: percent change over the past 7 trading days
- `return_30d`: percent change over the past 30 trading days
- `ma_7d`: 7-day moving average of the close price
- `ma_30d`: 30-day moving average of the close price
- `ma_90d`: 90-day moving average of the close price
- `volatility_30d`: recent price movement based on daily returns over 30 trading days
- `volume_change`: percent change in trading volume from the previous trading day

### Sentiment Variables

- `avg_sentiment_score`: average news sentiment score for the company
- `avg_relevance_score`: average score for how closely the news relates to the company
- `article_count`: number of news articles used for the sentiment score

Positive sentiment scores usually mean more bullish news. Negative scores usually mean more bearish news. Scores close to zero are closer to neutral.

### Target Variables

- `future_close_7d`: close price 7 trading days later
- `future_return_7d`: percent return 7 trading days later
- `target_up_7d`: 1 if the stock moved up enough after 7 trading days, otherwise 0

The notebook uses `target_up_7d`. A value of 1 means the stock rose more than 1% over the next 7 trading days.

### Model Result Variables

- `model`: model type used for testing
- `accuracy`: percent of total predictions that were correct
- `balanced_accuracy`: accuracy adjusted for both classes

### Current Prediction Variables

- `date`: newest trading date used for the company
- `ticker`: stock symbol
- `close`: latest close price
- `return_7d`: recent 7-day return
- `volatility_30d`: recent 30-day volatility
- `avg_sentiment_score`: recent Alpha Vantage news sentiment score
- `article_count`: number of recent articles used for sentiment
- `predicted_up`: 1 means positive signal, 0 means caution
- `prediction_signal`: plain-English version of `predicted_up`
- `prediction_probability_up`: model estimate for the chance of an upward move

## Current Prediction Output

The current prediction file gives one row per company:

```text
outputs/model_results/latest_direction_predictions.csv
```

The output should be read as a model signal, not as financial advice.

## Notebook Application

The final section of the notebook is an application called `AI Investment Signal Assistant`.

The application uses:

- processed stock data
- model prediction results
- Alpha Vantage sentiment data
- OpenAI summaries when an OpenAI key is available

Application features:

- Single Company Briefing: selects one ticker and returns a signal summary
- Company Comparison: compares two tickers using prediction, sentiment, return, and volatility
- Watchlist Review: ranks the six tickers based on risk preference and signal filter

The notebook includes saved example outputs and Colab `# @param` controls for changing the app inputs.

## Repository Structure

```text
notebooks/              Jupyter notebooks for analysis and project demo
data/processed/         Cleaned and model-ready CSV files
data/external/          API and sentiment CSV files
outputs/charts/         Generated charts for notebook and slides
outputs/model_results/  Model metrics and results
```
