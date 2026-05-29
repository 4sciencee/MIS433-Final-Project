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
- Gemini or OpenAI API: plain-English summaries for the final application feature

## Main Notebook

The main notebook is:

```text
notebooks/AI_Investment_Signals.ipynb
```

The notebook is organized as a full modeling process from top to bottom:

1. Set up folders, imports, and project paths
2. Load stock data
3. Clean and format the stock data
4. Create basic charts and summary statistics
5. Add calculated features
6. Add Alpha Vantage sentiment data
7. Create the target variable
8. Compare important variables against the target
9. Train and test Logistic Regression, Decision Tree, and Random Forest models
10. Compare model results
11. Review feature importance
12. Create current prediction outputs

The notebook uses saved CSV files by default so it runs quickly and does not repeatedly call APIs. Fresh data can still be pulled by changing the refresh options at the top of the notebook.

## Modeling Summary

The model predicts stock direction, not exact price.

The current best model setup is:

```text
Model: Random Forest
Prediction window: 7 trading days
Target: stock rises more than 0.5%
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
Feature group: price and volume features with sentiment
Settings: 100 trees, max depth 3, minimum leaf size 5
Threshold: 0.45
Accuracy: 58.3%
Balanced accuracy: 54.6%
F1 score for upward moves: 67.7%
```

The latest prediction output gives positive signals for AMZN, MSFT, GOOGL, AVGO, and NVDA. It gives a caution signal for AMD.

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

This file shows the model type, model settings, accuracy, balanced accuracy, F1 score, training rows, testing rows, and split date.

### `outputs/model_results/test_set_predictions.csv`

Predictions made on the historical test set.

This file shows how the selected model performed on older data where the correct answer is already known.

### `outputs/model_results/model_feature_importance.csv`

Feature importance values from the selected Random Forest model.

This file shows which variables had more influence in the model.

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
- `target_feature_comparison.png`: compares important feature averages against the target
- `model_comparison_balanced_accuracy.png`: compares the strongest model tests by balanced accuracy
- `model_feature_importance.png`: shows which features mattered most in the model

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
- `return_3d`: percent change over the past 3 trading days
- `return_5d`: percent change over the past 5 trading days
- `return_7d`: percent change over the past 7 trading days
- `return_10d`: percent change over the past 10 trading days
- `return_14d`: percent change over the past 14 trading days
- `return_30d`: percent change over the past 30 trading days
- `ma_7d`: 7-day moving average of the close price
- `ma_14d`: 14-day moving average of the close price
- `ma_30d`: 30-day moving average of the close price
- `ma_90d`: 90-day moving average of the close price
- `volatility_7d`: recent price movement based on daily returns over 7 trading days
- `volatility_14d`: recent price movement based on daily returns over 14 trading days
- `volatility_30d`: recent price movement based on daily returns over 30 trading days
- `volume_change`: percent change in trading volume from the previous trading day
- `avg_volume_30d`: average trading volume over the past 30 trading days
- `volume_vs_avg_30d`: current volume compared to the 30-day average volume
- `close_vs_ma_30d`: close price compared to the 30-day moving average
- `ma_7d_vs_30d`: short-term moving average compared to the 30-day moving average

### Sentiment Variables

- `avg_sentiment_score`: average news sentiment score for the company
- `avg_relevance_score`: average score for how closely the news relates to the company
- `article_count`: number of news articles used for the sentiment score

Positive sentiment scores usually mean more bullish news. Negative scores usually mean more bearish news. Scores close to zero are closer to neutral.

### Target Variables

- `future_close_3d`: close price 3 trading days later
- `future_return_3d`: percent return 3 trading days later
- `target_up_3d`: 1 if the stock moved up enough after 3 trading days, otherwise 0
- `future_close_5d`: close price 5 trading days later
- `future_return_5d`: percent return 5 trading days later
- `target_up_5d`: 1 if the stock moved up enough after 5 trading days, otherwise 0
- `future_close_7d`: close price 7 trading days later
- `future_return_7d`: percent return 7 trading days later
- `target_up_7d`: 1 if the stock moved up enough after 7 trading days, otherwise 0
- `future_close_10d`: close price 10 trading days later
- `future_return_10d`: percent return 10 trading days later
- `target_up_10d`: 1 if the stock moved up enough after 10 trading days, otherwise 0

The current model uses `target_up_7d`. A value of 1 means the stock rose more than 0.5% over the next 7 trading days.

### Model Result Variables

- `model`: model type used for testing
- `settings`: model settings used in that test
- `threshold`: probability cutoff used for the up/down prediction
- `accuracy`: percent of total predictions that were correct
- `balanced_accuracy`: accuracy adjusted for both classes
- `f1_up`: score focused on predicting the upward-move class
- `train_rows`: number of rows used to train the model
- `test_rows`: number of rows used to test the model
- `split_date`: date used to separate training data from testing data

### Current Prediction Variables

- `prediction_window_days`: number of trading days being predicted
- `predicted_up`: 1 means positive signal, 0 means caution
- `prediction_signal`: plain-English version of `predicted_up`
- `target_cutoff_used`: minimum future return needed to count as an upward move
- `prediction_probability_up`: model estimate for the chance of an upward move

## Current Prediction Output

The current prediction file gives one row per company:

```text
outputs/model_results/latest_direction_predictions.csv
```

The output should be read as a model signal, not as financial advice.

## Repository Structure

```text
notebooks/              Jupyter notebooks for analysis and project demo
data/raw/               Original downloaded stock CSV files
data/processed/         Cleaned and model-ready CSV files
data/external/          API and sentiment CSV files
src/                    Reusable Python functions
outputs/charts/         Generated charts for notebook and slides
outputs/model_results/  Model metrics and results
outputs/screenshots/    App/dashboard screenshots for presentation
app/                    Optional Streamlit dashboard
```
