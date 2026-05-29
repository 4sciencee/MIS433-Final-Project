# MIS 433 AI Investment Signals

Final project workspace for exploring whether historical stock trends, volatility, trading volume, and recent news sentiment can help forecast short-term investment signals among leading AI infrastructure companies.

## Project Idea

This project focuses on AI-related public companies:

- NVDA
- MSFT
- GOOGL
- AMZN
- AMD
- AVGO

Instead of trying to predict exact future stock prices, the goal is to build a recommendation-support model that compares these companies and produces a short-term forecast/risk summary.

## Planned Data Sources

- `yfinance`: historical stock price data from Yahoo Finance
- Alpha Vantage API: recent news and sentiment data
- Gemini API: AI-generated summaries using the free tier, unless OpenAI is required by the professor

## Current Workflow

1. Pull stock data with `yfinance`
2. Clean and combine the data
3. Create charts and basic analysis
4. Add features like returns, moving averages, volatility, and volume change
5. Add Alpha Vantage sentiment data
6. Create a 7-day prediction target
7. Build and evaluate a simple model
8. Generate AI summaries

## Carlos Update: Processed Data Files

I started working through the stock data pipeline in `notebooks/AI_Investment_Signals.ipynb`.

So far, I pulled stock data, cleaned it, added calculated features, added Alpha Vantage sentiment scores, and created a 7-day up/down target for the model.

The main idea I used here is not predicting the exact stock price, but predicting whether a stock is up or not up 7 trading days later.

The notebook generated these shared files:

### `model_ready_stock_data.csv`

This is the full cleaned dataset that combines the stock data, calculated features, sentiment data, and target columns.

It includes:

- stock prices
- ticker and date
- calculated stock features
- Alpha Vantage sentiment scores
- article counts
- the 7-day future return
- the 7-day up/down target

This file is useful for looking at the full dataset and understanding everything that was created.

### `training_ready_stock_data.csv`

This is the file I set up for model training.

It only includes rows where we already know what happened 7 trading days later. For example, the model can look at a past date, use the stock features and sentiment from that date, and learn whether the stock went up or down 7 trading days later.

The model inputs are columns like returns, moving averages, volatility, volume change, sentiment score, and article count.

The model target is:

```text
target_up_7d
```

Where:

```text
1 = the stock went up 7 trading days later
0 = the stock did not go up 7 trading days later
```

### `latest_prediction_rows.csv`

This file has the newest stock rows.

This file shouldn't be used for training because the future 7-day result has not happened yet. Instead, after training the model using `training_ready_stock_data.csv`, it can be used to make current predictions.

The prediction is not an exact future stock price. The model is predicting direction:

```text
Will this stock likely be up or not up 7 trading days from now?
```

### `daily_sentiment_scores.csv`

This file is in `data/external/`.

It has the daily average news sentiment score by ticker from Alpha Vantage. The sentiment score is also merged into the processed stock files, but this file is useful if we want to look at sentiment separately.

### Why The Files Are Split This Way

The newest 7 trading days do not have a real answer yet because we do not know what the stock price will be 7 trading days later.

Because of that, the rows are split into two groups:

- `training_ready_stock_data.csv`: older rows where the 7-day result is already known
- `latest_prediction_rows.csv`: newest rows where the model can make predictions, but we cannot train on them yet

## Possible Modeling Flow

1. Use `training_ready_stock_data.csv` to train the model.
2. Evaluate the model on older historical data.
3. Use `latest_prediction_rows.csv` to make current 7-day direction predictions.
4. Use AI to summarize the model results in plain English.

The model learns from the past, then applies what it learned to the newest rows.

## Luca Update

You can add any updates here if you want, like what you have worked on so far, any data pulls or API testing you have done, charts you created, or next steps you want to work on.

## Notebooks

We can use the `notebooks/` folder to upload our Colab/Jupyter notebooks as we work through the project. This makes it easier to share progress and keep everything in one place.


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
