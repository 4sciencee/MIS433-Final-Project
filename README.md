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
6. Create prediction targets for short-term stock movement
7. Build and compare simple models
8. Use the best model setup to create current prediction signals
9. Add AI summaries as the next project feature

## Carlos Update: Main Notebook Progress

I started working through the stock data pipeline in `notebooks/AI_Investment_Signals.ipynb`.

So far, I pulled stock data, cleaned it, added calculated features, added Alpha Vantage sentiment scores, tested several model setups, and created a final prediction output.

The main idea is not predicting the exact stock price. The model predicts whether a stock is likely to have a meaningful upward move over a short time period.

The best setup from the current notebook is:

```text
Model: Random Forest
Prediction window: 7 trading days
Target: stock rises more than 1.5%
Features: price/volume features + Alpha Vantage sentiment
Metric used to choose model: balanced accuracy
```

The notebook generated these shared files and outputs:

### `model_ready_stock_data.csv`

This is the full cleaned dataset that combines the stock data, calculated features, sentiment data, and target columns.

It includes:

- stock prices
- ticker and date
- calculated stock features
- Alpha Vantage sentiment scores
- article counts
- future returns for different short-term windows
- target columns used for modeling

This file is useful for looking at the full dataset and understanding everything that was created.

### `training_ready_stock_data.csv`

This is the file set up for model training.

It includes older rows where we already know what happened later. For example, the model can look at a past date, use the stock features and sentiment from that date, and learn whether the stock had a meaningful upward move later.

The model inputs are columns like returns, moving averages, volatility, volume change, sentiment score, and article count.

### `latest_prediction_rows.csv`

This file has the newest stock rows.

This file is used for current predictions, not training. It contains the newest available row for each ticker.

### `daily_sentiment_scores.csv`

This file is in `data/external/`.

It has the daily average news sentiment score by ticker from Alpha Vantage. The sentiment score is also merged into the processed stock files, but this file is useful if we want to look at sentiment separately.

### `model_comparison.csv`

This file is in `outputs/model_results/`.

It stores the model testing results. The notebook compares Logistic Regression, Decision Tree, and Random Forest with different prediction windows, cutoffs, and thresholds.

### `focused_model_tuning.csv`

This file is in `outputs/model_results/`.

It stores a focused tuning test on the best setup from the broader model comparison.

### `latest_direction_predictions.csv`

This file is in `outputs/model_results/`.

This is the final prediction output from the main notebook. It gives one current prediction for each company.

```text
ticker
latest close price
prediction signal
prediction probability
sentiment score
article count
```

### Chart Outputs

The main notebook also saves charts in `outputs/charts/`:

- normalized stock performance
- volatility comparison
- latest sentiment by company
- model feature importance

These can be used for the notebook, slides, or presentation screenshots.

## Possible Modeling Flow

1. Use `training_ready_stock_data.csv` to train the model.
2. Evaluate the model on older historical data.
3. Compare model results in `model_comparison.csv`.
4. Use `latest_prediction_rows.csv` to make current direction predictions.
5. Use Gemini/OpenAI to summarize the prediction results in plain English.

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
