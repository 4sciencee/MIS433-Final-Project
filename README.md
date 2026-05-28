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

## Main Notebook

The main project notebook is:

```text
notebooks/AI_Investment_Signals.ipynb
```

## Setup Instructions

To run locally in VS Code:

1. Clone the repo.
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env`.
4. Add your Alpha Vantage API key to `.env`:

```text
ALPHA_VANTAGE_API_KEY=your_key_here
```

5. Open `notebooks/AI_Investment_Signals.ipynb` and run the cells.

To run in Google Colab:

1. Open the notebook from GitHub or upload it to Colab.
2. Mount Google Drive when prompted.
3. Store the Alpha Vantage key in Colab Secrets as `ALPHA_VANTAGE_API_KEY`.
4. Run the notebook cells from top to bottom.

Do not commit a real `.env` file or API key to GitHub.

## Repository Structure

```text
notebooks/              Jupyter notebooks for analysis and project demo
data/                   Generated stock, sentiment, and model-ready CSV files
src/                    Reusable Python functions
output/                 Generated charts for notebook and slides
app/                    Optional Streamlit dashboard
```
