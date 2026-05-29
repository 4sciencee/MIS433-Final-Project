import pandas as pd


def add_stock_features(df):
    """Add basic stock features used for modeling."""
    df = df.copy()
    df["daily_return"] = df.groupby("ticker")["close"].pct_change()
    df["return_7d"] = df.groupby("ticker")["close"].pct_change(7)
    df["return_30d"] = df.groupby("ticker")["close"].pct_change(30)
    df["ma_7d"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(7).mean())
    df["ma_30d"] = df.groupby("ticker")["close"].transform(lambda s: s.rolling(30).mean())
    df["volatility_30d"] = df.groupby("ticker")["daily_return"].transform(lambda s: s.rolling(30).std())
    df["volume_change"] = df.groupby("ticker")["volume"].pct_change()
    return df


def add_prediction_target(df, days_ahead=7):
    """Create a target for whether close price goes up later."""
    df = df.copy()
    df["future_close_7d"] = df.groupby("ticker")["close"].shift(-days_ahead)
    df["future_return_7d"] = (df["future_close_7d"] / df["close"]) - 1
    df["target_up_7d"] = (df["future_return_7d"] > 0).where(
        df["future_close_7d"].notna()
    )
    df["target_up_7d"] = df["target_up_7d"].astype("Int64")
    return df
