from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


FEATURE_COLUMNS = [
    "daily_return",
    "return_7d",
    "return_30d",
    "ma_7d",
    "ma_30d",
    "volatility_30d",
    "volume_change",
    "avg_sentiment_score",
    "article_count",
]


def prepare_training_data(df, feature_columns=None):
    """Keep only rows that have complete features and a known target."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    required_columns = feature_columns + ["target_up_7d"]
    training_df = df.dropna(subset=required_columns).copy()
    training_df["target_up_7d"] = training_df["target_up_7d"].astype(int)
    return training_df


def train_random_forest(train_df, feature_columns=None):
    """Train a basic Random Forest model for short-term direction prediction."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    train_df = prepare_training_data(train_df, feature_columns)
    X_train = train_df[feature_columns]
    y_train = train_df["target_up_7d"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, test_df, feature_columns=None):
    """Return standard classification metrics for the model."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    test_df = prepare_training_data(test_df, feature_columns)
    X_test = test_df[feature_columns]
    y_test = test_df["target_up_7d"]
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }
