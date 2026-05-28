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
]


def train_random_forest(train_df, feature_columns=None):
    """Train a basic Random Forest model for short-term direction prediction."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    X_train = train_df[feature_columns]
    y_train = train_df["target_up_7d"]

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_classifier(model, test_df, feature_columns=None):
    """Return standard classification metrics for the model."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    X_test = test_df[feature_columns]
    y_test = test_df["target_up_7d"]
    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }

