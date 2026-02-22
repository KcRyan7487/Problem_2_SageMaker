import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

MODEL_FILENAME = "model.pkl"


def find_training_csv(train_dir: str) -> str:
    expected = os.path.join(train_dir, "winequality-red.csv")
    if os.path.exists(expected):
        return expected

    csv_files = [f for f in os.listdir(train_dir) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {train_dir}")
    return os.path.join(train_dir, csv_files[0])


if __name__ == "__main__":
    train_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    output_dir = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    csv_path = find_training_csv(train_dir)
    print(f"Loading training data from: {csv_path}")
    df = pd.read_csv(csv_path, sep=";")

    X = df.drop("quality", axis=1)
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    print(f"RMSE: {rmse:.4f}")

    model_path = os.path.join(model_dir, MODEL_FILENAME)
    joblib.dump(model, model_path)
    print(f"Saved model to: {model_path}")

    metrics = {"rmse": rmse, "rows": int(len(df)), "features": int(X.shape[1])}
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f)
    print(f"Saved metrics to: {metrics_path}")
