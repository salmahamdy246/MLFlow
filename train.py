import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

mlflow.set_experiment("House_Price_Prediction")


def load_and_split_data():
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    return train_test_split(X, y, test_size=0.20, random_state=42)


def train_and_log_run(
    run_name, max_depth, learning_rate, X_train, X_val, y_train, y_val
):
    with mlflow.start_run(run_name=run_name):
        model = xgb.XGBRegressor(
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_estimators=100,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.xgboost.log_model(model, artifact_path="model")
        print(
            f"Logged {run_name} -> RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}"
        )


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_and_split_data()

    experiments = [
        {"name": "Run 1", "max_depth": 3, "learning_rate": 0.1},
        {"name": "Run 2", "max_depth": 5, "learning_rate": 0.05},
        {"name": "Run 3", "max_depth": 7, "learning_rate": 0.01},
    ]

    for exp in experiments:
        train_and_log_run(
            exp["name"],
            exp["max_depth"],
            exp["learning_rate"],
            X_train,
            X_val,
            y_train,
            y_val,
        )