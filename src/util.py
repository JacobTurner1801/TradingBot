from yfinance import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import numpy as np


def get_data(ticker, period) -> pd.DataFrame:
    data = download(ticker, period=period)
    return data


def symetric_mape(y_test, predictions):
    constant = 100 / len(y_test)
    return constant * np.sum(
        2 * np.abs(predictions - y_test) / (np.abs(y_test) + np.abs(predictions))
    )


def split_data(data: pd.DataFrame):
    y = data["Close"]
    X = data.drop(["Close"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    return X_train, X_test, y_train, y_test


def make_frame(metrics):
    frame = pd.DataFrame()
    frame["MAE"] = [metrics[0]]
    frame["RMSE"] = [metrics[1]]
    frame["SMAPE"] = [metrics[2]]
    frame["R2"] = [metrics[3]]
    return frame


def get_metrics_results(y_test, predictions):
    print(predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    smape = symetric_mape(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return make_frame([mae, rmse, smape, r2])
