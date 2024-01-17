from yfinance import *
import pandas as pd
from sklearn.model_selection import train_test_split


def get_data(ticker, period) -> pd.DataFrame:
    data = download(ticker, period=period)
    return data


def split_data(data: pd.DataFrame):
    x_train, y_train, x_test, y_test = train_test_split(
        data, train_size=0.7, test_size=0.3
    )
    return x_train, y_train, x_test, y_test
