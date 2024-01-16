from yfinance import *
import pandas as pd


def get_data(ticker, period) -> pd.DataFrame:
    data = download(ticker, period=period)
    return data
