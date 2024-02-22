# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance
from keys.read_keys import *
from util import get_data, split_data, get_metrics_results
from models.xgboost.gb_model import *
from models.lstm.ls_model import *
from models.lstm.ls_multi_layered_model import *
from keras.optimizers import Adam, Adagrad, RMSprop
from stocks.stocks_logic import logic

from pybroker import Alpaca
from alpaca.trading.client import TradingClient


OPS = [
    Adam(learning_rate=0.001),
    Adagrad(learning_rate=0.001),
    RMSprop(learning_rate=0.001),
]


def xg_make_preds_more_readable(preds):
    """
    Converts predictions from 2d array into 1d array and makes them easier to read as a dataframe later.
    @return new_preds: DataFrame
    """
    actual_items = []
    for item in preds["Close"]:
        actual_items.append(item[0])
    new_preds = pd.DataFrame(index=preds["Date"], data=actual_items, columns=["Close"])
    return new_preds


def xg_path_bayesian_optimisation(stock):
    df = get_data(stock, "max")
    df.dropna()
    X_train, x_test, y_train, y_test = split_data(df)
    model = create_model_bayesian_optimisation(X_train, y_train)
    model.fit(X_train, y_train)
    predictions = model.predict(x_test)
    df_res = get_metrics_results(y_test, predictions)
    preds = generate_five_day_predictions_xgb(df)
    preds = xg_make_preds_more_readable(preds)
    return df_res, preds


def xg_path_mvp(stock):
    df_res, preds = xg_path_bayesian_optimisation(stock)
    return df_res, preds


def create_dataset_lstm(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i : (i + time_steps), 0]
        X.append(a)
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)


def ls_do_data_prep(data):
    # Extracting closing prices
    data = data["Close"].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create sequences and labels
    def create_sequences(data, seq_length):
        sequences = []
        labels = []
        for i in range(len(data) - seq_length):
            seq = data[i : i + seq_length]
            label = data[i + seq_length]
            sequences.append(seq)
            labels.append(label)
        return np.array(sequences), np.array(labels)

    # Set sequence length (e.g., 5 days)
    seq_length = 5
    X, y = create_sequences(scaled_data, seq_length)
    split_percentage = 0.8
    split = int(split_percentage * len(X))

    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test, scaler


def make_lstm_preds_better_df(preds):
    """
    Makes the predictions from the lstm model more readable
    @return preds: DataFrame
    """
    ac_p = []
    for item in preds["Close"]:
        ac_p.append(item)
    new_preds = pd.DataFrame(index=preds["Date"], data=ac_p, columns=["Close"])
    return new_preds


def ls_path_mvp(stock):
    amazon_df = get_data(stock, "max")
    amazon_df.dropna()
    X_train, X_test, y_train, y_test, scaler = ls_do_data_prep(amazon_df)

    model = lstm_multi_layered("relu", 5, [100, 50, 20], OPS[0])
    model.fit(X_train, y_train, epochs=50, batch_size=16)
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    df_res = get_metrics_results(y_test, predicted_prices)
    preds = generate_five_day_predictions_lstm(amazon_df, model)
    return df_res, make_lstm_preds_better_df(preds)


def run_type_model_mvp():
    inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
    if inp == 1:
        metrics, preds = xg_path_mvp("AMZN")
        print(f"metrics: {metrics}")
        print(f"preds: {preds}")
        preds.to_csv("xgboost_preds.csv")
        metrics.to_csv("xgboost_metrics_initial.csv")
    elif inp == 2:
        metrics, preds = ls_path_mvp("AMZN")
        print(f"metrics: {metrics}")
        print(f"preds: {preds}")
        preds.to_csv("lstm_preds.csv")
        metrics.to_csv("lstm_metrics_initial.csv")
    else:
        print("Invalid input")


def get_keys():
    xk = read_xgb_key()
    lk = read_lstm_key()
    xs = read_xgb_sec()
    ls = read_lstm_sec()
    return xk, lk, xs, ls


def get_todays_price(stock_symbol: str):
    # Get today's date
    today = datetime.now().strftime("%Y-%m-%d")
    # Download historical data for today
    data = yfinance.download(stock_symbol, start=today, end=today)
    # Extract the close price for today
    close_price_today = data["Close"].iloc[0]
    return close_price_today


def run_stock_stuff(tc: TradingClient, stock_symbol: str, path_to_preds: str):
    # load xgboost preds file
    preds = pd.read_csv(path_to_preds)
    # get tomorrows date
    tomorrow = datetime.now() + timedelta(days=1)
    # get prediction for tomorrow
    tomorrow_pred = preds.loc[tomorrow]
    print(f"tomorrow's prediction: {tomorrow_pred}")
    # buy or sell based on prediction
    todays_price = get_todays_price(stock_symbol)
    logic(tomorrow_pred, todays_price, stock_symbol, 10, tc)
    print("done")


def main():
    xk, lk, xs, ls = get_keys()
    inp = int(input("1 for MVP, 2 for full software:"))
    if inp == 1:  # MVP
        run_type_model_mvp()
    elif inp == 2:  # Alpaca stuff
        inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
        if inp == 1:
            # connect to xgboost alpaca account
            xg = Alpaca(xk, xs)
            df = xg.query(
                symbols=["BARC.L"],
                start_date="1/9/2022",
                end_date="1/9/2023",
                timeframe="1d",
            )
            print(f"shape: {df.shape}")
            xg_alp = TradingClient(xk, xs, paper=True)
            print(xg_alp.get_account())
            run_stock_stuff(xg_alp, "BCS", "xgboost_preds.csv")
        if inp == 2:
            lstm = Alpaca(lk, ls)
            df = lstm.query(
                symbols=["AMZN"],
                start_date="1/9/2022",
                end_date="1/9/2023",
                timeframe="1d",
            )
            print(f"shape: {df.shape}")
            ls_alp = TradingClient(lk, ls, paper=True)
            print(ls_alp.get_account())
            run_stock_stuff(ls_alp, "BCS", "lstm_preds.csv")
    else:
        print("Invalid input")
    return 0


if __name__ == "__main__":
    main()
