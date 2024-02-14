# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keys.read_keys import *
from util import get_data, split_data, get_metrics_results
from models.xgboost.gb_model import *
from models.lstm.ls_model import *
from models.lstm.ls_multi_layered_model import *
from pybroker import Alpaca


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


def test_lstms(X_train, y_train):
    model = lstm_multi_layered("relu", 5, [10])
    model.fit(X_train, y_train, epochs=50, batch_size=16)
    return model


def ls_path_mvp(stock):
    amazon_df = get_data(stock, "max")
    amazon_df.dropna()
    X_train, X_test, y_train, y_test, scaler = ls_do_data_prep(amazon_df)
    model = lstm_multi_layered("relu", 5, [100, 50, 20], "adam")
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
    elif inp == 2:
        metrics, preds = ls_path_mvp("AMZN")
        print(f"metrics: {metrics}")
        print(f"preds: {preds}")
        preds.to_csv("lstm_preds.csv")
    else:
        print("Invalid input")


def get_keys():
    xk = read_xgb_key()
    lk = read_lstm_key()
    xs = read_xgb_sec()
    ls = read_lstm_sec()
    return xk, lk, xs, ls


def main():
    inp = int(input("1 for MVP, 2 for full software:"))
    if inp == 1:  # MVP
        run_type_model_mvp()
    elif inp == 2:  # Alpaca stuff
        inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
        if inp == 1:
            # create xgboost model
            amazon_df: pd.DataFrame = get_data("AMZN", "max")
            amazon_df.dropna()
            preds = generate_five_day_predictions_xgb(amazon_df)
        if inp == 2:
            # create single layer lstm model
            amazon_df: pd.DataFrame = get_data("AMZN", "max")
            amazon_df.dropna()
            close_data = amazon_df["Close"].values.reshape(-1, 1)
            # put close_data and date index into new df of just close data
            data: pd.DataFrame = pd.DataFrame(
                close_data, index=amazon_df.index, columns=["Close"]
            )
            data = data.sort_index()
            # model = lstm_single_layer_model(data)
            model = lstm_multi_layered(data, "relu", 1)
            preds = generate_five_day_predictions_lstm(data, model)
            # since we have created the predictions, we can now connect to alpaca
            # and buy or sell based on the predictions
            xgb_key, lstm_key, xgb_sec, lstm_sec = get_keys()
            # alpaca = Alpaca()
            # alpaca.connect()
            # print("Connected to Alpaca")
    else:
        print("Invalid input")
    return 0


if __name__ == "__main__":
    main()
