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


def xg_path_mvp():
    amazon_df = get_data("AMZN", "max")
    amazon_df.dropna()
    X_train, x_test, y_train, y_test = split_data(amazon_df)
    predictions = train_and_fit(X_train, y_train, x_test)  # model created here
    df_res = get_metrics_results(y_test, predictions)
    # df_res.to_csv("xgboost_mvp_results.csv")
    return df_res


def create_dataset_lstm(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i : (i + time_steps), 0]
        X.append(a)
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)


def ls_do_data_prep(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    vals = data
    scaled_data = scaler.fit_transform(vals.reshape(-1, 1))
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    # create datasets
    time_steps = 10
    X_train, y_train = create_dataset_lstm(train_data, time_steps)
    X_test, y_test = create_dataset_lstm(test_data, time_steps)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test, y_train, y_test, scaler


def ls_path_mvp():
    amazon_df = get_data("AMZN", "max")
    amazon_df.dropna()
    data = amazon_df["Close"].values.reshape(-1, 1)
    X_train, X_test, y_train, y_test, scaler = ls_do_data_prep(data)
    model = lstm_single_layer_model(X_train)
    predictions = run_model(model, X_train, y_train, X_test)
    predictions = scaler.inverse_transform(predictions)
    df_res = get_metrics_results(y_test, predictions)
    # df_res.to_csv("lstm_single_layers_results.csv")
    return df_res


def run_type_model_mvp():
    inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
    if inp == 1:
        results = xg_path_mvp()
        print(results)
    elif inp == 2:
        results = ls_path_mvp()
        print(results)
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
    if inp == 1:
        run_type_model_mvp()
    elif inp == 2:
        inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
        if inp == 1:
            # create xgboost model
            amazon_df: pd.DataFrame = get_data("AMZN", "max")
            amazon_df.dropna()
            generate_five_day_predictions_xgb(amazon_df)
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
            generate_five_day_predictions_lstm(data, model)
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
