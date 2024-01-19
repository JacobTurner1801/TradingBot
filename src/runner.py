# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
from keys.read_keys import *
from util import get_data, split_data, get_metrics_results
from models.xgboost.gb_model import *


def xg_path():
    xgb_key = read_xgb_key()
    xgb_sec = read_xgb_sec()
    amazon_df = get_data("AMZN", "max")
    amazon_df.dropna()
    # create xgboost model
    X_train, x_test, y_train, y_test = split_data(amazon_df)
    predictions = train_and_fit(X_train, y_train, x_test)
    df_res = get_metrics_results(y_test, predictions)
    return df_res


def ls_path():
    lstm_key = read_lstm_key()
    lstm_sec = read_lstm_sec()
    amazon_df = get_data("AMZN", "max")


def main():
    inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
    if inp == 1:
        df = xg_path()
        print(df)
    elif inp == 2:
        ls_path()
    else:
        print("Invalid input")
    return 0


if __name__ == "__main__":
    main()
