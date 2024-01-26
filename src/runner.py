# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keys.read_keys import *
from util import get_data, split_data, get_metrics_results
from models.xgboost.gb_model import *
from models.lstm.ls_model import lstm_single_layer_model, run_model
from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense

def xg_path():
    xgb_key = read_xgb_key()
    xgb_sec = read_xgb_sec()
    amazon_df = get_data("AMZN", "max")
    amazon_df.dropna()
    X_train, x_test, y_train, y_test = split_data(amazon_df)
    predictions = train_and_fit(X_train, y_train, x_test)  # model created here
    df_res = get_metrics_results(y_test, predictions)
    return df_res


def ls_path():
    lstm_key = read_lstm_key()
    lstm_sec = read_lstm_sec()
    amazon_df = get_data("AMZN", "max")
    amazon_df.dropna()
    scalar = MinMaxScaler(feature_range=(0, 1))
    close_data = amazon_df["Close"].values.reshape(-1, 1)
    scaled_data = scalar.fit_transform(close_data)
    train_data = scaled_data[0:int(close_data.size), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        if i<= 61:
            print(x_train)
            print(y_train)
            print()
        
    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1], 1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    
    test_data = scaled_data[close_data.size - 60: , :]
    # Create the data sets x_test and y_test
    x_test = []
    y_test = close_data[len(x_train):, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scalar.inverse_transform(predictions)
    df_res = get_metrics_results(y_test, predictions)
    return df_res
    


def main():
    inp = int(input("Enter 1 for XGBoost, 2 for LSTM: "))
    if inp == 1:
        results = xg_path()
        print(results)
    elif inp == 2:
        results = ls_path()
        print(results)
    else:
        print("Invalid input")
    return 0


if __name__ == "__main__":
    main()
