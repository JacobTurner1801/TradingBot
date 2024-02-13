from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from models.lstm.lstm_util import *

def lstm_single_layer_model(x_train):
    """
    Single layer LSTM model (for MVP)
    @return model
    """
    # model
    model = Sequential()
    model.add(create_single_layer(x_train, activation_func="tanh"))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def run_model(model: Sequential, X_train, y_train, x_test):
    """
    Run single LSTM model, this is not tested for the multilayered one
    See ls_multi_layered_model.py for the multilayered model
    Epochs = 50, batch_size = 32
    @return predictions
    """
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    predictions = model.predict(x_test)
    return predictions

def run_model_whole_dataset(model: Sequential, X, y, ep, bs):
    """
    Run single layered model for the entire dataset
    @return nothing, just fits the model
    """
    model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=ep, batch_size=bs)

def generate_five_day_predictions_lstm(df, model):
    """
    Generate 5 day predictions using single layer LSTM model
    @return nothing, just prints the predictions 
    """
    # Feature scaling
    scaler = MinMaxScaler()
    df["Close_scaled"] = scaler.fit_transform(df[["Close"]])

    # Create sequences and labels for the entire dataset
    sequence_length = 10
    X, y = [], []

    for i in range(
        len(df) - sequence_length - 5
    ):  # 5 for predicting 5 days into the future
        X.append(df["Close_scaled"].values[i : i + sequence_length])
        y.append(df["Close_scaled"].values[i + sequence_length + 5])

    X, y = np.array(X), np.array(y)

    run_model_whole_dataset(model, X, y, 50, 32)

    # Generate the next 5 items in the sequence
    last_sequence = df["Close_scaled"].values[-sequence_length:]
    next_items = []

    # Assume last_date is the last date in your dataset
    last_date = df.index[-1]

    for i in range(5):
        # Reshape the sequence for prediction
        input_sequence = last_sequence.reshape(1, sequence_length, 1)

        # Predict the next item
        next_item_scaled = model.predict(input_sequence)[0, 0]
        next_item_scaled = np.reshape(next_item_scaled, (1, -1))
        # print(f"next item scaled: {next_item_scaled}")
        # weird hack to make everything work correctly, idk why this is needed xd
        extracted_value_from_scaled = next_item_scaled[0][0]
        # put into np.array
        extracted = np.array([extracted_value_from_scaled])
        # reshape into 2d array
        extracted = extracted.reshape(-1, 1)
        # Append the next item to the sequence
        last_sequence = np.append(last_sequence[1:], extracted_value_from_scaled)
        # print(f"last sequence: {last_sequence}")
        # Inverse transform to get the actual stock price
        next_item = scaler.inverse_transform(extracted)
        # print(f"next item: {next_item}")
        # Append the next date
        next_date = last_date + timedelta(days=1)
        last_date = next_date

        next_items.append({"Date": next_date, "Close": next_item[0][0]})

    
    # Display the generated next 5 items with dates
    for item in next_items:
        print(
            f"Date: {item['Date'].strftime('%Y-%m-%d')}, Close Price: {item['Close']:.2f}"
        )
    # TODO: return the predictions
# end generate_five_day_predictions_lstm
