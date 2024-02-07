from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler


def lstm_single_layer_model(x_train):
    # model
    model = Sequential()
    model.add(create_single_layer(x_train))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_model(model: Sequential, X_train, y_train, x_test):
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    predictions = model.predict(x_test)
    return predictions


def run_model_whole_dataset(model: Sequential, X, y, ep, bs):
    model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, epochs=ep, batch_size=bs)


def create_single_layer(x_train):
    return LSTM(units=50, activation="relu", input_shape=(x_train.shape[1], 1))


def generate_five_day_predictions(df):
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

    model = lstm_single_layer_model(X)

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

        # Append the next item to the sequence
        last_sequence = np.append(last_sequence[1:], next_item_scaled)

        # Inverse transform to get the actual stock price
        next_item = scaler.inverse_transform([[next_item_scaled]])[0, 0]

        # Append the next date
        next_date = last_date + timedelta(days=i + 1)

        next_items.append({"Date": next_date, "Close": next_item})

    # Display the generated next 5 items with dates
    for item in next_items:
        print(
            f"Date: {item['Date'].strftime('%Y-%m-%d')}, Close Price: {item['Close']:.2f}"
        )
