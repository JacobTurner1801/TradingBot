from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense


def lstm_single_layer_model(x_train):
    # model
    model = Sequential()
    model.add(LSTM(units=50, activation="tanh", input_shape=(x_train.shape[1], 1)))
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_model(model: Sequential, X_train, y_train, x_test):
    model.fit(X_train, y_train, epochs=50, batch_size=32)
    predictions = model.predict(x_test)
    return predictions
