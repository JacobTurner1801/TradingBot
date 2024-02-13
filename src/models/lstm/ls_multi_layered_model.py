from models.lstm.lstm_util import *
from keras.models import Sequential
from keras.layers import Dense


def single(af, sq):
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            activation=af,
            input_shape=(sq, 1),
        )
    )
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def lstm_multi_layered(activation_func="relu", num_layers=10, seq_length=5):
    """
    Create LSTM model
    @return model
    """
    if num_layers == 1:
        return single(activation_func, seq_length)

    model = Sequential()
    for i in range(num_layers):
        model.add(
            LSTM(
                units=50,
                activation=activation_func,
                return_sequences=True if i < num_layers - 1 else False,
                input_shape=(seq_length, 1),
            )
        )
    model.add(Dense(units=1))
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


"""
Run the model with the given data
@return predictions
"""


def run_model_multi_layered(model: Sequential, X, y, X_test, ep, bs):
    model.fit(X, y, epochs=ep, batch_size=bs)
    predictions = model.predict(X_test)
    return predictions
