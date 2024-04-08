from models.lstm.lstm_util import *
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam


def single(af, sq, lr):
    model = Sequential()
    model.add(
        LSTM(
            units=50,
            activation=af,
            input_shape=(sq, 1),
        )
    )
    op = Adam(learning_rate=lr)
    model.add(Dense(units=1))
    model.compile(optimizer=op, loss="mean_squared_error")
    return model


def lstm_multi_layered(
    activation_func="relu",
    seq_length=5,
    neuron_list=[100, 50, 20],
    optimize="adam",
):
    """
    Create LSTM model
    @return model
    """
    model = Sequential()
    for i in range(len(neuron_list)):
        model.add(
            LSTM(
                units=neuron_list[i],
                activation=activation_func,
                return_sequences=True if i < len(neuron_list) - 1 else False,
                input_shape=(seq_length, 1),
            )
        )
    model.add(Dense(units=1))
    model.compile(optimizer=optimize, loss="mean_squared_error")
    return model
