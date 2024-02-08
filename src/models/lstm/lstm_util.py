from keras.layers import LSTM

"""__summary
create single LSTM layer
units=50, activation=relu (by default), input_shape=(x_train.shape[1], 1) for stocks
@return LSTM layer
"""
def create_single_layer(x_train, activation_func="relu"):
    return LSTM(units=50, activation=activation_func, input_shape=(x_train.shape[1], 1))
