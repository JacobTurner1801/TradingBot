from models.lstm.lstm_util import *
from keras.models import Sequential
from keras.layers import Dense


"""
Create multi-layered LSTM model
@return model
"""
def lstm_multi_layered(X, activation_func="relu", num_layers=10):
    model = Sequential()
    for i in range(0, num_layers):
        model.add(
            LSTM(units=50, activation=activation_func, 
                 return_sequences=True))
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