from keras.models import Sequential
from keras.layers import Dropout, LSTM, Dense

def lstm_single_layer_model(x_train):
    # define model architecture

    # Initialize model
    model = Sequential()
    
    # LSTM layer 1
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.25))
    
    # final layer
    model.add(Dense(units = 1))
    return model

def run_model(model: Sequential, X_train, y_train, x_test):
    model.compile(optimizer = "adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=1, batch_size=1)
    predictions = model.predict(x_test)
    return predictions
