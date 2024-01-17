from xgboost import *


def train_and_fit(x_train, y_train, x_test):
    model = XGBRegressor()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction
