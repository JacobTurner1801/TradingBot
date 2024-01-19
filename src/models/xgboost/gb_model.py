import xgboost as xgboost


def train_and_fit(x_train, y_train, x_test):
    model = xgboost.XGBRegressor()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction
