import xgboost as xgboost
import numpy as np
import pandas as pd
from datetime import timedelta

def create_model(n_est=1000, lr=0.01, md=5):
    """
    Create xgboost model without using bayesian optimisation
    @return model
    """
    model = xgboost.XGBRegressor(
        n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=1
    )
    return model

def train_and_fit(x_train, y_train, x_test):
    """
    train and run xgboost model on a split dataset
    @return predictions
    """
    model = create_model()
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    return prediction

def generate_five_day_predictions_xgb(df: pd.DataFrame):
    """
    Generate 5 day predictions using xgboost model
    @return nothing, just prints the predictions
    """
    df = df.dropna()
    df = df.sort_index()
    # lets only work with time series data then I can refactor later
    data = pd.DataFrame(data=df["Close"].values, index=df.index, columns=["Close"])
    X, y = [], []
    for i in range(len(data) - 10 - 5):
        X.append(data["Close"].values[i : i + 10])
        y.append(data["Close"].values[i + 10 + 5])
    X, y = np.array(X), np.array(y)
    model = create_model()
    model.fit(X, y)
    last_sequence = data["Close"].values[-10:]
    # print(f"first last_sequence: {last_sequence}")
    next_items = []
    last_date = data.index[-1]
    for i in range(5):
        input_sequence = last_sequence.reshape(1, 10)
        # print(f"input_seq: {input_sequence}")
        next_item = model.predict(input_sequence)
        # print(f"next item: {next_item}")
        last_sequence = np.append(last_sequence[1:], next_item)
        next_items.append(
            {"Date": last_date + timedelta(days=i + 1), "Close": next_item}
        )
    for item in next_items:
        print(f"Date: {item['Date']}, Close: {item['Close']}")
    # TODO: return the predictions
# end generate_five_day_predictions_xgb