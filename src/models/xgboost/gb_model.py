from sklearn.model_selection import cross_val_score
import xgboost as xgboost
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from bayes_opt import BayesianOptimization


def create_model(n_est=1000, lr=0.01, md=5):
    """
    Create xgboost model without using bayesian optimisation
    @return model
    """
    model = xgboost.XGBRegressor(
        n_estimators=n_est, learning_rate=lr, max_depth=md, random_state=1
    )
    return model


def create_model_best():
    """
    Create xgboost model using best parameters
    @return model
    """
    model = xgboost.XGBRegressor(
        n_estimators=578, learning_rate=0.02733, max_depth=9, random_state=1
    )
    return model


def bayesian_optimisation(X, y):
    """
    Perform bayesian optimisation on the model
    @return model
    """

    def xgb_evaluate(n_est, lr, md):
        model = xgboost.XGBRegressor(
            n_estimators=int(n_est),
            learning_rate=lr,
            max_depth=int(md),
            random_state=1,
        )
        return np.mean(
            cross_val_score(
                model, X, y, cv=5, n_jobs=-1, scoring="neg_mean_squared_error"
            )
        )

    xgb_bo = BayesianOptimization(
        xgb_evaluate,
        {
            "n_est": (100, 1000),
            "lr": (0.01, 0.5),
            "md": (3, 20),
        },
    )

    xgb_bo.maximize(init_points=10, n_iter=10)
    params = xgb_bo.max["params"]
    model = xgboost.XGBRegressor(
        n_estimators=int(params["n_est"]),
        learning_rate=params["lr"],
        max_depth=int(params["md"]),
        random_state=1,
    )
    return model


def create_model_bayesian_optimisation(X, y):
    """
    Create xgboost model using bayesian optimisation
    @return model
    """
    model = bayesian_optimisation(X, y)
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


def generate_five_day_predictions_xgb(df: pd.DataFrame, model):
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
    model.fit(X, y)
    last_sequence = data["Close"].values[-10:]
    # print(f"first last_sequence: {last_sequence}")
    next_items = []
    last_date = datetime.today().strftime("%Y-%m-%d")
    last_date = datetime.strptime(last_date, "%Y-%m-%d")  # convert to datetime
    for i in range(5):
        input_sequence = last_sequence.reshape(1, 10)
        # print(f"input_seq: {input_sequence}")
        next_item = model.predict(input_sequence)
        # print(f"next item: {next_item}")
        last_sequence = np.append(last_sequence[1:], next_item)
        next_items.append(
            {"Date": last_date + timedelta(days=i + 1), "Close": next_item}
        )

    # for item in next_items:
    #     print(f"Date: {item['Date']}, Close: {item['Close']}")
    return pd.DataFrame(next_items)


# end generate_five_day_predictions_xgb
