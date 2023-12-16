import os


def read_xgb_key():
    """
    Reads the XGBoost API key from the environment variable XGB_API_KEY.
    """
    return os.environ.get("XGB_API_KEY")


def read_xgb_sec():
    """
    Reads the XGBoost API secret from the environment variable XGB_API_SECRET.
    """
    return os.environ.get("XGB_API_SECRET")


# TODO: Add LSTM keys
# def read_lstm_key():
#     """
#     Reads the LSTM API key from the environment variable LSTM_API_KEY.
#     """
#     return os.environ.get("LSTM_API_KEY")


# def read_lstm_sec():
#     """
#     Reads the LSTM API secret from the environment variable LSTM_API_SECRET.
#     """
#     return os.environ.get("LSTM_API_SECRET")
