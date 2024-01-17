# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
from keys.read_keys import *
from util import get_data


def main():
    xgb_key = read_xgb_key()
    xgb_sec = read_xgb_sec()
    lstm_key = read_lstm_key()
    lstm_sec = read_lstm_sec()
    # amazon_df = get_data("AMZN", "max")


if __name__ == "__main__":
    main()
