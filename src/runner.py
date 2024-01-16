# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
from keys.read_keys import read_xgb_key, read_xgb_sec
from util import get_data


def main():
    xgb_key = read_xgb_key()
    xgb_sec = read_xgb_sec()
    # TODO: read lstm keys
    amazon_df = get_data("AMZN", "max")


if __name__ == "__main__":
    main()
