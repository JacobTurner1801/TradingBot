# Author: Jacob Turner
# Main job is to run each model on the data up until today
# and then predict the next day's price
from keys.read_keys import read_xgb_key, read_xgb_sec


def main():
    # read the api keys for xgboost and lstm accounts
    # xgb_key = read_xgb_key()
    # xgb_sec = read_xgb_sec()
    # print("{}, {}".format(xgb_key, xgb_sec))
    print("Hello World")

if __name__ == "__main__":
    main()
