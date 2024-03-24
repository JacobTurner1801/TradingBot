import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime


def plot_predictions(preds_path: str, stock: str):
    """
    Plots the predictions and actuals.
    @param preds: DataFrame
    @param actuals: DataFrame
    """
    orig = yf.download(stock, start="2024-03-19", end="2024-03-23")
    orig = orig[["Close"]]
    dates = orig.index[-len(orig) :]
    orig = pd.DataFrame(orig, columns=["Close"], index=dates)
    preds = pd.read_csv(preds_path)
    preds["Date"] = pd.to_datetime(preds["Date"])
    # remove last row
    preds = preds[:-1]
    plt.figure(figsize=(15, 6))
    # Plot close prices
    plt.plot(orig, label="Close Price", marker="o", linestyle="--")
    # Plot validation predictions
    plt.plot(
        preds["Date"], preds["Close"], label="Prediction", marker="o", linestyle="--"
    )
    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Price Predictions")
    # Add legend
    plt.legend()
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    # Show the plot
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


def merge_metrics_files(path: str):
    """
    Merge metrics files from a given path (in this case the ../metrics_mvp folder)
    """
    # get the contents of the directory
    files = os.listdir(path)
    # filter for csv files
    files = [file for file in files if file.endswith(".csv")]
    # read in all the files as csv
    dfs = [pd.read_csv(path + file) for file in files]
    # concatenate the dataframes
    df = pd.concat(dfs)
    # drop the index column
    df = df.drop(columns=["Unnamed: 0"])
    # save the file
    df.to_csv(path + "merged_metrics.csv")
    return df


def create_moving_average(df: pd.DataFrame, window: int):
    """
    Creates a moving average of the close price.
    @param df: DataFrame
    @param window: int
    @return moving_average: DataFrame
    """
    moving_average = df["Close"].rolling(window=window).mean()
    return moving_average


def plot_validation_prices_from_csvs(stock: str, validation_path: str):
    orig = yf.download(stock, end="2024-03-18")
    orig = orig[["Close"]]
    dates = orig.index[-len(orig) :]
    orig = pd.DataFrame(orig, columns=["Close"], index=dates)
    df_val = pd.read_csv(validation_path)
    df_val["Date"] = pd.to_datetime(df_val["Date"])
    df_val["Avg"] = create_moving_average(df_val, 8)
    # df_preds = pd.read_csv(preds_path)
    # df_preds["Date"] = pd.to_datetime(df_preds["Date"])

    plt.figure(figsize=(15, 6))
    # Plot close prices
    plt.plot(orig, label="Close Price")

    # Plot validation predictions
    plt.plot(
        df_val["Date"],
        df_val["Avg"],
        label="Validation Prediction",
    )

    # make the validation predictions appear above the actual close prices

    # Plot future prediction (as a single point)
    # plt.plot(
    #     df_preds["Date"],
    #     df_preds["Close"],
    #     label="Future Prediction",
    # )

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Price Predictions LSTM Validation Data")

    # Add legend
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


# plot_validation_prices_from_csvs("BCS", "../validation_preds_lstm.csv")


def main():
    stock = "BCS"
    actuals = yf.download(stock, start="2024-03-19", end="2024-03-23")
    actuals = actuals[["Close"]]
    dates = actuals.index[-len(actuals) :]
    actuals = pd.DataFrame(actuals, columns=["Close"], index=dates)
    # print(actuals)
    print("Do you want to:")
    print("1. Plot validation")
    print("2. Plot predictions")
    inp = int(input("Enter 1 or 2: "))
    if inp == 1:
        validation_path = str(input("Enter path to validation predictions: "))
        plot_validation_prices_from_csvs(stock, validation_path)
    elif inp == 2:
        preds_path = input("Enter path to predictions: ")
        plot_predictions(preds_path, stock)
    else:
        print("Invalid input")


main()
