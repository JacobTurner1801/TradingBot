import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yfinance as yf
import numpy as np


def plot_predictions(preds: pd.DataFrame, actuals: pd.DataFrame):
    """
    Plots the predictions and actuals.
    @param preds: DataFrame
    @param actuals: DataFrame
    """
    # set colour theme
    sns.axes_style("dark")
    plt.plot(preds, label="Predictions")
    plt.plot(actuals, label="Actuals")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Predictions vs Actuals")
    plt.legend()
    plt.show()


def plot_metrics(metrics: pd.DataFrame):
    """
    Plots the metrics.
    @param metrics: DataFrame
    """
    sns.axes_style("dark")
    plt.plot(metrics)
    plt.xlabel("Metric")
    plt.ylabel("Value")
    plt.title("Metrics")
    plt.show()


def create_metrics_table(metrics: pd.DataFrame):
    """
    Creates a table of metrics.
    @param metrics: DataFrame
    @return table: DataFrame
    """
    return pd.plotting.table(data=metrics, loc="center", colWidths=[0.2, 0.2, 0.2, 0.2])


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


def plot_close_prices_from_csvs(validation_path: str, preds_path: str):
    orig = yf.download("BCS", period="max")
    orig = orig[["Close"]]
    dates = orig.index[-len(orig) :]
    orig = pd.DataFrame(orig, columns=["Close"], index=dates)
    df_val = pd.read_csv(validation_path)
    df_val["Date"] = pd.to_datetime(df_val["Date"])
    df_val["Avg"] = create_moving_average(df_val, 31)
    df_preds = pd.read_csv(preds_path)
    df_preds["Date"] = pd.to_datetime(df_preds["Date"])

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
    plt.plot(
        df_preds["Date"],
        df_preds["Close"],
        label="Future Prediction",
    )

    # Add labels and title
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"BCS Price Predictions")

    # Add legend
    plt.legend()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Show the plot
    plt.grid(True)
    # plt.tight_layout()
    plt.show()


plot_close_prices_from_csvs("../validation_preds_xgb.csv", "../xgboost_preds_2.csv")
