import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# import numpy as np


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


# CHANGE THIS PATH FOR THE CSV FILES TO MERGE
df = merge_metrics_files("../../metrics_mvp/xgb/")
print(df)
