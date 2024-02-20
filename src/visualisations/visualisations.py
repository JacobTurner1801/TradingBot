import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
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
