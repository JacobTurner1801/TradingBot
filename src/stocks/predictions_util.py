import pandas as pd
import datetime
from util import get_metrics_results


def read_predictions(path) -> pd.DataFrame:
    """
    Reads predictions from a file and returns a DataFrame.
    @param path: str
    @return predictions: DataFrame
    """
    # we have a csv with index as date and column as close
    return pd.read_csv(path)


def get_current_date() -> datetime:
    """
    Gets the current date.
    @return date: datetime
    """
    return datetime.datetime.now().date()


def get_todays_prediction(df: pd.DataFrame, date: datetime._Date):
    """
    Gets the prediction for today's date.
    @param path: str
    @param date: str
    @return prediction
    """
    return df.loc[date, "Close"]


def get_metrics_for_todays_prediction(pred_for_today, actual_close: float):
    """
    Gets the metrics for today's prediction.
    @param preds: DataFrame
    @param actual_close: float
    @return metrics: dict
    """
    return get_metrics_results(actual_close, pred_for_today)
