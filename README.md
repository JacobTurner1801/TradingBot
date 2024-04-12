# Machine Learning Trading Bot

This bot uses machine learning to predict stock prices and uses those predictions as indicators to trade in a sandbox environment.

## Use
- Clone the repo
- make sure that you have the API keys for two different alpaca accounts, as I'm using one for xgboost and one for LSTM.
  - save these in ```XGB_API_KEY```, ```LSTM_API_KEY```and ```XGB_SECRET_KEY```, ```LSTM_SECRET_KEY``` respectively.
- run ```python runner.py``` from the src folder and follow the instructions.

## Demo
- see the mp4 file in this repo for a demo for generating predictions.
- the demo does not include alpaca interaction due to the dissertation not being submitted at the time of recording.
