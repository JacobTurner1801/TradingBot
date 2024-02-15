from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest


def create_buy_order(account: TradingClient, symbol, qty):
    market_order_req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        type=type,
        time_in_force=TimeInForce.DAY,
    )
    return account.submit_order(market_order_req)


def create_sell_order(account: TradingClient, symbol, qty):
    market_order_req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        type=type,
        time_in_force=TimeInForce.DAY,
    )
    return account.submit_order(market_order_req)
