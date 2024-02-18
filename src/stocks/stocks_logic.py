from orders import create_buy_order, create_sell_order
from alpaca.trading.client import TradingClient


def logic(tomorrow_pred, current_price, sym, qty, acc: TradingClient):
    if tomorrow_pred > current_price:
        order = create_buy_order(account=acc, symbol=sym, qty=qty)
        print(
            f"Buying {qty} shares of {sym} at {current_price} with order id {order.id}"
        )
    elif tomorrow_pred < current_price:
        order = create_sell_order(account=acc, symbol=sym, qty=qty)
        print(
            f"Selling {qty} shares of {sym} at {current_price} with order id {order.id}"
        )
    else:
        print("No action needed")
        return
