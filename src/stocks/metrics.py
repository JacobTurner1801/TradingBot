# calculate profit / loss given start and end
def calculate_profit_loss(start_cash, end_cash):
    # calculate profit
    profit = end_cash - start_cash
    return profit


def calculate_roi(gain_from_investment, cost_of_investment):
    # calculate return on investment
    roi = (gain_from_investment - cost_of_investment) / cost_of_investment
    return roi
