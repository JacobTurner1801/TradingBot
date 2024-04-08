# calculate profit / loss given start and end
def calculate_profit_loss(start_cash, end_cash):
    # calculate profit
    profit = end_cash - start_cash
    return profit


def calculate_roi(gain_from_investment, cost_of_investment):
    # calculate return on investment
    roi = (gain_from_investment - cost_of_investment) / cost_of_investment
    return roi


def calculate_winning_percentage(total_trades, profitable):
    # calculate winning percentage
    win_perc = profitable / total_trades
    return win_perc * 100


def main():
    start = int(input("Enter start cash: "))
    end = int(input("Enter end cash: "))
    profit = calculate_profit_loss(start, end)
    print(f"profit: {profit}")
    gain = int(input("Enter gain from investment: "))
    cost = int(input("Enter cost of investment: "))
    roi = calculate_roi(gain, cost)
    print(f"roi: {roi}")
    total = int(input("Enter total trades: "))
    profitable = int(input("Enter profitable trades: "))
    win_perc = calculate_winning_percentage(total, profitable)
    print(f"winning percentage: {win_perc}")
    print("done")


main()
