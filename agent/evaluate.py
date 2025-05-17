from utils import calculate_profit


def fitness(prices, index, action, window=5):
    return evaluate(prices, index, window) * action

    return action * calculate_profit(prices, index)


def evaluate(prices, index, window=5):
    if len(prices) - window <= index:
        return None
    return calculate_profit(prices, index, max_days=window)
