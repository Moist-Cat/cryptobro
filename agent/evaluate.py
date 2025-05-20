from agent.config import EVAL_WINDOW
from utils import calculate_profit


def fitness(prices, index, action, window=EVAL_WINDOW):
    return evaluate(prices, index, window) * action


def evaluate(prices, index, window=EVAL_WINDOW):
    if len(prices) - window <= index:
        print("WARNING - Lookahead!", window, index, prices[-1])
        return 0
    return calculate_profit(prices, index, max_days=window)
