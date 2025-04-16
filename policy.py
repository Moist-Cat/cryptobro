from utils import mc_ci, estimate_parameters
from tqdm import tqdm

__all__ = ["policy_threshold"]

BUY = 1
SELL = -1
WAIT = 0

STATE = {
    "capital": 0,
    "stock": 0,
    "risk": 0,
}


def policy_threshold(
    prices, index, location, scale, risk, window_size=10, after_window=False
):
    """
    1. Calculate 10-day moving average
    2. See if it's (almost) outside bounds given an alhpa

    The size of the window and wheter to check before or after can be configured.
    """
    if index < window_size:
        return WAIT
    window = sum(prices[index - window_size : index]) / window_size

    S0 = prices[index - window_size * after_window]  # True/False are integers

    _, lower_mc, higher_mc = mc_ci(S0, location, scale, days=window_size, alpha=risk)

    A = lower_mc
    B = higher_mc
    C = window
    result = (C - A) / (B - A)

    if result > 0.9:
        return SELL
    elif result < 0.1:
        return BUY
    return WAIT


def execute_action(
    price, state, action: int, order_size=1, buy_everything=True, sell_everything=False
):
    """
    Execute an action. The order size and (somewhat) dynamic sub-policies to buy and sell can
    be configured
    """
    state["capital"] -= price * action
    if state["stock"] < 0 and action > 0 and buy_everything:
        order_size = -1 * state["stock"]
    elif state["stock"] > 0 and action < 0 and sell_everything:
        order_size = 1 * state["stock"]
    state["stock"] += order_size * action


def _net_worth(prices, state, index=-1):
    """
    Implementation-specific net-worth calculation. Works with any float.
    """
    return state["stock"] * prices[index] + state["capital"]


def general_policy(prices, policy, state, risk, skip_repeated_actions=False):
    """
    In general, iterate the prices and apply a policy.
    Notice prices is simply an iterable, you can generate
    prices using any model and this function will eat them seamlessly.
    """
    policy = policy or (lambda p, i, l, s: 0)
    state = state or STATE
    # we don't want to modify the original state in any case
    state = state.copy()
    state["risk"] = risk

    location, scale = estimate_parameters(prices)

    action_log = []
    debt_log = [0]
    last_action = 0
    for index in tqdm(range(len(prices))):
        # last prices sub-policy
        price = prices[index]
        action = policy(prices, index, location, scale, risk)
        # skip "scraping the wave" option
        if action and action == last_action and skip_repeated_actions:
            continue
        last_action = action
        action_log.append(action)
        execute_action(price, state, action)

        health = _net_worth(prices, state, index)
        if health < 0:
            debt_log.append(health)

    logs = {
        "action": action_log,
        "debt": debt_log,
    }

    return state, logs
