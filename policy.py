from utils import mc_ci, estimate_parameters
from tqdm import tqdm

__all__ = ["policy_threshold"]

BUY = 1
SELL = -1
WAIT = 0

STATE = {
    "capital": 0,
    "stock": 0,
}


def policy_threshold(prices, index, location=0, scale=1, risk=0.05):
    if index < 10:
        # we calculate a 10-day window
        return WAIT
    window = sum(prices[index - 10 : index]) / 10

    S0 = prices[index - 10]

    _, lower_mc, higher_mc = mc_ci(S0, location, scale, days=10, alpha=risk)

    A = lower_mc
    B = higher_mc
    C = window
    result = (C - A)/(B - A)

    if result > 0.9:
        return SELL
    elif result < 0.1:
        return BUY
    return WAIT


def execute_action(price, state, action: int, order_size=1):
    state["capital"] -= price * action
    state["stock"] += order_size * action


def general_policy(prices, policy=None, state=None, risk=0.05):
    """
    In general, iterate the prices and apply a policy.
    Notice prices is simply an iterable, you can generate
    prices using any model and this function will eat them seamlessly.
    """
    policy = policy or (lambda p, i, l, s: 0)
    state = state or STATE
    # we don't want to modify the original state in any case
    state = state.copy()

    location, scale = estimate_parameters(prices)

    action_log = []
    for index in tqdm(range(len(prices))):
        price = prices[index]
        action = policy(prices, index, location, scale, risk)
        action_log.append(action)
        execute_action(price, state, action)

    return state, action_log
