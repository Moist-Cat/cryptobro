import numpy as np
from utils import mc_ci, estimate_parameters
from tqdm import tqdm
import random

BUY = 1
SELL = -1
WAIT = 0

STATE = {
    "capital": 0,
    "stock": 0,
    "risk": 0,
}


def _net_worth(prices, state, index=-1):
    """
    Implementation-specific net-worth calculation. Works with any float.
    """
    return state["stock"] * prices[index] + state["capital"]


def _extract_logs(logs: dict, log: dict):
    """
    Given
    logs: dict[list]
    log: dict

    Merge the dicts by appending the new value and maintaining the same keys.
    """
    for key in log.keys():
        if key not in logs:
            logs[key] = []

        logs[key].append(log[key])


# policies
def policy_threshold(
    prices, index, location, scale, risk, window_size=10, before_window=True
):
    """
    1. Calculate 10-day moving average
    2. See if it's (almost) outside bounds given an alhpa

    The size of the window and wheter to check before or after can be configured.
    """
    if index < window_size:
        return {"action": WAIT, "logs": {}}
    window = sum(prices[index - window_size : index]) / window_size

    S0 = prices[index - window_size * before_window]  # True/False are integers

    _, lower_mc, higher_mc = mc_ci(S0, location, scale, days=window_size, alpha=risk)

    A = lower_mc
    B = higher_mc
    C = window
    result = (C - A) / (B - A)

    action = WAIT

    if result > 0.9:
        action = SELL
    elif result < 0.1:
        action = BUY

    return {
        "action": action,
        "logs": {
            "ci": (lower_mc, higher_mc),
            "moving_average": window,
        },
    }

def policy_logmed(
    prices, index, location, scale, risk, window_size=10, before_window=True
):
    """
    1. Calculate 10-day moving average
    2. See if it's (almost) outside bounds given an alhpa

    The size of the window and wheter to check before or after can be configured.
    """
    if index < window_size:
        return {"action": WAIT, "logs": {}}

    frame = prices[index - window_size : index]

    window = np.median(frame)

    S0 = prices[index - window_size * before_window]  # True/False are integers

    _, lower_mc, higher_mc = mc_ci(S0, location, scale, days=window_size, alpha=risk)

    A = lower_mc
    B = higher_mc
    C = window
    result = (C - A) / (B - A)

    action = WAIT

    if result > 1:
        action = SELL
    elif result < 0.0:
        action = BUY

    return {
        "action": action,
        "logs": {
            "ci": (lower_mc, higher_mc),
            "moving_average": window,
        },
    }


def policy_monkey(prices, index, location, scale, risk):
    """
    Random. Use this to judge if your policy sucks.
    """
    return {
        "action": random.choice((-1, 0, 1)),
        "logs": {},
    }


POLICIES = [policy_threshold, policy_monkey, policy_logmed]


def execute_action(
    price, state, action: int, order_size=1, buy_everything=False, sell_everything=False
):
    """
    Execute an action. The order size and (somewhat) dynamic sub-policies to buy and sell can
    be configured
    """
    if state["stock"] < 0 and action > 0 and buy_everything:
        order_size = -1 * state["stock"]
    elif state["stock"] > 0 and action < 0 and sell_everything:
        order_size = 1 * state["stock"]

    state["capital"] -= action * order_size * price
    state["stock"] += order_size * action


def general_policy(
    prices, state, risk, policy, handle_action, skip_repeated_actions=False
):
    """
    In general, iterate the prices and apply a policy.
    Notice prices is simply an iterable, you can generate
    prices using any model and this function will eat them seamlessly.

        handle_action(price, state, action)

    This is a function that decides what to do when executing an action.
    Modifies the state and returns None.
    """
    policy = policy or (lambda p, i, l, s: 0)
    state = state or STATE
    # we don't want to modify the original state in any case
    state = state.copy()
    state["risk"] = risk

    location, scale = estimate_parameters(prices)

    policy_logs = {}
    action_log = []
    net_log = []
    last_action = 0
    current_order = (0, float("inf"), float("-inf"))
    for index in tqdm(range(len(prices))):
        price = prices[index]

        if (current_order[1] <= price or current_order[2] >= price) and False:
            handle_action(price, state, -current_order[0])
            action_log.append(-current_order[0])
            health = _net_worth(prices, state, index)
            net_log.append(health)
            current_order = (0, float("inf"), float("-inf"))
            continue

        if current_order[0] and False:
            action_log.append(WAIT)
            health = _net_worth(prices, state, index)
            net_log.append(health)
            continue

        res = policy(prices, index, location, scale, risk)
        action = res["action"]
        _extract_logs(policy_logs, res["logs"])

        if action == BUY:
            top = 1.05 * price
            bottom = 0.99 * price
            current_order = (action, top, bottom)
        elif action == SELL:
            top = 0.99 * price
            bottom = 1.05 * price
            current_order = (action, top, bottom)

        # skip "scraping the wave" option
        if action and action == last_action and skip_repeated_actions:
            continue
        last_action = action
        action_log.append(action)
        handle_action(price, state, action)

        health = _net_worth(prices, state, index)
        net_log.append(health)

    logs = {
        "action": action_log,
        "net_worth": net_log,
        "policy": policy_logs,
    }

    return state, logs
