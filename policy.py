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
    prices, index, risk, history, window_size=10, before_window=True
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

    _, lower_mc, higher_mc = mc_ci(prices, 0, index, days=window_size, alpha=risk)

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


# rsi
def policy_rsi(
    prices,
    index,
    risk,
    history,
    days=14,
    overbought=70,
    oversold=30,
    risk_adjusted_thresholds=True,
):
    """RSI-based trading policy with risk-adjusted thresholds

    Args:
        prices: Array of historical prices
        index: Current time index in the prices array
        risk: Risk tolerance (0-1) affecting threshold sensitivity
        history: Trading position history
        days: RSI calculation period
        overbought: Default overbought threshold
        oversold: Default oversold threshold
        risk_adjusted_thresholds: Whether to adjust thresholds based on risk

    Returns:
        dict: Trading action (-1, 0, 1) and logs
    """
    # Calculate RSI using pure function
    rsi_value = _calculate_rsi(prices, index, days)

    if rsi_value is None:  # Not enough data
        return {
            "action": WAIT,
            "logs": {},
        }

    # Adjust thresholds based on risk
    if risk_adjusted_thresholds:
        overbought, oversold = _adjust_thresholds(risk, overbought, oversold)

    # Determine action
    action = WAIT
    if rsi_value >= overbought:
        action = SELL
    elif rsi_value <= oversold:
        action = BUY

    return {
        "action": action,
        "logs": {
            "rsi": rsi_value,
            "overbought": overbought,
            "oversold": oversold,
        },
    }


def _calculate_rsi(prices, current_index, period=14):
    """Pure function calculating RSI for given index"""
    if current_index < period:
        return None

    gains = []
    losses = []

    for i in range(current_index - period + 1, current_index + 1):
        if i == 0:
            continue
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
        else:
            losses.append(abs(change))

    avg_gain = sum(gains) / period if gains else 0
    avg_loss = sum(losses) / period if losses else 0

    if avg_loss == 0:
        return 100 if avg_gain != 0 else 50

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _adjust_thresholds(risk, base_overbought, base_oversold):
    """Adjust RSI thresholds based on risk tolerance (0-1)"""
    # Higher risk tolerance widens the neutral zone
    risk_factor = 1 - abs(risk - 0.5) * 2  # 0-1 where 0.5 is neutral
    adjusted_ob = base_overbought + (30 * (1 - risk_factor))
    adjusted_os = base_oversold - (30 * (1 - risk_factor))

    # Keep thresholds within reasonable bounds
    return (min(max(adjusted_ob, 60), 90), max(min(adjusted_os, 40), 10))


def policy_med(
    prices, index, risk, history, window_size=10, before_window=True
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

    _, lower_mc, higher_mc = mc_ci(prices, 0, index, days=window_size, alpha=risk)

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


def policy_monkey(prices, index, risk, history):
    """
    Random. Use this to judge if your policy sucks.
    """
    return {
        "action": random.choice((-1, 0, 1)),
        "logs": {},
    }


POLICIES = [policy_threshold, policy_monkey, policy_med, policy_rsi]

BASE_CAPITAL = 1 * (10**6)


def _get_order_size(price, volume):
    return volume // price


def execute_action(price, state, action: int):
    """
    Execute an action. The order size and (somewhat) dynamic sub-policies to buy and sell can
    be configured
    """
    # we always want to have the same volume of movement
    # through all the simulation

    order_size = _get_order_size(price, BASE_CAPITAL)

    state["capital"] -= action * order_size * price
    state["stock"] += order_size * action


def _get_lookback(risk):
    if risk > 0.79:
        return 7
    elif risk > 0.65:
        return 15
    elif risk > 0.49:
        return 27
    elif risk > 0.32:
        return 37
    elif risk > 0.19:
        return 45
    return 45


def general_policy(
    prices,
    state,
    risk,
    policy,
    handle_action,
    orders=True,
    concurrent_orders=False,
    auto_close=5,
    reverse_strategy=False,
    memoryless=False,
    apply_order=True,
):
    """
    In general, iterate the prices and apply a policy.
    Notice prices is simply an iterable, you can generate
    prices using any model and this function will eat them seamlessly.

        handle_action(price, state, action)

    This is a function that decides what to do when executing an action.
    Modifies the state and returns None.

    `orders` enables or disables stop-loss and take-profit
    `concurrent_orders` enables or disables concurrent orders (if we should wait when we have an order up)
    `auto_close` Closes a stale order after `auto_close` days. The CIs are calculated using this value so be careful.
    `memoryless` enables or disables memory (transform our simulation into a Markov process)
    `apply_order` if you only want to see the orders made by the policy disable this
    """
    policy = policy or (lambda p, i, l, s: 0)
    state = state or STATE
    # we don't want to modify the original state in any case
    state = state.copy()
    state["risk"] = risk

    policy_logs = {}
    action_log = []
    net_log = []
    # a tuple with
    # (index (completed), price, order)
    # where order is a tuple with
    # (TYPE, upper_bound, lower_bound, index (issues))
    order_log = []
    last_action = 0
    current_order = (0, float("inf"), float("-inf"), 0)
    for index in tqdm(range(len(prices))):
        price = prices[index]

        # Confidence intervals for stop-loss and profit
        S0 = price
        _, lower_mc, higher_mc = mc_ci(prices, 0, index, days=auto_close, alpha=risk)

        # Close Order
        # we also sell/buy immediately if the order is too old
        if (
            (current_order[1] <= price or current_order[2] >= price)
            or index - current_order[3] >= auto_close
        ) and orders:
            # we complete the operation before doing anything else
            if not memoryless:
                handle_action(price, state, -current_order[0])
            order_log.append((index, price, current_order))
            current_order = (0, float("inf"), float("-inf"), index)

            # Choose to show logs and make the operation more
            # `realistic`. This makes the graphic somewhat dirty
            # so disable it if you want less noise
            if apply_order:
                action_log.append(-current_order[0])
                health = _net_worth(prices, state, index)
                net_log.append(health)
                continue

        # we can't place another order if we
        # already have an order placed
        if current_order[0] and not concurrent_orders:
            action_log.append(WAIT)
            health = _net_worth(prices, state, index)
            net_log.append(health)
            continue

        res = policy(prices, index, risk, action_log)
        action = res["action"]
        if reverse_strategy:
            # no funny bitwise business was posssible *sob*
            action = -action
        _extract_logs(policy_logs, res["logs"])

        # independant of the order type
        top = higher_mc
        bottom = lower_mc
        current_order = (action, top, bottom, index)

        last_action = action
        action_log.append(action)
        if not memoryless:
            handle_action(price, state, action)

        health = _net_worth(prices, state, index)
        net_log.append(health)

    logs = {
        "action": action_log,
        "order": order_log,
        "net_worth": net_log,
        "policy": policy_logs,
    }

    return state, logs
