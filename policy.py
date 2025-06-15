import numpy as np
from utils import mc_ci, estimate_parameters, calculate_profit, calculate_rsi
from tqdm import tqdm
import random

from agent import (
    Agent,
    Manager,
    Brain,
    load_agent,
    evaluate,
)

BUY = 1
SELL = -1
WAIT = 0

STATE = {
    "capital": 0,
    "stock": 0,
    "risk": 0,
}


class GlobalState:
    """
    For policies that need to perform heavy number-crunching. Don't abuse it.
    """

    pass


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


def policy_agent(
    prices,
    index,
    risk,
    history,
    load=False,
    min_value=-11.0,
    max_value=11.0,
    genes="",
    size=-1,
    layers=-1,
):
    """
    Use the agent.

    Enable `load` if you want to load an exported agent.
    """
    if not history or not hasattr(GlobalState, "agent"):
        # new agent
        if load:
            agent = load_agent()
        else:
            if genes:
                genes = eval(genes)
                if not isinstance(genes, list) or not isinstance(genes[0], float):
                    raise ValueError("The genes should be a list of float")
                agent = Agent(brain=Brain(genes=genes, size=size, layers=layers))
            else:
                agent = Agent()
        GlobalState.agent = agent

    agent = GlobalState.agent
    # we don't use .advance to avoid switching datasets
    # at the end
    agent.dataset = prices
    agent.index = index

    action = agent.decide(agent.get_state())
    print("COT:", agent.brain._cot)
    print("Action:", action)
    if len(agent.brain._cot) < agent.brain.MAX_CHILDREN + 1:
        action = WAIT
    if action > max_value or action < min_value:
        action = WAIT

    return {
        "action": action,
        # chain of thought, etc
        "logs": {},
    }


# rsi
def policy_rsi(
    prices,
    index,
    risk,
    history,
    days=10,
    overbought=90,
    oversold=30,
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
    rsi_value = calculate_rsi(prices, index, days)

    if rsi_value is None:  # Not enough data
        return {
            "action": WAIT,
            "logs": {},
        }

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


def policy_med(
    prices,
    index,
    risk,
    history,
    window_size=10,
    before_window=True,
    assume_laplacian=True,
    oversold=1.0,
    overbought=0.0,
):
    """
    1. Calculate 10-day moving mean/median
    2. See if it's outside bounds given an alpha

    The size of the window and wheter to check before or after can be configured.
    `assume_laplace` controls whether to use the median or the mean.
    The median is more resistant to volatility and thus, is the default.
    """
    if index < window_size:
        return {"action": WAIT, "logs": {}}

    frame = prices[index - window_size : index]

    if assume_laplacian:
        window = np.median(frame)
    else:
        window = np.mean(frame)

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
            "moving_centrality": window,
        },
    }


def policy_expected_value(prices, index, risk, history):
    return {
        "action": np.sign(estimate_parameters(prices[: index + 1])[1]),
        "logs": {},
    }


def policy_monkey(prices, index, risk, history, no_wait=False):
    """
    Random. Use this to judge if your policy sucks.
    Enable `no_wait` to prevent the monkey from HOLD/WAITing
    """
    if no_wait:
        action = random.choice((-1, 1))
    else:
        action = random.choice((-1, 0, 1))

    return {
        "action": action,
        "logs": {},
    }


POLICIES = [
    policy_agent,
    policy_monkey,
    policy_med,
    policy_expected_value,
    policy_rsi,
]


def general_policy(
    prices,
    state,
    risk,
    policy,
    reverse_strategy=False,
    auto_close=7,
):
    """
    In general, iterate the prices and apply a policy.
    Notice prices is simply an iterable, you can generate
    prices using any model and this function will eat them seamlessly.

        handle_action(price, state, action)

    This is a function that decides what to do when executing an action.
    Modifies the state and returns None.

    `auto_close` Closes a stale order after `auto_close` days. The CIs are calculated using this value so be careful.

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
    for index in tqdm(range(len(prices))):
        price = prices[index]

        res = policy(prices, index, risk, action_log)
        action = res["action"]
        if reverse_strategy:
            # no funny bitwise business was posssible *sob*
            action = -action

        _extract_logs(policy_logs, res["logs"])

        action_log.append(action)
        state["capital"] += evaluate.fitness(prices, index, action, auto_close)

        health = _net_worth(prices, state, index)
        net_log.append(health)

    logs = {
        "action": action_log,
        "net_worth": net_log,
        "policy": policy_logs,
    }

    return state, logs
