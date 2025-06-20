from pathlib import Path
import os

from pathlib import Path
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from tqdm import tqdm

from meta import load_csv, DIST

# autocorrelation
def find_clusters(violations, gap=2):
    """Group consecutive violations within 'gap' days"""
    clusters = []
    current_cluster = []
    for v in violations:
        if current_cluster:
            if v["day"] <= current_cluster[-1]["day"] + gap:
                # close enough to cluster
                current_cluster.append(v)
            else:
                # the cluster has been broken
                # adding a "cluster" of size == 1 is poinless
                if len(current_cluster) > 2:
                    clusters.append(current_cluster)
                current_cluster = [v]
        else:
            current_cluster = [v]
    return clusters


def get_log_ratios(prices):
    ratios = prices[1:] / prices[:-1]
    log_ratios = np.log(ratios)

    return log_ratios


def estimate_parameters(prices, dist=DIST):
    """Estimate mu and sigma from price data."""
    return dist.fit(get_log_ratios(prices))


def simulate_prices(prices, N, num_paths=1, start=0, end=None):
    """
    Simulate $N$ days starting from $S_0$. Assumes each
    ratio distributes Laplace with parameters $mu$ and $sigma$.
    """
    if not end:
        end = len(prices) - 1
    S0 = prices[end]
    if start > 45:
        args = estimate_parameters(prices[start : end + 1])
    else:
        args = estimate_parameters(prices)

    prices = np.zeros((num_paths, N + 1))
    prices[:, 0] = S0

    d = DIST(*args)
    for t in range(1, N + 1):
        prices[:, t] = prices[:, t - 1] * np.exp(
            d.rvs(size=num_paths),
        )
    return prices


def stock_price_ci(prices, days=7, alpha=0.05, n_sim=10000, start=0, end=None):
    """
    Constructs confidence intervals for future stock prices using loglaplace distribution properties.

    Parameters:
    S0 (float): Current stock price
    mu (float): location of the Laplace
    sigma (float): scale
    days (int): Time horizon in days
    alpha (float): Error (default 0.05 for 95% CI)
    n_sim (int): Number of simulations for Monte Carlo approach (only approach for now)

    Returns:
    dict: Contains confidence intervals and other statistics
    plt.Figure: Visualization of the price distribution
    """
    # Analytical method (loglaplace distribution)
    # nashi
    if end is None:
        end = len(prices) - 1

    # MC
    future_prices, lower_mc, upper_mc = mc_ci(prices, start, end, days, alpha, n_sim)

    return {
        "current_price": prices[end],
        "days_ahead": days,
        "confidence_level": 1 - alpha,
        "future_prices": future_prices,
        "monte_carlo_ci": (lower_mc, upper_mc),
        "ci": (lower_mc, upper_mc),
    }


def mc_ci(prices, start, end, days=7, alpha=0.05, n_sim=10000):
    """
    Generate confidence intervals with a Monte Carlo simulation.
    """
    # Notice we take into account each day!
    paths = simulate_prices(
        prices,
        days,
        num_paths=n_sim,
        start=start,
        end=end
    )

    path_min = np.min(paths, axis=1)
    path_max = np.max(paths, axis=1)

    # Find tightest [L, U] that contains (1-alpha) paths
    L = np.percentile(path_min, 100 * alpha / 2)
    U = np.percentile(path_max, 100 * (1 - alpha / 2))

    return paths[:, -1], L, U


def process_window(prices, start, end, window_size, alpha, n_sim):
    """Process single window and return violation if exists"""
    current_window = prices[end : end + window_size + 1]  # included

    # Get confidence interval
    res = stock_price_ci(
        prices, days=window_size, alpha=alpha, n_sim=n_sim, start=start, end=end
    )
    lower, upper = res["ci"]

    # Check window
    for price in current_window[1:]:
        if not (lower <= price and price <= upper):
            return {
                "day": end,
                "price": price,
                "lower": lower,
                "upper": upper,
                "window": window_size,
            }
    return None


def calculate_profit(prices, index, max_days=7, alpha=0.66, n_sim=1000):
    """
    Utility function to quickly compute profit/debt.
    """
    if (index + max_days) >= len(prices):
        return 0

    res = process_window(prices, 0, index, max_days, alpha, n_sim)
    start = prices[index]

    if res is None:
        end = prices[index + max_days]
    else:
        end = res["lower"] if res["price"] < res["lower"] else res["upper"]

    # 1.03 - 1 = 0.03
    # 0.95 - 1 = -0.05
    return ((end / start) - 1) * 100


def calculate_rsi(prices, current_index, period=10):
    """Pure function calculating RSI for given index"""
    if current_index < period:
        return 50
    elif current_index == len(prices):
        return 50

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

if __name__ == "__main__":
    with open("datasets/SOLUSDT.csv") as file:
        prices = load_csv(file)

    for i in range(1000):
        print(i)
        stock_price_ci(prices)
