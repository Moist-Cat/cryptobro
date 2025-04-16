from scipy.stats import laplace
import numpy as np


def estimate_parameters(prices):
    """Estimate mu and sigma from price data."""
    ratios = prices[1:] / prices[:-1]
    log_ratios = np.log(ratios)

    return laplace.fit(log_ratios)


def simulate_prices(S0, mu, sigma, N, num_paths):
    prices = np.zeros((num_paths, N + 1))
    prices[:, 0] = S0
    for t in range(1, N + 1):
        prices[:, t] = prices[:, t - 1] * np.exp(
            np.random.laplace(mu, sigma, num_paths)
        )
    return prices


def mc_ci(S0, mu, sigma, days=5, alpha=0.05, n_sim=10000):
    """
    Monte Carlo simulation
    We get the last price since `simulate_prices` simulates the whole thing
    NOTE A possible use-case for the rest of the values could be to
    construct smaller CIs... but one might as well just re-run the algorithm
    with the updated price
    """
    future_prices = simulate_prices(S0, mu, sigma, days, n_sim)[
        :, -1
    ]  # the last simulated price corresponds to the price after `days` days

    lower_mc = np.percentile(future_prices, 100 * alpha / 2)
    upper_mc = np.percentile(future_prices, 100 * (1 - alpha / 2))

    return future_prices, lower_mc, upper_mc
