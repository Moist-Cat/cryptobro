from scipy.stats import laplace
import numpy as np
import inspect
import functools
import pickle
import os
from pathlib import Path
import hashlib
from tqdm import tqdm

# elegant!
def persistent_lru_cache(maxsize=100000, cache_dir="/var/tmp/.cache", typed=False):
    """Persistent least-recently-used cache decorator.

    Arguments:
        maxsize: Maximum number of cached items (None for unlimited)
        cache_dir: Directory to store cached items
        typed: If True, arguments of different types will be cached separately
    """
    # I can't use shelve to load the whole thing once because it's like a +10 GB
    def decorator(func):
        # Ensure cache directory exists
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a unique key for these arguments
            key = (args, frozenset(kwargs.items())) if kwargs else args
            # Create a hash of the key for the filename
            key_hash = hashlib.sha256(pickle.dumps(key)).hexdigest()
            cache_file = Path(cache_dir) / f"{func.__name__}_{key_hash}.pkl"

            # Try to load from cache
            if cache_file.exists():
                with open(cache_file, "rb") as f:
                    try:
                        result = pickle.load(f)
                        return result
                    except (pickle.PickleError, EOFError) as exc:
                        pass  # Cache file corrupted, will recompute

            # Compute and cache if not found
            result = func(*args, **kwargs)
            with open(cache_file, "wb") as f:
                pickle.dump(result, f)

            return result

        # Maintain LRU behavior by tracking access times
        wrapper.cache_info = functools.lru_cache(maxsize=maxsize, typed=typed)(
            wrapper
        ).cache_info
        return wrapper

    return decorator


# comfy
def get_algorithm_params(func):
    """Returns a dictionary of parameter names and their default values."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


# super comfy
def param_config(st, params, param_values, algo=""):
    for param, default in params.items():
        if isinstance(default, bool):
            param_values[param] = st.sidebar.checkbox(param, default)
        elif isinstance(default, float) or isinstance(default, int):
            param_values[param] = st.sidebar.number_input(
                f"{param} {' of ' + algo if algo else ''}", value=default
            )
        elif isinstance(default, dict):
            # function definition
            st.sidebar.write(f"Configure behaviour for param `{param}`")
            algorithm_name = st.sidebar.selectbox(
                f"Select function to apply for `{param}`", list(default.keys())
            )
            algorithm = default[algorithm_name]
            st.sidebar.write(algorithm.__doc__)
            # lol, lmao
            sub_params = get_algorithm_params(algorithm)
            sub_values = {
                "function": algorithm,
            }

            param_config(
                st, sub_params, sub_values, algo=algo + f" {algorithm_name} for {param}"
            )

            param_values[param] = sub_values
        else:
            param_values[param] = st.sidebar.text_input(param, default)

    return param_values


# u-u-ultra comfy
def predicate_config(st, hypothesis, operators, thesis):
    param = st.sidebar.selectbox("If", hypothesis)
    operator = st.sidebar.selectbox("Operator", ["Any", ">", "<", "="])
    op_value = st.sidebar.number_input("Value", value=0.0)
    st.sidebar.write("Then:")

    prob = {}

    param_config(st, {"With probability": thesis}, prob)

    return {
        "param": param,
        "operator": operator,
        "value": op_value,
        "prob": prob,
    }


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
