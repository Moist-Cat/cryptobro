import inspect
import functools
import pickle
import os
from pathlib import Path
import hashlib

#from scipy.stats import norm as laplace
from scipy.stats import laplace
from sklearn.decomposition import PCA, TruncatedSVD
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
import numpy as np
from tqdm import tqdm

from itertools import groupby

# autocorrelation
def find_clusters(violations, gap=2):
    """Group consecutive violations within 'gap' days"""
    clusters = []
    for k, g in groupby(enumerate(violations), lambda x: x[0]-x[1]['day']):
        group = list(g)
        if len(group) > 1:  # At least 2 consecutive violations
            start_day = group[0][1]['day']
            end_day = group[-1][1]['day']
            clusters.append((start_day, end_day))
    return cluster

def test_autocorrelation(returns, lags=15):
    """Test for autocorrelation in log returns"""
    lb_test = acorr_ljungbox(returns, lags=lags, return_df=True)
    return lb_test[['lb_stat', 'lb_pvalue']]

def rolling_autocorrelation(returns, window=30, lag=1):
    """Rolling autocorrelation at specified lag"""
    autocorrs = []
    for i in range(len(returns) - window):
        window_returns = returns[i:i+window]
        acf_vals = acf(window_returns, nlags=lag, fft=False)
        autocorrs.append(acf_vals[lag])
    return np.array(autocorrs)

def general_pca(directory_path):
    """
    Processes CSV files in a directory, performs PCA analysis, and returns Streamlit-ready results
    Returns: dict containing {
        'pca_model': PCA object,
        'transformed_data': ndarray,
        'stats': dict,
        'figures': dict,
        'matrix_shape': tuple,
        'files_used': list
    }
    """
    # Load and filter CSV files
    file_data = []
    valid_files = []

    # DAYS = 100
    DAYS = 350

    for filename in os.listdir(directory_path):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, "r") as f:
                data = load_csv(f)
                if len(data) >= DAYS:
                    truncated_prices = data[-DAYS:]
                    ratios = truncated_prices[1:] / truncated_prices[:-1]
                    log_ratios = np.log(ratios)
                    file_data.append(log_ratios)
                    valid_files.append(filename)

    if not file_data:
        return {"error": f"No valid files found with ≥ {DAYS} rows"}

    # Create matrix and handle missing values
    matrix = np.vstack(file_data)
    # matrix = SimpleImputer(strategy='mean').fit_transform(matrix)
    # matrix = SimpleImputer(strategy='median').fit_transform(matrix)

    # Standardize data
    # scaler = StandardScaler()
    # scaled_data = scaler.fit_transform(matrix)
    scaled_data = matrix

    # Perform PCA
    pca = PCA(n_components=0.95)  # Keep 95% variance
    # pca = TruncatedSVD(n_components=3)  # Keep 95% variance
    transformed = pca.fit_transform(scaled_data)

    # Calculate statistics
    stats = {
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
        "component_loadings": pca.components_.tolist(),
        "n_components": pca.n_components_,
        #'n_components': 3,
        "mean_per_component": np.mean(transformed, axis=0).tolist(),
        "std_per_component": np.std(transformed, axis=0).tolist(),
    }

    # Create visualizations
    return {
        "pca_model": pca,
        "transformed_data": transformed,
        "stats": stats,
        "matrix_shape": matrix.shape,
        "files_used": valid_files,
    }

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

def boot_ci(S0, mu, sigma, days=5, alpha=0.05, n_sim=10000):
    """
    Simulate price paths and compute bootstrapped percentiles

    Args:
        S0: Initial price
        mu: Drift parameter
        sigma: Scale parameter (b for Laplace)
        N: Number of time steps
        num_paths: Number of paths to simulate
        n_bootstrap: Number of bootstrap samples
        alpha: Confidence level (e.g., 0.05 for 95% CI)

    Returns:
        dict: {
            'paths': simulated paths (num_paths × N),
            'percentiles': bootstrapped percentiles,
            'ci': confidence intervals for percentiles
        }
    """
    # 1. Simulate base paths (Laplace-distributed log returns)
    #future_prices = simulate_prices(S0, mu, sigma, days, n_sim)[
    SIZE = 100
    future_prices = simulate_prices(S0, mu, sigma, days, SIZE)[
        :, -1
    ]

    # 2. Bootstrap percentiles
    #bootstrap_percentiles = np.zeros((n_sim, 3))  # For 5%, 50%, 95%
    bootstrap_percentiles = np.zeros((n_sim, 3))  # For 5%, 50%, 95%

    for i in range(n_sim):
        # Resample paths with replacement
        idx = np.random.choice(SIZE, size=SIZE, replace=True)
        resampled = future_prices[idx]

        # Compute percentiles (using symmetry assumption)
        bootstrap_percentiles[i][0] = np.percentile(resampled, 100*alpha/2, axis=0)  # Lower
        bootstrap_percentiles[i][1] = np.median(resampled, axis=0)                   # Median
        bootstrap_percentiles[i][2] = np.percentile(resampled, 100*(1-alpha/2), axis=0)  # Upper

    # 3. Compute mean and CI of percentiles
    # bold assumption of normality
    percentiles = {
        'lower': np.mean(bootstrap_percentiles[:,0], axis=0),
        'median': np.mean(bootstrap_percentiles[:,1], axis=0),
        'upper': np.mean(bootstrap_percentiles[:,2], axis=0)
    }

    ci = {
        'lower': np.percentile(bootstrap_percentiles[:,0], [2.5, 97.5], axis=0),
        'median': np.percentile(bootstrap_percentiles[:,1], [2.5, 97.5], axis=0),
        'upper': np.percentile(bootstrap_percentiles[:,2], [2.5, 97.5], axis=0)
    }

    return future_prices, percentiles["lower"], percentiles["upper"]



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

    lower_mc = np.percentile(future_prices, 100 * alpha / 2, method="median_unbiased")
    upper_mc = np.percentile(
        future_prices, 100 * (1 - alpha / 2), method="median_unbiased"
    )

    return future_prices, lower_mc, upper_mc

if __name__ == "__main__":
    mc_ci(100, 0, 0.03, days=7, alpha=0.1, n_sim=10000)
    boot_ci(100, 0, 0.03, days=7, alpha=0.1, n_sim=10000)
