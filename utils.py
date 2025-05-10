from statsmodels.tsa.api import ExponentialSmoothing
import os

from scipy.stats import laplace, gennorm
from sklearn.decomposition import PCA, TruncatedSVD
import numpy as np
from tqdm import tqdm

DIST = gennorm
#DIST = laplace
DIST_NAME = "gennorm"
#DIST_NAME = "laplace"

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


# pca
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


def estimate_parameters(prices):
    """Estimate mu and sigma from price data."""
    ratios = prices[1:] / prices[:-1]
    log_ratios = np.log(ratios)

    return DIST.fit(log_ratios)


def simulate_prices(prices, N, num_paths=1, start=0, end=None):
    """
    Simulate $N$ days starting from $S_0$. Assumes each
    ratio distributes Laplace with parameters $mu$ and $sigma$.
    """
    if not end:
        end = len(prices) - 1
    S0 = prices[end]
    if start > 45:
        args = estimate_parameters(prices[start:end+1])
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


def mc_ci(prices, start, end, days=5, alpha=0.05, n_sim=10000):
    """
    Generate confidence intervals with a Monte Carlo simulation.
    """
    # Notice we take into account each day!
    paths = simulate_prices(prices, days, n_sim, start=start, end=end)

    path_min = np.min(paths, axis=1)
    path_max = np.max(paths, axis=1)

    # Find tightest [L, U] that contains (1-alpha) paths
    L = np.percentile(path_min, 100 * alpha / 2)
    U = np.percentile(path_max, 100 * (1 - alpha / 2))

    return paths[:, -1], L, U


if __name__ == "__main__":
    mc_ci(100, 0, 0.03, days=7, alpha=0.1, n_sim=10000)
    boot_ci(100, 0, 0.03, days=7, alpha=0.1, n_sim=10000)
