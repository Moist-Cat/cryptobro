import random
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import estimate_parameters, simulate_prices, mc_ci
from main import load_csv
from scipy.stats import chi2_contingency

DB_DIR = Path("./datasets")

# Core functional components
def generate_datasets(real_prices: list, n_sim=1000):
    """Generate synthetic datasets from real data parameters"""

    synthetic = []
    for prices in real_prices:
        mu, sigma = estimate_parameters(prices)

        # Random Laplacian walks
        synthetic.append(
            simulate_prices(random.choice(prices), mu, sigma, len(prices), 1)[0]
        )

    mixed = []
    for idx in range(0, len(real_prices), 2):
        prices = real_prices[idx]
        mu, sigma = estimate_parameters(prices)

        mixed.append(
            simulate_prices(random.choice(prices), mu, sigma, len(prices), 1)[0]
        )

    mixed.extend(random.choices(real_prices, k=len(real_prices) - len(mixed)))
    return synthetic, mixed


def _make_samples(prices_list: list, test_size=31):
    """
    Returns two chunks. One for testing and another for validation.
    The second one is slightly largers. This can be used to evaluate the predictive
    ability of the model.
    """
    test = []
    validation = []
    for prices in prices_list:
        prices_len = len(prices)
        start = random.randint(0, prices_len - 500)
        end = random.randint(start + 240, prices_len - test_size)

        test.append(prices[start:end])
        validation.append(prices[start:end + test_size])

    return test, validation


# mu
def infer_trend(trend_vector):
    positive = [1] * len(trend_vector)
    negative = [-1] * len(trend_vector)

    if trend_vector == positive:
        return 1
    #elif trend_vector == negative:
    #    return -1
    return 0

# buy and hold
def infer_trend(trend_vector):
    return 1

def infer_trend(trend_vector):
    return random.choice([-1, 1])

def detect_trends(prices_list, window_sizes):
    """Core trend detection algorithm"""
    trends = []
    for prices in prices_list:
        trend_vector = [0] * len(window_sizes)
        parameter_vector = []
        for i, ws in enumerate(window_sizes):
            mu, sigma = estimate_parameters(prices[-ws:])
            parameter_vector.append(mu)

            trend_vector[i] = np.sign(mu)
        trends.append(infer_trend(trend_vector))
    return trends


def validate_trends(test, validation, trends):
    failures = 0
    guesses = 0
    for i in range(len(validation)):
        sample = validation[i]
        sample_test = test[i]
        trend = trends[i]
        if not trend:
            continue
        failures += int(np.sign(sample[-1] - sample_test[-1]) != np.sign(trend))
        guesses += 1

    return len(validation), guesses, failures


def show_trends_with_metrics(prices: list, window_sizes: list):
    test, validation = _make_samples(prices, 200)
    trends = detect_trends(test, window_sizes)
    metrics = validate_trends(test, validation, trends)

    change = []
    sign = []
    for i in range(len(validation)):
        t = test[i]
        v = validation[i]

        tr = trends[i]

        change.append(v[-1] - t[-1])
        sign.append(tr)

    return {
        "total": metrics[0],
        "guesses": metrics[1],
        "failures": metrics[2],
        "raw": (change, sign),
    }


def main():
    st.title("Laplacian Trend Hypothesis Verification")

    # Data loading
    real_prices = []
    for filename in DB_DIR.glob("*.csv"):
        with open(filename) as f:
            real_prices.append(load_csv(f))

    # Configuration
    window_sizes = st.sidebar.multiselect(
        "Window sizes",
        [480, 240, 120, 60, 30],
        default=[
            240,
        ],
    )
    n_runs = st.sidebar.number_input("Number of runs", 1, 10000, 10)

    if st.button("Run Experiments"):
        results = validate_trend_hyphotesis(real_prices, window_sizes, n_runs)
        display_final_results(results, n_runs)


def validate_trend_hyphotesis(real_prices, window_sizes, n_runs):
    """Run multiple experiments and aggregate results"""
    # Initialize accumulators
    agg_metrics = {
        "real": {"total": 0, "guesses": 0, "failures": 0},
        "synth": {"total": 0, "guesses": 0, "failures": 0},
        "mixed": {"total": 0, "guesses": 0, "failures": 0},
    }

    price_change = []
    lap_sign = []

    progress_bar = st.progress(0)
    # bootstrapping!
    for run in range(n_runs):
        # Generate new synthetic data each run
        synthetic, mixed = generate_datasets(
            random.choices(real_prices, k=len(real_prices))
        )

        # Calculate metrics
        r = show_trends_with_metrics(real_prices, window_sizes)
        s = show_trends_with_metrics(synthetic, window_sizes)
        m = show_trends_with_metrics(mixed, window_sizes)

        # Accumulate results
        for key in ["total", "guesses", "failures"]:
            agg_metrics["real"][key] += r[key]
            agg_metrics["synth"][key] += s[key]
            agg_metrics["mixed"][key] += m[key]

        progress_bar.progress((run + 1) / n_runs)

        price_change.extend(r["raw"][0])
        lap_sign.extend(r["raw"][1])

    # Average metrics
    for category in agg_metrics:
        for key in agg_metrics[category]:
            agg_metrics[category][key] /= n_runs
        if agg_metrics[category]["guesses"] == 0:
            agg_metrics[category]["guesses"] = -1
            agg_metrics[category]["failures"] = -1

    agg_metrics["raw"] = (price_change, lap_sign)

    return agg_metrics


def display_final_results(results, n_runs):
    """Display aggregated metrics"""
    st.subheader(f"Final Results ({n_runs} runs average)")

    # col1, col2, col3 = st.columns(3)
    cols = st.columns(3)
    datasets = ["real", "synth", "mixed"]

    for index, dataset in enumerate(datasets):
        with cols[index]:
            st.markdown(f"### {dataset.capitalize()} Data")
            recall = results[dataset]["guesses"] / results[dataset]["total"]
            precision = (results[dataset]["guesses"] - results[dataset]["failures"]) / (
                results[dataset]["guesses"]
            )
            st.metric("Precision", f"{precision:.2%}")
            st.metric("Recall", f"{recall:.2%}")

    change, lap_mu = results["raw"]

    change = [np.sign(c) for c in change]

    less_c = sum(1 for i in range(len(change)) if lap_mu[i] == -1 and lap_mu[i] == change[i])
    less_i = sum(1 for i in range(len(change)) if lap_mu[i] == -1 and lap_mu[i] != change[i])

    more_c = sum(1 for i in range(len(change)) if lap_mu[i] == 1 and lap_mu[i] == change[i])
    more_i = sum(1 for i in range(len(change)) if lap_mu[i] == 1 and lap_mu[i] != change[i])

    print(f"{less_c=} {less_i=} {more_c=} {more_i=}")

    cross = pd.crosstab(change, lap_mu)

    cont_table = chi2_contingency(cross)

    st.write("**Pearson's chi-squared test**")

    st.dataframe(cont_table)
    st.dataframe(cross)

    # Raw data display
    with st.expander("Detailed Metrics"):
        st.write(
            """
            | Dataset   | Total | Guesses | Failures | Precision | Recall |
            |-----------|-------|---------|----------|-----------|--------|
            | Real      | {real_total:.1f} | {real_guess:.1f} | {real_fail:.1f} | {real_prec:.2%} | {real_rec:.2%} |
            | Synthetic | {synth_total:.1f} | {synth_guess:.1f} | {synth_fail:.1f} | {synth_prec:.2%} | {synth_rec:.2%} |
            | Mixed     | {mixed_total:.1f} | {mixed_guess:.1f} | {mixed_fail:.1f} | {mixed_prec:.2%} | {mixed_rec:.2%} |
            """.format(
                real_total=results["real"]["total"],
                real_guess=results["real"]["guesses"],
                real_fail=results["real"]["failures"],
                real_prec=results["real"]["failures"] / results["real"]["guesses"],
                real_rec=results["real"]["guesses"] / results["real"]["total"],
                synth_total=results["synth"]["total"],
                synth_guess=results["synth"]["guesses"],
                synth_fail=results["synth"]["failures"],
                synth_prec=results["synth"]["failures"] / results["synth"]["guesses"],
                synth_rec=results["synth"]["guesses"] / results["synth"]["total"],
                mixed_total=results["mixed"]["total"],
                mixed_guess=results["mixed"]["guesses"],
                mixed_fail=results["mixed"]["failures"],
                mixed_prec=results["mixed"]["failures"] / results["mixed"]["guesses"],
                mixed_rec=results["mixed"]["guesses"] / results["mixed"]["total"],
            )
        )


if __name__ == "__main__":
    main()
