import random
import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import estimate_parameters, simulate_prices, mc_ci, calculate_rsi
from main import load_csv
from scipy.stats import chi2_contingency

from meta import _configure_algorithm, DB_DIR

# Core functional components
def generate_datasets(real_prices: list, n_sim=1000):
    """Generate synthetic datasets from real data parameters"""

    synthetic = []
    for prices in real_prices:
        # Random Laplacian walks
        synthetic.append(simulate_prices(prices, len(prices), 1)[0])

    mixed = []
    for idx in range(0, len(real_prices), 2):
        prices = real_prices[idx]

        mixed.append(simulate_prices(prices, len(prices), 1)[0])

    mixed.extend(random.choices(real_prices, k=len(real_prices) - len(mixed)))
    return synthetic, mixed


def trend_threshold(
    prices, risk=0.66, window_size=10, before_window=True, days=15, upper=0.9, lower=0.1
):
    """
    1. Calculate 10-day moving average
    2. See if it's (almost) outside bounds given an alhpa

    The size of the window and wheter to check before or after can be configured.
    """
    window = (
        sum(prices[(len(prices) - 1) - window_size : len(prices) - 1]) / window_size
    )

    _, lower_mc, higher_mc = mc_ci(
        prices, 0, len(prices) - 1, days=window_size, alpha=risk, n_sim=1000
    )

    A = lower_mc
    B = higher_mc
    C = window
    result = (C - A) / (B - A)

    action = 0

    if result > upper:
        action = -1
    elif result < lower:
        action = 1

    return action


def trend_rsi(
    prices,
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
    rsi_value = calculate_rsi(prices, len(prices) - 1, days)

    if rsi_value is None:  # Not enough data
        return 0

    # Determine action
    action = 0
    if rsi_value >= overbought:
        action = -1
    elif rsi_value <= oversold:
        action = 1

    return action


# buy and hold
def trend_hodl(prices):
    """
    Simply buy the asset at any given opportunity.
    Works with some categories of assets.
    """
    return 1


def trend_monkey(prices):
    """
    Gives a random answer. $E[x] = 0$ or, in this case
    the precision should be `50%`.
    """
    return random.choice([-1, 1])


TREND = [trend_hodl, trend_monkey, trend_threshold, trend_rsi]


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


def make_samples(
    prices_list: list,
    test_size=31,
):
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
        validation.append(prices[start : end + test_size])

    return test, validation


def make_sample(prices_list):
    prices_len = len(prices)
    start = random.randint(0, prices_len - 500)
    end = random.randint(start + 240, prices_len - test_size)


def show_trends_with_metrics(prices: list, trend_detection_algo, lookahead=5):
    test, validation = make_samples(prices, lookahead)
    trends = [trend_detection_algo(p) for p in test]
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
    n_runs = st.sidebar.number_input("Number of runs", 1, 10000, 10)
    lookahead = st.sidebar.number_input("Days ahead to verify", 5, 1000, 5)

    detect_trend = _configure_algorithm(st, TREND, "Trend-detection algorithm")

    if st.button("Run Experiments"):
        results = validate_trend_hyphotesis(
            real_prices, n_runs, lookahead, detect_trend
        )
        display_final_results(results, n_runs)


def validate_trend_hyphotesis(real_prices, n_runs, lookahead, trend_detection_algo):
    """Run multiple experiments and aggregate results"""
    # Initialize accumulators
    agg_metrics = {
        "real": {"total": 0, "guesses": 0, "failures": 0},
        # "synth": {"total": 0, "guesses": 0, "failures": 0},
        # "mixed": {"total": 0, "guesses": 0, "failures": 0},
    }

    price_change = []
    lap_sign = []

    progress_bar = st.progress(0)
    # bootstrapping!
    for run in range(n_runs):
        # Generate new synthetic data each run
        # synthetic, mixed = generate_datasets(
        #    random.choices(real_prices, k=len(real_prices))
        # )

        # Calculate metrics
        r = show_trends_with_metrics(real_prices, trend_detection_algo, lookahead)
        # s = show_trends_with_metrics(synthetic, trend_detection_algo, lookahead)
        # m = show_trends_with_metrics(mixed, trend_detection_algo, lookahead)

        # Accumulate results
        for key in ["total", "guesses", "failures"]:
            agg_metrics["real"][key] += r[key]
            # agg_metrics["synth"][key] += s[key]
            # agg_metrics["mixed"][key] += m[key]

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
    # datasets = ["real", "synth", "mixed"]
    datasets = [
        "real",
    ]

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

    less_c = sum(
        1 for i in range(len(change)) if lap_mu[i] == -1 and lap_mu[i] == change[i]
    )
    less_i = sum(
        1 for i in range(len(change)) if lap_mu[i] == -1 and lap_mu[i] != change[i]
    )

    more_c = sum(
        1 for i in range(len(change)) if lap_mu[i] == 1 and lap_mu[i] == change[i]
    )
    more_i = sum(
        1 for i in range(len(change)) if lap_mu[i] == 1 and lap_mu[i] != change[i]
    )

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
            """.format(
                real_total=results["real"]["total"],
                real_guess=results["real"]["guesses"],
                real_fail=results["real"]["failures"],
                real_prec=results["real"]["failures"] / results["real"]["guesses"],
                real_rec=results["real"]["guesses"] / results["real"]["total"],
                # synth_total=results["synth"]["total"],
                # synth_guess=results["synth"]["guesses"],
                # synth_fail=results["synth"]["failures"],
                # synth_prec=results["synth"]["failures"] / results["synth"]["guesses"],
                # synth_rec=results["synth"]["guesses"] / results["synth"]["total"],
                # mixed_total=results["mixed"]["total"],
                # mixed_guess=results["mixed"]["guesses"],
                # mixed_fail=results["mixed"]["failures"],
                # mixed_prec=results["mixed"]["failures"] / results["mixed"]["guesses"],
                # mixed_rec=results["mixed"]["guesses"] / results["mixed"]["total"],
            )
        )


if __name__ == "__main__":
    main()
