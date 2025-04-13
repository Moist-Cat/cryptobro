import random
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import (
    norm,
    kstest,
    ttest_ind,
    ks_2samp,
    shapiro,
    lognorm,
    probplot,
    ttest_1samp,
    laplace,
)
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

# ----------------------
# Core Functions
# ----------------------


def load_csv():
    """Load CSV file and preprocess."""
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        uploaded_file = st.session_state.file

    if uploaded_file:
        # since we already might have already read it
        uploaded_file.seek(0)
        st.session_state.file = uploaded_file
        df = pd.read_csv(uploaded_file)
        df = df.dropna()
        if "Close" in df.columns:
            prices = df["Close"]
        elif "Open" in df.columns:
            prices = df["Open"]
        else:
            print("WARNING - Could not determine price column")
            prices = df.iloc[:, 0]  # Assume first column is price

        st.session_state.df = prices
        return prices.values
    return None


def test_zero_mean(prices, alpha=0.05):
    """
    Tests H₀: μ = 0 vs H₁: μ ≠ 0 for log-returns of prices.
    Assumes normality and n > 30 (uses z-test approximation).

    Parameters:
    - prices: Array of stock prices
    - alpha: Significance level

    Returns:
    - dict containing test results and conclusions
    - fig: Plot of the test visualization
    """
    # Calculate log-returns
    log_returns = np.log(prices[1:] / prices[:-1])
    n = len(log_returns)
    sample_mean = np.mean(log_returns)
    sample_std = np.std(log_returns, ddof=1)

    # Z-test (since n > 30)
    z_score = sample_mean / (sample_std / np.sqrt(n))
    p_value = 2 * (1 - norm.cdf(np.abs(z_score)))  # Two-tailed

    # Critical values
    z_critical = norm.ppf(1 - alpha / 2)
    ci_lower = sample_mean - z_critical * sample_std / np.sqrt(n)
    ci_upper = sample_mean + z_critical * sample_std / np.sqrt(n)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.linspace(-4, 4, 1000)
    ax.plot(x, norm.pdf(x), label="Standard Normal")
    ax.axvline(z_score, color="r", linestyle="--", label=f"Z-score = {z_score:.2f}")
    ax.fill_between(
        x,
        0,
        norm.pdf(x),
        where=(x < -z_critical) | (x > z_critical),
        color="red",
        alpha=0.2,
        label="Rejection Region",
    )
    ax.set_title(f"Z-Test for Zero Mean (α = {alpha})")
    ax.legend()

    return {
        "n": n,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "z_score": z_score,
        "p_value": p_value,
        "critical_value": z_critical,
        "ci_95%": (ci_lower, ci_upper),
        "reject_H0": p_value < alpha,
        "conclusion": (
            "Reject H₀ (mean ≠ 0)"
            if p_value < alpha
            else "Fail to reject H₀ (no evidence against μ=0)"
        ),
    }, fig


def verify_lognormal(prices):
    """
    Check if price ratios Y_n = S_n/S_{n-1} are lognormal.
    Returns:
    - ks_stat, p_value: Kolmogorov-Smirnov test results
    - shapiro_stat, shapiro_p: Shapiro-Wilk test results
    - fig_hist: Histogram plot
    - fig_qq: Q-Q plot
    """
    ratios = prices[1:] / prices[:-1]
    log_ratios = np.log(ratios)

    # Normality tests
    ks_stat, p_value = kstest(
        log_ratios, "norm", args=(log_ratios.mean(), log_ratios.std())
    )

    shapiro_stat, shapiro_p = (
        shapiro(log_ratios) if len(log_ratios) <= 5000 else (np.nan, np.nan)
    )

    # Create plots
    fig_hist, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(log_ratios, bins=200, density=True, alpha=0.6, label="Log Ratios")
    x = np.linspace(log_ratios.min(), log_ratios.max(), 100)
    ax1.plot(
        x, norm.pdf(x, log_ratios.mean(), log_ratios.std()), "r-", label="Fitted Normal"
    )
    ax1.set_title("Log-Ratios Distribution vs Normal Fit")
    ax1.legend()

    fig_qq, ax2 = plt.subplots(figsize=(10, 4))
    probplot(log_ratios, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot of Log-Ratios")
    ax2.get_lines()[0].set_markerfacecolor("b")
    ax2.get_lines()[0].set_markersize(4.0)
    ax2.get_lines()[1].set_color("r")
    ax2.get_lines()[1].set_linewidth(2.0)

    return {
        "ks_test": (ks_stat, p_value),
        "shapiro_test": (shapiro_stat, shapiro_p),
        "mean": log_ratios.mean(),
        "std": log_ratios.std(),
        "fig_hist": fig_hist,
        "fig_qq": fig_qq,
    }


# what the heck am I doing
def verify_loglaplace(prices):
    """
    Check if price ratios Y_n = S_n/S_{n-1} are loglaplace.
    What's that? I mean that the logarithm returns a laplace.
    Does that exist? Yep.
    https://en.wikipedia.org/wiki/Log-Laplace_distribution
    Returns:
    - ks_stat, p_value: Kolmogorov-Smirnov test results
    - shapiro_stat, shapiro_p: Shapiro-Wilk test results
    - fig_hist: Histogram plot
    - fig_qq: Q-Q plot
    """
    ratios = prices[1:] / prices[:-1]
    log_ratios = np.log(ratios)

    # Laplace tests
    mean, variance = laplace.fit(log_ratios)
    print(f"INFO - Fitted params {mean=} {variance=}")
    ks_stat, p_value = kstest(log_ratios, "laplace", args=laplace.fit(log_ratios))
    shapiro_stat, shapiro_p = 0, 0

    # Create plots
    fig_hist, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(log_ratios, bins=200, density=True, alpha=0.6, label="Log Ratios")
    x = np.linspace(log_ratios.min(), log_ratios.max(), 100)
    ax1.plot(x, laplace.pdf(x, *laplace.fit(log_ratios)), "r-", label="Fitted Laplace")
    ax1.set_title("Log-Ratios Distribution vs Laplace Fit")
    ax1.legend()

    fig_qq, ax2 = plt.subplots(figsize=(10, 4))
    probplot(log_ratios, dist="laplace", plot=ax2)
    ax2.set_title("Q-Q Plot of Log-Ratios")
    ax2.get_lines()[0].set_markerfacecolor("b")
    ax2.get_lines()[0].set_markersize(4.0)
    ax2.get_lines()[1].set_color("r")
    ax2.get_lines()[1].set_linewidth(2.0)

    return {
        "ks_test": (ks_stat, p_value),
        "shapiro_test": (shapiro_stat, shapiro_p),
        "mean": log_ratios.mean(),
        "std": log_ratios.std(),
        "fig_hist": fig_hist,
        "fig_qq": fig_qq,
    }


def estimate_parameters(prices):
    """Estimate mu and sigma from price data."""
    ratios = prices[1:] / prices[:-1]
    log_ratios = np.log(ratios)

    return laplace.fit(log_ratios)


def simulate_prices(S0, mu, sigma, T, N, num_paths):
    """Simulate stock prices using GBM."""
    dt = T / N
    prices = np.zeros((num_paths, N + 1))
    prices[:, 0] = S0
    for t in range(1, N + 1):
        # Z = np.random.normal(0, 1, num_paths)
        # prices[:, t] = prices[:, t - 1] * np.exp(mu * dt + sigma * np.sqrt(dt) * Z)
        mu = st.session_state.mu_hat
        sigma = st.session_state.sigma_hat
        prices[:, t] = prices[:, t - 1] * np.exp(
            np.random.laplace(mu, sigma, num_paths)
        )
    return prices


# ----------------------
# Policy Functions
# ----------------------


def alpha_based_policy(alpha):
    return "Early Exercise" if alpha > 0 else "Wait Until Expiry"


def policy_early_exercise(prices, strike_price):
    exercise_times = np.argmax(prices > strike_price, axis=1)
    exercise_values = np.array(
        [
            prices[i, t] - strike_price if t > 0 else 0
            for i, t in enumerate(exercise_times)
        ]
    )
    return exercise_values, exercise_times


def policy_wait_until_expiry(prices, strike_price):
    exercise_values = np.maximum(prices[:, -1] - strike_price, 0)
    exercise_times = np.where(exercise_values > 0, prices.shape[1] - 1, 0)
    return exercise_values, exercise_times


def policy_alpha_conditional(prices, strike_price, alpha):
    if alpha > 0:
        return policy_early_exercise(prices, strike_price)
    return policy_wait_until_expiry(prices, strike_price)


def stock_price_ci(S0, mu, sigma, days=5, alpha=0.05, n_sim=10000):
    """
    Constructs confidence intervals for future stock prices using lognormal distribution properties.

    Parameters:
    S0 (float): Current stock price
    mu (float): Annualized drift (expected return)
    sigma (float): Annualized volatility
    days (int): Time horizon in days
    alpha (float): Significance level (default 0.05 for 95% CI)
    n_sim (int): Number of simulations for Monte Carlo approach

    Returns:
    dict: Contains confidence intervals and other statistics
    plt.Figure: Visualization of the price distribution
    """
    # Convert annualized parameters to daily
    #t = days / 252  # Trading days convention
    t = 1
    mu_daily = mu * t
    sigma_daily = sigma * np.sqrt(t)

    # Analytical method (lognormal distribution)
    z = laplace.ppf(1 - alpha / 2)
    log_S = np.log(S0) + (mu - 0.5 * sigma**2) * t
    std_log_S = sigma * np.sqrt(t)

    lower_analytical = np.exp(log_S - z * std_log_S)
    upper_analytical = np.exp(log_S + z * std_log_S)

    # Monte Carlo simulation
    daily_returns = np.random.laplace(
        (mu - 0.5 * sigma**2) * t, sigma * np.sqrt(t), n_sim
    )
    future_prices = S0 * np.exp(daily_returns)

    lower_mc = np.percentile(future_prices, 100 * alpha / 2)
    upper_mc = np.percentile(future_prices, 100 * (1 - alpha / 2))

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(future_prices, bins=50, density=True, alpha=0.7)
    ax.axvline(lower_analytical, color="r", linestyle="--", label="Analytical CI")
    ax.axvline(upper_analytical, color="r", linestyle="--")
    ax.axvline(lower_mc, color="g", linestyle=":", label="Monte Carlo CI")
    ax.axvline(upper_mc, color="g", linestyle=":")
    ax.set_title(f"Stock Price Distribution After {days} Days")
    ax.set_xlabel("Price")
    ax.set_ylabel("Density")
    ax.legend()

    return {
        "current_price": S0,
        "days_ahead": days,
        "confidence_level": 1 - alpha,
        "analytical_ci": (lower_analytical, upper_analytical),
        "monte_carlo_ci": (lower_mc, upper_mc),
        "ci": (lower_mc, upper_mc),
        "mean_price": np.mean(future_prices),
        "median_price": np.median(future_prices),
        "probability_above_current": np.mean(future_prices > S0),
    }, fig


# ----------------------
# Evaluation Functions
# ----------------------


def evaluate_policies(prices, strike_price, alpha):
    policies = {
        "Early Exercise": policy_early_exercise(prices, strike_price),
        "Wait Until Expiry": policy_wait_until_expiry(prices, strike_price),
        "Alpha-Based": policy_alpha_conditional(prices, strike_price, alpha),
    }
    results = {}
    for name, (values, times) in policies.items():
        results[name] = {
            "Mean Payoff": np.mean(values),
            "Std Payoff": np.std(values),
            "Exercise Prob": np.mean(times > 0),
            "Mean Exercise Time": np.mean(times[times > 0]) if np.any(times > 0) else 0,
        }
    return pd.DataFrame(results).T


def compare_real_vs_simulated(
    real_prices, simulated_prices, real_strike, simulated_strike
):
    """Compare real vs simulated payoffs (MSE, stats)."""
    real_payoff = np.maximum((real_prices[-1] - real_strike) / real_strike, 0)
    simulated_payoffs = np.maximum(
        (simulated_prices[:, -1] - simulated_strike) / simulated_strike, 0
    )

    mse = mean_squared_error([real_payoff] * len(simulated_payoffs), simulated_payoffs)

    fig, ax = plt.subplots()
    ax.hist(simulated_payoffs, bins=200, alpha=0.6, label="Simulated Payoffs")
    ax.axhline(real_payoff, color="r", linestyle="--", label="Real Payoff")
    ax.set_title("Simulated vs Real Payoff")
    ax.legend()

    return {
        "MSE": mse,
        "Real Payoff": real_payoff,
        "Simulated Mean": np.mean(simulated_payoffs),
        "Simulated Std": np.std(simulated_payoffs),
    }, fig


def validate_ci_coverage(
    prices, mu, sigma, window_size=5, alpha=0.05, n_sim=1000, fun=stock_price_ci
):
    """
    Validates CI coverage by checking how often prices stay within predicted intervals.

    Parameters:
    prices (array): Historical price data
    mu (float): Annualized drift
    sigma (float): Annualized volatility
    window_size (int): Days ahead to predict (n in your algorithm)
    alpha (float): Significance level
    n_sim (int): Number of simulations for Monte Carlo CI

    Returns:
    dict: Test results including coverage score and detailed statistics
    """
    wins = 0
    total = 0
    violations = []
    ci_widths = []

    for i in tqdm(range(len(prices) - window_size)):
        S0 = prices[i]
        current_window = prices[i : i + window_size + 1]  # Include day 0

        res, fig = stock_price_ci(
            S0, mu, sigma, days=window_size, alpha=alpha, n_sim=n_sim
        )
        # we don't care about the figure
        plt.close()
        lower, upper = res["ci"]

        # Check if all future prices are within CI
        within_ci = True
        for price in current_window[1:]:
            if price < lower or price > upper and within_ci:
                within_ci = False
                violations.append(
                    {
                        "day": i,
                        "price": price,
                        "lower": lower,
                        "upper": upper,
                        "window": window_size,
                    }
                )
                break

        if within_ci:
            wins += 1
        total += 1

    coverage = wins / total
    expected = 1 - alpha

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Coverage plot
    ax1.axhline(expected, color="r", linestyle="--", label="Expected Coverage")
    ax1.bar(["Actual"], [coverage], label=f"Actual (n={total})")
    ax1.set_ylim(0, 1)
    ax1.set_title(f"CI Coverage ({window_size}-day windows)")
    ax1.legend()

    # Violation plot
    if violations:
        violation_days = [v["day"] for v in violations]
        ax2.hist(violation_days, bins=20)
        ax2.set_title("Distribution of CI Violations Over Time")
        ax2.set_xlabel("Day in Series")
    else:
        ax2.text(0.5, 0.5, "No Violations", ha="center", va="center")

    plt.tight_layout()

    return {
        "coverage_score": coverage,
        "expected_coverage": expected,
        "total_windows": total,
        "successful_windows": wins,
        "violation_rate": 1 - coverage,
        "violations": violations,
    }, fig


# ----------------------
# Plotting
# ---------------------


def plot_real_prices(
    real_prices, strike_price, exercise_day=None, name="Real Stock Price"
):
    """Plot real stock prices with strike price and exercise day (if applicable)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(real_prices, label=name, color="blue")
    ax.axhline(strike_price, color="red", linestyle="--", label="Strike Price")

    if exercise_day is not None and exercise_day < len(real_prices):
        ax.axvline(exercise_day, color="green", linestyle=":", label="Exercise Day")
        ax.scatter(
            exercise_day,
            real_prices[exercise_day],
            color="black",
            s=100,
            label=f"Exercise Price: {real_prices[exercise_day]:.2f}",
        )

    ax.set_title(f"{name} with Exercise Policy")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend()
    return fig


def plot_simulated_prices(
    simulated_prices, strike_price, expiration_day, exercise_days=None
):
    """Plot simulated stock price paths with strike and exercise days (if provided)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot all simulated paths (transparent lines)
    for path in simulated_prices:
        ax.plot(path, alpha=0.1, color="gray")

    # Highlight mean path
    mean_path = np.mean(simulated_prices, axis=0)
    ax.plot(mean_path, color="blue", label="Mean Simulated Price")

    # Key lines
    ax.axhline(strike_price, color="red", linestyle="--", label="Strike Price")
    ax.axvline(expiration_day, color="purple", linestyle="-.", label="Expiration Day")

    # Exercise markers (if provided)
    if exercise_days is not None:
        for i, day in enumerate(exercise_days):
            if day > 0:  # Only plot if exercised
                ax.scatter(
                    day,
                    simulated_prices[i, day],
                    color="black",
                    s=50,
                    alpha=0.5,
                    label=f"Exercise Day (Path {i+1})" if i == 0 else "",
                )

    ax.set_title("Simulated Stock Paths with Exercise Policy")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend()
    return fig


# ----------------------
# Streamlit App
# ----------------------


def main():
    st.title("Option Exercise Policy Analyzer")
    st.sidebar.header("Navigation")
    section = st.sidebar.radio("Go to", ["CSV Validation", "Policy Evaluation"])

    if section == "CSV Validation":
        st.header("1. CSV Validation & Distribution Test")
        prices = load_csv()
        st.session_state.prices = prices

        if prices is not None:
            price_sample = st.session_state.df.sample(1000).sort_index().values
            with st.expander("Step 1: Estimate Parameters", expanded=True):
                mu_hat, sigma_hat = estimate_parameters(price_sample)
                st.markdown(
                    f"""
                **Estimated Parameters**:  
                - $\hat{{\mu}}$ = {mu_hat:.6f}  
                - $\hat{{\sigma}}$ = {sigma_hat:.6f}  
                - $\hat{{\\alpha}}$ = {mu_hat + 0.5 * sigma_hat**2:.6f}
                """
                )
                st.session_state.mu_hat = mu_hat
                st.session_state.sigma_hat = sigma_hat

            with st.expander(
                "Step 2: Verify distribution with new parameters", expanded=True
            ):
                # res = verify_lognormal(price_sample)
                res = verify_loglaplace(price_sample)
                ks_stat, p_value = res["ks_test"]
                shapiro_stat, shapiro_p = res["shapiro_test"]
                fig = res["fig_hist"]
                qq = res["fig_qq"]

                res, zero_fig = test_zero_mean(price_sample)
                zero_p = res["p_value"]
                # literally me
                zero_int = res["ci_95%"]
                zero_concl = res["conclusion"]

                st.pyplot(fig)
                st.pyplot(qq)

                st.markdown(
                    f"""
                **Kolmogorov-Smirnov Test**:  
                - Statistic = {ks_stat:.4f}  
                - p-value = {p_value:.4f}  
                - **Conclusion**: {'It fits the distribution (fail to reject H₀)' if p_value > 0.05 else 'It doesn not fit the distribution (reject H₀)'}

                **Shapiro-Wilk Test**:
                - Statistic = {shapiro_stat:.4f}  
                - p-value = {shapiro_p:.4f}  
                - **Conclusion**: {'It fits the distribution (fail to reject H₀)' if shapiro_p > 0.05 else 'It does not fit the distribution (reject H₀)'}

                **Null Hyphotesis: The mean of the distribution is zero (assumes normality)**:
                - Confidence interval = {zero_int} 
                - p-value = {zero_p:.4f}  
                - **Conclusion**: {zero_concl}
                """
                )

                st.pyplot(zero_fig)

    elif section == "Policy Evaluation":
        st.header("2. Policy Evaluation")

        # Parameters (use estimated or manual)
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.number_input("Initial Price (S₀)", value=100.0)
            strike_price = st.number_input("Strike Price", value=105.0)
            T = st.number_input("Time to Expiry (T)", value=30)
        with col2:
            mu = st.number_input(
                "Drift (μ)",
                value=getattr(st.session_state, "mu_hat", 0.0005),
                step=0.00001,
                format="%0.5f",
            )
            sigma = st.number_input(
                "Volatility (σ)",
                value=getattr(st.session_state, "sigma_hat", 0.002),
                step=0.00001,
                format="%0.5f",
            )
            num_paths = st.number_input("Paths", 1000)

        alpha = mu + 0.5 * sigma**2
        st.markdown(
            f"**$\\alpha = \\mu + \\sigma^2/2 = {alpha:.4f}$** → **Policy: {alpha_based_policy(alpha)}**"
        )

        if st.button("Run Policy Simulation"):
            with st.expander("Simulation Results", expanded=True):
                simulated_prices = simulate_prices(
                    S0, mu, sigma, T, T, num_paths
                )  # T steps = T days
                results = evaluate_policies(simulated_prices, strike_price, alpha)
                st.dataframe(results.style.format("{:.2f}"))

            real_prices = st.session_state.prices
            real_prices = real_prices[2000 : 2000 + T]
            print("INFO - Real prices", real_prices)
            real_strike = strike_price / S0 * real_prices[0]
            with st.expander("Simulation Plot", expanded=True):
                fig_real = plot_real_prices(real_prices, real_strike, exercise_day=None)
                fig_sim = plot_simulated_prices(
                    simulated_prices, strike_price, expiration_day=T, exercise_days=None
                )

                st.pyplot(fig_real)
                st.pyplot(fig_sim)

            # If CSV was loaded, compare real vs simulated
            if "prices" in st.session_state and st.session_state.prices is not None:
                comparison, fig = compare_real_vs_simulated(
                    real_prices,
                    simulated_prices,
                    real_strike,
                    strike_price,
                )

                with st.expander("Payoff plot", expanded=True):
                    st.pyplot(fig)
                with st.expander("Payoff stats", expanded=True):
                    st.markdown(
                        f"""
                    **Real vs Simulated**:  
                    - Real Payoff = {comparison['Real Payoff']:.2f}  
                    - Simulated Mean = {comparison['Simulated Mean']:.2f}  
                    - MSE = {comparison['MSE']:.2f}
                    """
                    )

                res, fig = stock_price_ci(S0, mu, sigma, T, 0.05, num_paths)

                confidence = res["confidence_level"]
                anal_low, anal_high = res["analytical_ci"]
                anal_low = round(float(anal_low), 2)
                anal_high = round(float(anal_high), 2)

                mc_low, mc_high = res["monte_carlo_ci"]
                mc_low = round(float(mc_low), 2)
                mc_high = round(float(mc_high), 2)

                above = res["probability_above_current"]

                with st.expander("CI Plot", expanded=True):
                    st.pyplot(fig)
                with st.expander("CI Stats", expanded=True):
                    st.markdown(
                        f"""
                    **Expected interval**:  
                    - Confidence = {confidence}  
                    - lower, upper (M-C) = {mc_low, mc_high}
                    - lower, upper (Analytic) = {anal_low, anal_high}
                    - above  = {above}
                    """
                    )

        if st.button("Validate cofidence interval"):
            res, fig = validate_ci_coverage(
                #st.session_state.prices,
                st.session_state.prices,
                mu,
                sigma,
                window_size=5,
                alpha=0.05,
                #alpha=0.1,
                n_sim=num_paths,
                fun=stock_price_ci,
            )

            with st.expander("Expectation VS Reality", expanded=True):
                st.pyplot(fig)

            coverage = res["coverage_score"]
            expected = res["expected_coverage"]
            violations = res["violations"]
            violation_rate = res["violation_rate"]
            total = res["total_windows"]
            wins = res["successful_windows"]

            with st.expander("CI Validation Stats", expanded=True):
                st.markdown(
                    f"""
                **Confidence interval validation**:  
                - Coverage = {coverage} 
                - Expected = {expected} 
                - Violation rate = {violation_rate} 
                - Number of window analysed = {total} 
                - Wins = {wins}
                """
                )


if __name__ == "__main__":
    if "prices" not in st.session_state:
        st.session_state.df = None
        st.session_state.prices = None
        st.session_state.file = None
        st.session_state.mu_hat = 0
        st.session_state.sigma_hat = 1
    main()
