from collections import Counter
import concurrent.futures

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

import policy
from utils import estimate_parameters, simulate_prices, mc_ci

# ----------------------
# Core Functions
# ----------------------


def load_csv():
    """Load CSV file and preprocess."""
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded_file:
        uploaded_file = st.session_state.file

    if uploaded_file:
        # since we might have read it already
        uploaded_file.seek(0)
        st.session_state.file = uploaded_file
        df = pd.read_csv(uploaded_file)
        df = df.dropna()

        valid_columns = ["Close", "Open", "Prices", "Price"]
        for col in valid_columns:
            if col in df.columns:
                prices = df[col]
                break
        else:
            print("WARNING - Could not determine price column")
            prices = df.iloc[:, 0]  # Assume first column is price (bad assumption)

        st.session_state.df = prices
        return prices.values
    return None


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
        "mean": log_ratios.mean(),
        "std": log_ratios.std(),
        "fig_hist": fig_hist,
        "fig_qq": fig_qq,
    }


# ----------------------
# Policy Functions
# ----------------------
def stock_price_ci(S0, mu, sigma, days=5, alpha=0.05, n_sim=10000):
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
    # NOTE More research is needed since the sum of exp distributes
    # Gamma.

    future_prices, lower_mc, upper_mc = mc_ci(S0, mu, sigma, days, alpha, n_sim)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(future_prices, bins=50, density=True, alpha=0.7)
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
        "monte_carlo_ci": (lower_mc, upper_mc),
        "ci": (lower_mc, upper_mc),
        "mean_price": np.mean(future_prices),
        "median_price": np.median(future_prices),
        "probability_above_current": np.mean(future_prices > S0),
    }, fig


# ----------------------
# Evaluation Functions
# ----------------------
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

        res, fig = fun(S0, mu, sigma, days=window_size, alpha=alpha, n_sim=n_sim)
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
# Streamlit App
# ----------------------


def _show_policy(start_state, state, logs, prices):
    debt = logs["debt"]
    actions = logs["action"]
    c = Counter(actions)
    buy, sell, wait = c[1], c[-1], c[0]

    net_worth = policy._net_worth(prices, state)

    with st.expander(f"Results for policy", expanded=True):
        st.markdown(
            f"""
        **Balance**
        - Result:  {'In debt!' if net_worth < 0 else 'Good'}
        - Net Worth: {net_worth}
        - Days in debt: {len(debt)}
        - Worst debt: {min(debt) if debt else 0}
        
        **Capital**
        - Initial Capital: {start_state['capital']}
        - Final Capital: {state['capital']}
        - Profit: {state['capital'] - start_state['capital']}

        **Assets**
        - Initial Assets: {start_state['stock']}
        - Final Assets: {state['stock']}
        - Difference: {state['stock'] - start_state['stock']}

        **Actions**
        - Buy: {buy}
        - Sell: {sell}
        - Wait: {wait}

        **Risk**
        - Risk {state['risk']}
        """
        )


def main():
    st.title("Option Exercise Policy Analyzer")
    st.sidebar.header("Navigation")
    section = st.sidebar.radio(
        "Go to",
        ["CSV Validation", "CI Evaluation", "Policy Experimentation", "Analyse Market"],
    )

    if section == "CSV Validation":
        st.header("1. CSV Validation & Distribution Test")
        prices = load_csv()
        st.session_state.prices = prices

        if prices is not None:
            price_sample = (
                st.session_state.df.sample(min(1000, len(prices) // 2))
                .sort_index()
                .values
            )
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
                res = verify_loglaplace(price_sample)
                ks_stat, p_value = res["ks_test"]
                fig = res["fig_hist"]
                qq = res["fig_qq"]

                st.pyplot(fig)
                st.pyplot(qq)

                st.markdown(
                    f"""
                **Kolmogorov-Smirnov Test**:  
                - Statistic = {ks_stat:.4f}  
                - p-value = {p_value:.4f}  
                - **Conclusion**: {'It fits the distribution (fail to reject H₀)' if p_value > 0.05 else 'It doesn not fit the distribution (reject H₀)'}
                """
                )

    elif section == "CI Evaluation":
        st.header("2. CI Evaluation")

        # Parameters (use estimated or manual)
        col1, col2 = st.columns(2)
        with col1:
            S0 = st.number_input("Initial Price (S₀)", value=100.0)
            T = st.number_input("Days (T)", value=5)
            aleph = st.number_input("Error (alpha)", value=0.1)
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

        if st.button("Run CI Simulation"):
            if "prices" in st.session_state and st.session_state.prices is not None:
                res, fig = stock_price_ci(S0, mu, sigma, T, aleph, num_paths)

                confidence = res["confidence_level"]

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
                    - above  = {above}
                    """
                    )

        if st.button("Validate cofidence interval"):
            res, fig = validate_ci_coverage(
                # st.session_state.prices,
                st.session_state.prices[:100],
                mu,
                sigma,
                window_size=T,
                alpha=aleph,
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
                - Number of windows analysed = {total} 
                - Wins = {wins}
                """
                )
    elif section == "Policy Experimentation":
        prices = st.session_state.prices
        selection = st.sidebar.radio("Select policy", policy.__all__)

        mu = st.session_state.mu_hat
        sigma = st.session_state.sigma_hat

        selected_policy = getattr(policy, selection)

        start_state = policy.STATE.copy()

        initial_capital = st.number_input("Capital", value=0)
        initial_assets = st.number_input("Assets", value=0)
        risk = st.number_input("Risk", value=0.05)

        start_state["capital"] = initial_capital
        start_state["assets"] = initial_assets

        best_risk = risk
        forecast = selected_policy(prices, len(prices) - 1, mu, sigma, best_risk)
        if forecast > 0:
            forecast = "BUY"
        elif forecast < 0:
            forecast = "SELL"
        else:
            forecast = "WAIT"
        st.markdown(
            f"""
            **Today's forecast**: {forecast}
            - Price: {prices[-1]}
            - Risk: {best_risk}
        """
        )

        if st.button("Run"):
            state, logs = policy.general_policy(
                prices,
                policy=selected_policy,
                state=start_state,
                risk=risk,
            )
            _show_policy(start_state, state, logs, prices)

    elif section == "Analyse Market":
        prices = st.session_state.prices
        selection = st.sidebar.radio("Select policy", policy.__all__)

        selected_policy = getattr(policy, selection)

        start_state = policy.STATE.copy()

        initial_capital = st.number_input("Capital", value=0)
        initial_assets = st.number_input("Assets", value=0)
        days = st.multiselect("CI Days", [i for i in range(2, 31)], default=[5, 10, 30])
        risk = st.number_input("Risk", value=0.05)

        start_state["capital"] = initial_capital
        start_state["assets"] = initial_assets

        mu = st.session_state.mu_hat
        sigma = st.session_state.sigma_hat

        if st.button("Run"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
                futures = [
                    executor.submit(
                        policy.general_policy,
                        prices,
                        selected_policy,
                        start_state,
                        alpha / 100,
                    )
                    for alpha in range(5, 100, 1)
                ]
                results = [f.result() for f in concurrent.futures.as_completed(futures)]

            results.sort(
                key=lambda v: policy._net_worth(prices, v[0]) + 3 * min(v[1]["debt"]),
                reverse=True,
            )

            full_positive = [
                result
                for result in results
                if result[0]["capital"] > 0 and result[0]["stock"] > 0
            ][:3]

            best_fit = results[:3]

            best_risk = best_fit[0][0]["risk"]
            forecast = selected_policy(prices, len(prices) - 1, mu, sigma, best_risk)
            if forecast > 0:
                forecast = "BUY"
            elif forecast < 0:
                forecast = "SELL"
            else:
                forecast = "WAIT"
            st.markdown(
                f"""
                **Today's forecast**: {forecast}
                - Price: {prices[-1]}
                - Risk: {best_risk}
            """
            )

            st.markdown("**Best ROI**")
            for result in best_fit:
                state, logs = result
                _show_policy(start_state, state, logs, prices)

            st.markdown("**Best positive outcome**")
            for result in full_positive:
                state, logs = result
                _show_policy(start_state, state, logs, prices)

            st.markdown("**Confidence intervals**")
            for d in days:
                res, fig = stock_price_ci(prices[-1], mu, sigma, d, risk, 1000)

                confidence = res["confidence_level"]
                above = res["probability_above_current"]

                mc_low, mc_high = res["monte_carlo_ci"]
                mc_low = round(float(mc_low), 2)
                mc_high = round(float(mc_high), 2)

                with st.expander("CI Stats", expanded=True):
                    st.markdown(
                        f"""
                    **Expected interval ({d} days)**:  
                    - Confidence = {confidence}  
                    - lower, upper (M-C) = {mc_low, mc_high}
                    - above  = {above}
                    """
                    )

            res, fig = stock_price_ci(prices[-1], mu, sigma, days[-1], best_risk, 1000)

            confidence = res["confidence_level"]
            above = res["probability_above_current"]

            mc_low, mc_high = res["monte_carlo_ci"]
            mc_low = round(float(mc_low), 2)
            mc_high = round(float(mc_high), 2)

            with st.expander("CI Stats", expanded=True):
                st.markdown(
                    f"""
                **Expected interval ({d} days)**:  
                - Confidence = {confidence}  
                - lower, upper (M-C) = {mc_low, mc_high}
                - above  = {above}
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
