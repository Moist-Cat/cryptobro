from collections import Counter
import concurrent.futures
import functools
from pathlib import Path

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
from utils import (
    estimate_parameters,
    simulate_prices,
    mc_ci,
    get_algorithm_params,
    param_config,
    predicate_config,
)

DIST = laplace
# DIST = norm
DIST_NAME = "laplace"
# DIST_NAME = "norm"


def _configure_algorithm(st, algorithms: list, msg="Select alorithm"):
    algo_names = [fun.__name__ for fun in algorithms]

    algorithm_name = st.sidebar.radio(msg, algo_names)
    algorithm = lambda: None
    for a in algorithms:
        if a.__name__ == algorithm_name:
            algorithm = a

    params = get_algorithm_params(algorithm)
    param_values = {}

    st.sidebar.write(f"**Configure parameters for the {algorithm_name} algorithm:**")
    st.sidebar.write(algorithm.__doc__)
    param_config(st, params, param_values)

    # We are overwriting the function default values
    algorithm = functools.partial(algorithm, **param_values)

    return algorithm


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

        # NOTE With large samples, the model begins to deteriorate
        if len(prices) > 365:
            prices = prices[-356:]

        st.session_state.df = prices
        st.session_state.prices = prices.values

        return st.session_state.prices
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
    mean, variance = DIST.fit(log_ratios)
    print(f"INFO - Fitted params {mean=} {variance=}")
    ks_stat, p_value = kstest(log_ratios, DIST_NAME, args=DIST.fit(log_ratios))

    # Create plots
    fig_hist, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(log_ratios, bins=200, density=True, alpha=0.6, label="Log Ratios")
    x = np.linspace(log_ratios.min(), log_ratios.max(), 100)
    ax1.plot(x, DIST.pdf(x, *DIST.fit(log_ratios)), "r-", label="Fitted Laplace")
    ax1.set_title("Log-Ratios Distribution vs Laplace Fit")
    ax1.legend()

    fig_qq, ax2 = plt.subplots(figsize=(10, 4))
    probplot(log_ratios, dist=DIST_NAME, plot=ax2)
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
    # Variance?-Gamma.
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

def plot_predictions_vs_outcomes(prices, actions, window=3):
    """
    Plots price trajectory with action markers and subsequent window performance
    Args:
        prices: Array of historical prices
        actions: Array of actions (1, 0, -1)
        window: Days to evaluate after prediction
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Main price plot
    ax.plot(prices, label='Price', alpha=0.7, linewidth=2)
    
    # Calculate future returns
    future_returns = np.zeros(len(prices))
    for i in range(len(prices)-window):
        future_returns[i] = (prices[i+window] / prices[i] - 1) * 100
    
    # Plot actions with annotations
    for i, action in enumerate(actions[:-window]):
        if not action:
            continue

        if action == 1:
            color = "green"
        else:
            color = "red"

        ax.scatter(i, prices[i], color=color, alpha=0.8, s=80)
        #ax.text(i, prices[i], f'{future_returns[i]:+.1f}%', fontsize=8, ha='center', va='top')
    
    ax.set_title(f'Trading Signals with Subsequent {window}D Returns')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    st.pyplot(fig)

def calculate_policy_metrics(prices, actions, window=3):
    """
    Calculates key performance metrics:
    1. Signal Accuracy: % of correct directional predictions
    2. Return Consistency: Average return after each signal type
    3. Holding Period Analysis: Optimal response window
    """
    results = {
        'buy': {'returns': [], 'correct': 0, 'total': 0},
        'sell': {'returns': [], 'correct': 0, 'total': 0}
    }
    
    for i in range(len(actions)-window):
        if actions[i] == 0:
            continue
            
        #future_return = (prices[i+window] - prices[i]) / prices[i]
        future_return = prices[i+window] - prices[i]
        action_type = 'buy' if actions[i] == 1 else 'sell'
        
        results[action_type]['returns'].append(future_return)
        results[action_type]['total'] += 1
        
        if (action_type == 'buy' and future_return > 0) or \
           (action_type == 'sell' and future_return < 0):
            results[action_type]['correct'] += 1
    
    # Streamlit metrics display
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("BUY signal", 
                 f"{results['buy']['total']}",
                 help="Total BUY signals emitted")
        
        if results['buy']['total'] > 0:
            st.metric("Precision BUY", 
                     f"{results['buy']['correct']/results['buy']['total']:.1%}",
                     help="Percent of correct signals")
            st.metric("Average return", 
                     f"{np.mean(results['buy']['returns'])*100:.2f}%")
    
    with col2:
        st.metric("SELL signal", 
                 f"{results['sell']['total']}",
                 help="Total SELL signals")
        
        if results['sell']['total'] > 0:
            st.metric("Precision SELL", 
                     f"{results['sell']['correct']/results['sell']['total']:.1%}",
                     help="Percent of correct signals")
            st.metric("Average return", 
                     f"{np.mean(results['sell']['returns'])*100:.2f}%")

def enhanced_evaluation(net_worth, actions, prices):
    """
    Improved evaluation considering:
    1. Risk-adjusted returns (Sharpe Ratio)
    2. Maximum drawdown
    3. Prediction consistency
    4. Debt avoidance penalty
    """
    # Maximum Drawdown
    peak = np.maximum.accumulate(net_worth)
    drawdown = (peak - net_worth) / peak
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Returns", "Risk", "Details"])
    
    with tab1:
        col1, col2 = st.columns(2)
        col1.metric("Final net worth", f"${net_worth[-1]:,.2f}")
        
        fig1, ax1 = plt.subplots()
        ax1.plot(net_worth)
        ax1.set_title("Net worth evolution")
        st.pyplot(fig1)
    
    with tab2:
        st.metric("Min Balance", f"${np.min(net_worth):,.2f}")
        
        fig2, ax2 = plt.subplots()
        ax2.plot(drawdown)
        ax2.set_title("Drawdown records")
        st.pyplot(fig2)
    
    with tab3:
        st.dataframe({
            "Day": range(len(net_worth)),
            "Net worth": net_worth,
            "Min balance": np.min(net_worth),
        })

# Example usage in Streamlit app
def policy_validation_page(logs, prices):
    st.title("Policy validation")
    
    with st.expander("Signal analysis", expanded=True):
        #window = st.slider("Validation window (days)", 1, 10, 3)
        window = 7

        plot_predictions_vs_outcomes(prices, logs['action'], window)

        calculate_policy_metrics(prices, logs['action'], window)
    
    with st.expander("Advanced evaluation"):
        enhanced_evaluation(
            logs['net_worth'],
            logs['action'],
            prices
        )


# ----------------------
# Streamlit App
# ----------------------


def _show_policy(start_state, state, logs, prices):
    debt = [val for val in logs["net_worth"] if val < 0]
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
        ["CSV Validation", "CI Evaluation", "Policy Experimentation", "Utils"],
    )

    if section == "CSV Validation":
        st.header("1. CSV Validation & Distribution Test")
        prices = load_csv()

        if prices is not None:
            price_sample = st.session_state.prices
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

            st.session_state.mu_hat = mu
            st.session_state.sigma_hat = sigma

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
                st.session_state.prices,
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

        number_of_days = 0
        risk_val = 0.05

        number_of_days = st.number_input(
            "Number of days", value=number_of_days, min_value=0, max_value=len(prices) - 1
        )
        days = st.multiselect("CI Days", [i for i in range(2, 31)], default=[5, 10, 30])
        risk = st.number_input("Risk", value=risk_val)

        mu = st.session_state.mu_hat
        sigma = st.session_state.sigma_hat

        general_policy = _configure_algorithm(
            st, [policy.general_policy], "General policy"
        )
        handle_action = _configure_algorithm(
            st, [policy.execute_action], "Action sub-policy"
        )
        selected_policy = _configure_algorithm(st, policy.POLICIES, "Select policy")

        start_state = policy.STATE.copy()

        best_risk = risk
        res = selected_policy(prices, len(prices) - 1, mu, sigma, best_risk)
        forecast = res["action"]
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

        if st.button("Run once with alpha"):
            state, logs = general_policy(
                prices[:number_of_days],
                state=start_state,
                risk=risk,
                policy=selected_policy,
                handle_action=handle_action,
            )
            _show_policy(start_state, state, logs, prices)

            policy_validation_page(logs, prices)

        if st.button("Calculate best alpha and run"):
            results = []
            for alpha in tqdm(range(5, 100, 1)):
                results.append(
                    general_policy(
                        prices,
                        state=start_state,
                        risk=alpha / 100,
                        policy=selected_policy,
                        handle_action=handle_action,
                    )
                )
            results.sort(
                key=lambda v: policy._net_worth(prices, v[0]) + 3 * min(v[1]["net_worth"]),
                reverse=True,
            )

            full_positive = [
                result
                for result in results
                if result[0]["capital"] > 0 and result[0]["stock"] > 0
            ][:3]

            best_fit = results[:3]

            best_risk = best_fit[0][0]["risk"]
            res = selected_policy(prices, len(prices) - 1, mu, sigma, best_risk)
            forecast = res["action"]
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

    elif section == "Utils":
        st.markdown(
            f"""
        **Variables**:  
        - mu: {st.session_state.mu_hat}
        - sigma: {st.session_state.sigma_hat}
        """
        )

        S0 = st.number_input("Initial Price (S₀)", value=100.0)
        T = st.number_input("Days (T)", value=365)

        mu = st.session_state.mu_hat
        sigma = st.session_state.sigma_hat

        if st.button("Generate Prices"):
            prices = simulate_prices(S0, mu, sigma, T, 1)
            df = pd.DataFrame({"Close": prices[0]})

            num = 0
            file_fmt =  "Downloads/gen{num}.csv"
            file = Path.home() / file_fmt.format(num=num)
            while file.exists():
                num += 1
                file = Path.home() / file_fmt.format(num=num)
            df.to_csv(file)



if __name__ == "__main__":
    if "prices" not in st.session_state:
        st.session_state.df = None
        st.session_state.prices = None
        st.session_state.file = None
        st.session_state.mu_hat = 0
        st.session_state.sigma_hat = 1
    main()
