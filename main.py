import math
import random
import os
from collections import Counter
import concurrent.futures
import functools
from pathlib import Path
from datetime import datetime

import seaborn as sns
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import scipy
from scipy.stats import (
    chi2_contingency,
    fisher_exact,
    norm,
    kstest,
    ttest_ind,
    ks_2samp,
    shapiro,
    lognorm,
    probplot,
    ttest_1samp,
)
from tqdm import tqdm

import policy
from utils import (
    # stats
    estimate_parameters,
    simulate_prices,
    stock_price_ci,
    process_window,
    # autocorr
    find_clusters,
)
from meta import (
    load_csv,
    update_required,
    _configure_algorithm,
    DB_DIR,
    DIST,
    DIST_NAME,
)
from chat import (
    expand_query,
    lsa_rag_retrieve,
    summarize_rag_results,
    summarize_news,
    assemble_prompt,
    chatbot,
)

from agent import (
    core,
    gene,
    save_agent,
)

from retrieval import technical, fundamental
from retrieval import client


def plot_violations(prices, violations, window_size=7):
    """
    Plot price series with violation regions highlighted

    Args:
        prices (array): Historical price data
        violations (list): List of violation dicts from validation
        window_size (int): Lookahead window for CIs
    """
    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot price series
    ax.plot(prices, color="black", alpha=0.7, label="Price")

    # Create violation patches
    violation_patches = []
    colors = []

    for v in violations:
        # Determine violation type and color
        if v["price"] < v["lower"]:
            color = "red"
            violation_type = "Lower"
        else:
            color = "green"
            violation_type = "Upper"

        # Create rectangle for violation window
        rect = Rectangle(
            (v["day"], v["lower"]),
            width=window_size,
            height=v["upper"] - v["lower"],
            alpha=0.2,
        )
        violation_patches.append(rect)
        colors.append(color)

        # Mark violation point
        ax.scatter(
            v["day"] + 1,
            v["price"],
            color=color,
            s=100,
            label=f"{violation_type} Bound Violation",
        )

    # Add violation regions
    pc = PatchCollection(violation_patches, alpha=0.2)
    pc.set_array(np.array([1 if c == "red" else 0 for c in colors]))
    ax.add_collection(pc)

    # Create custom legend
    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Price"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Lower Violation",
            markerfacecolor="red",
            markersize=10,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Upper Violation",
            markerfacecolor="green",
            markersize=10,
        ),
        PatchCollection(
            [Rectangle((0, 0), 1, 1)], alpha=0.2, label="CI Window", color="gray"
        ),
    ]

    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_title(f"Price Violations (Window Size: {window_size} days)")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")

    return fig


# Streamlit wrapper
def st_plot_violations(prices, violations, window_size=7):
    fig = plot_violations(prices, violations, window_size)
    st.pyplot(fig)

    # Add summary statistics
    lower_vios = sum(1 for v in violations if v["price"] < v["lower"])
    upper_vios = len(violations) - lower_vios

    col1, col2 = st.columns(2)
    col1.metric("Lower Bound Violations", lower_vios)
    col2.metric("Upper Bound Violations", upper_vios)

    st.write(
        f"Total violations: {len(violations)} ({len(violations)/len(prices):.1%} of days)"
    )


def display_results(result):
    """Display chi-square results and confusion matrix"""
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Contingency Table")
        st.dataframe(result["contingency_table"])

        # Calculate metrics
        table = result["contingency_table"].values
        if table.shape == (2, 2):
            TP = table[1, 1]
            FP = table[1, 0]
            FN = table[0, 1]
            precision = TP / (TP + FP) if TP + FP > 0 else 0
            recall = TP / (TP + FN) if TP + FN > 0 else 0
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if precision + recall > 0
                else 0
            )

            st.write(f"**Precision**: {precision:.2f}")
            st.write(f"**Recall**: {recall:.2f}")
            st.write(f"**F1 Score**: {f1:.2f}")

    with col2:
        st.write("### Chi-Square/Fisher's Test")
        st.write(f"Odds ratio = {result['odds_ratio']:.2f}")
        st.write(f"p-value = {result['p_value']:.4f}")

        # Interpret p-value
        if result["p_value"] < 0.05:
            st.success("Statistically significant (p < 0.05)")
        else:
            st.warning("Not statistically significant (p ≥ 0.05)")

        # Confusion matrix visualization
        if not result["contingency_table"].empty:
            plt.figure(figsize=(6, 4))
            sns.heatmap(result["contingency_table"], annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            st.pyplot(plt)


def plot_timeline(price_df):
    """Visualize price, events, and violations"""
    plt.figure(figsize=(12, 6))

    # Plot price
    plt.plot(price_df["Date"], price_df["Close"], label="Price", alpha=0.7)

    # Mark events
    events = price_df[price_df["event"] == 1]
    plt.scatter(
        events["Date"],
        events["Close"],
        color="green",
        marker="^",
        s=100,
        label="Protocol Updates",
    )

    # Mark violations
    violations = price_df[price_df["violation"] == 1]
    plt.scatter(
        violations["Date"],
        violations["Close"],
        color="red",
        marker="x",
        s=100,
        label="Price Violations",
    )

    plt.title(f"Price Timeline with Events and Violations")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    st.pyplot(plt)


def run_correlation_analysis(price_df, event_dates, alpha=0.95, n_sim=1000, k=5):
    """
    Analyze correlation between events and price violations
    Returns results for both directions of correlation
    """
    # Precompute violations
    prices = price_df["Close"].values
    n = len(prices)

    # Compute violations for each window
    res = validate_ci_coverage(
        prices,
        k,
        alpha,
        n_sim,
    )
    violation_indices = set(map(lambda i: i["day"], res["violations"]))

    # Create violation column
    price_df = price_df.copy()
    price_df["violation"] = 0
    price_df.loc[price_df.index.isin(violation_indices), "violation"] = 1

    # Create event column
    price_df["event"] = price_df["Date"].isin(event_dates["Date"]).astype(int)

    # Prepare analysis dataframes
    df = price_df[["Date", "event", "violation"]].copy()

    # Hypothesis A: Events cause violations (within k days after event)
    df["violation_next_k"] = 0
    for idx, row in df.iterrows():
        # if row["event"] == 1:
        future = df.iloc[idx + 1 : min(idx + 1 + k, len(df))]
        if any(future["violation"]):
            df.at[idx, "violation_next_k"] = 1

    # Hypothesis B: Violations caused by events (within k days before violation)
    df["event_prev_k"] = 0
    for idx, row in df.iterrows():
        # if row["violation"] == 1:
        past = df.iloc[max(0, idx - k) : idx]
        if any(past["event"]):
            df.at[idx, "event_prev_k"] = 1

    # Build contingency tables
    df_A = df
    contingency_A = pd.crosstab(
        df_A["event"],
        df_A["violation_next_k"],
        rownames=["Event"],
        colnames=["Violation in next k days"],
    )

    df_B = df  # Only consider days with violations
    contingency_B = pd.crosstab(
        df_B["violation"],
        df_B["event_prev_k"],
        rownames=["Violation"],
        colnames=["Event in previous k days"],
    )

    # Chi-square tests
    odds_ratio_A, p_value_A = fisher_exact(
        contingency_A.values, alternative="two-sided"
    )
    # odds_ratio_A, p_value_A, _, _ = chi2_contingency(contingency_A.values, correction=True)
    odds_ratio_B, p_value_B = fisher_exact(
        contingency_B.values, alternative="two-sided"
    )
    # odds_ratio_B, p_value_B, _, _ = chi2_contingency(contingency_B.values, correction=True)

    return {
        "price_df": price_df,
        "hypothesis_A": {
            "contingency_table": contingency_A,
            "odds_ratio": odds_ratio_A,
            "p_value": p_value_A,
        },
        "hypothesis_B": {
            "contingency_table": contingency_B,
            "odds_ratio": odds_ratio_B,
            "p_value": p_value_B,
        },
    }


def verify_dist(prices, dist=DIST, dist_name=DIST_NAME):
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
    # log_ratios = ratios

    # Laplace tests
    args = DIST.fit(log_ratios)
    print(f"INFO - Fitted params {args=}")
    ks_stat, p_value = kstest(log_ratios, DIST_NAME, args=DIST.fit(log_ratios))

    # Create plots
    fig_hist, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(
        log_ratios,
        bins=math.ceil(math.sqrt(len(ratios))),
        density=True,
        alpha=0.6,
        label="Log Ratios",
    )
    x = np.linspace(log_ratios.min(), log_ratios.max(), 100)
    ax1.plot(x, DIST.pdf(x, *DIST.fit(log_ratios)), "r-", label=f"Fitted {DIST_NAME}")
    ax1.set_title(f"Log-Ratios Distribution vs {DIST_NAME} Fit")
    ax1.legend()

    fig_qq, ax2 = plt.subplots(figsize=(10, 4))
    probplot(log_ratios, dist=DIST, plot=ax2, sparams=args)
    ax2.set_title("Q-Q Plot of Log-Ratios")
    ax2.get_lines()[0].set_markerfacecolor("b")
    ax2.get_lines()[0].set_markersize(4.0)
    ax2.get_lines()[1].set_color("r")
    ax2.get_lines()[1].set_linewidth(2.0)

    return {
        "ks_test": (ks_stat, p_value),
        "mean": args[0],
        "std": args[1],
        "fig_hist": fig_hist,
        "fig_qq": fig_qq,
    }


# ----------------------
# Evaluation Functions
# ----------------------
def validate_ci_coverage(
    prices,
    window_size=7,
    alpha=0.05,
    n_sim=1000,
    dynamic_estimation=False,
    lookback_days=0,
):
    """
    Validates CI coverage by checking how often prices stay within predicted intervals.

    Parameters:
    prices (array): Historical price data
    window_size (int): Days ahead to predict (n in your algorithm)
    alpha (float): Significance level
    n_sim (int): Number of simulations for Monte Carlo CI

    Returns:
    dict: Test results including coverage score and detailed statistics
    """
    total = 0
    violations = []
    ci_widths = []

    start = 0
    if dynamic_estimation:
        start = lookback_days

    for i in tqdm(range(start, len(prices) - window_size)):
        if dynamic_estimation:
            violation = process_window(
                prices,
                i - lookback_days,
                i,
                window_size,
                alpha,
                n_sim,
            )
        else:
            violation = process_window(prices, 0, i, window_size, alpha, n_sim)

        if violation:
            violations.append(violation)
        total += 1

    wins = total - len(violations)
    coverage = wins / total
    expected = 1 - alpha

    return {
        "coverage_score": coverage,
        "expected_coverage": expected,
        "total_windows": total,
        "successful_windows": wins,
        "violation_rate": 1 - coverage,
        "violations": violations,
    }


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
    ax.plot(prices, label="Price", alpha=0.7, linewidth=2)

    # Calculate future returns
    future_returns = np.zeros(len(prices))
    for i in range(len(prices) - window):
        future_returns[i] = (prices[i + window] / prices[i] - 1) * 100

    # Plot actions with annotations
    for i, action in enumerate(actions):
        if not action:
            continue

        if action > 0:
            color = "green"
        elif action < 0:
            color = "red"

        ax.scatter(i, prices[i], color=color, alpha=0.8, s=80)
        # ax.text(i, prices[i], f'{future_returns[i]:+.1f}%', fontsize=8, ha='center', va='top')

    ax.set_title(f"Trading Signals with Subsequent {window}D Returns")
    ax.set_xlabel("Days")
    ax.set_ylabel("Price")
    st.pyplot(fig)


def calculate_policy_metrics(prices, actions, window=7):
    """
    Calculates key performance metrics:
    1. Signal Accuracy: % of correct directional predictions
    2. Return Consistency: Average return after each signal type
    3. Holding Period Analysis: Optimal response window
    """
    results = {
        "buy": {"returns": [], "correct": 0, "total": 0},
        "sell": {"returns": [], "correct": 0, "total": 0},
    }

    for i in range(len(actions) - window):
        if actions[i] == 0:
            continue

        # future_return = (prices[i+window] - prices[i]) / prices[i]
        future_return = prices[i + window] - prices[i]
        future_gain = prices[i + window] / prices[i]
        # action_type = "buy" if actions[i] == 1 else "sell"
        action_type = "buy" if actions[i] > 0 else ("sell" if actions[i] else "wait")

        results[action_type]["returns"].append(future_gain - 1)
        results[action_type]["total"] += 1

        if (action_type == "buy" and future_return > 0) or (
            action_type == "sell" and future_return < 0
        ):
            results[action_type]["correct"] += 1

    # Streamlit metrics display
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "BUY signal", f"{results['buy']['total']}", help="Total BUY signals emitted"
        )

        if results["buy"]["total"] > 0:
            st.metric(
                "Precision BUY",
                f"{results['buy']['correct']/results['buy']['total']:.1%}",
                help="Percent of correct signals",
            )
            st.metric(
                "Average return", f"{np.mean(results['buy']['returns'])*100:.2f}%"
            )

    with col2:
        st.metric(
            "SELL signal", f"{results['sell']['total']}", help="Total SELL signals"
        )

        if results["sell"]["total"] > 0:
            st.metric(
                "Precision SELL",
                f"{results['sell']['correct']/results['sell']['total']:.1%}",
                help="Percent of correct signals",
            )
            st.metric(
                "Average return", f"{np.mean(results['sell']['returns'])*100:.2f}%"
            )


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
        st.dataframe(
            {
                "Day": range(len(net_worth)),
                "Net worth": net_worth,
                "Min balance": np.min(net_worth),
            }
        )


def policy_validation_page(logs, prices, window=7):
    st.title("Policy validation")

    with st.expander("Signal analysis", expanded=True):

        plot_predictions_vs_outcomes(prices, logs["action"], window)

        calculate_policy_metrics(prices, logs["action"], window)

    with st.expander("Advanced evaluation"):
        enhanced_evaluation(logs["net_worth"], logs["action"], prices)


# ----------------------
# Streamlit App
# ----------------------


def chat_section(
    risk,
    auto_close,
    forecast,
    rsi,
    expected_value,
    ci,
    hlc,
    user_query="General forecast",
):
    user_input = st.chat_input("Ask something...")
    if not user_input:
        return
    if user_input == "clear":
        return

    expanded_query = expand_query(
        auto_close=auto_close,
        risk=risk,
        position_size=forecast / 10,
        indicators={
            "rsi": rsi,
            "expected_value": expected_value,
            "hlc": hlc,
            "ci": ci,
        },
    )

    st.header(f"Q: {user_input}")

    st.header("Fundamental Insights")

    rag_summary = summarize_rag_results(
        lsa_rag_retrieve(st.session_state.symbol, auto_close)
    )

    with st.spinner("Fetching news..."):
        news_summary = summarize_news(
            fundamental.fetch_news(
                st.session_state.symbol
                if "USDT" not in st.session_state.symbol
                else st.session_state.symbol[:-4]
            )
        )

    # 5. Build and execute prompt
    prompt = assemble_prompt(
        expanded_query=expanded_query,
        rag_summary=rag_summary,
        news_summary=news_summary,
        user_query=user_input,
    )

    with st.spinner("Analyzing market conditions..."):
        response = chatbot.reply(prompt)

    response.replace("$", "")
    print(response)

    st.write(f"## Trading Recommendation\n{response}")

    # 7. Visual confirmation tools
    st.download_button("Save Analysis", response, file_name="trading_analysis.md")


def fundamentals_section():
    st.header("Fundamental Analysis")

    symbol = st.selectbox("Select symbol", ["SOL", "TRX", "ETH"])
    alpha = st.number_input(
        "Error (alpha)", value=0.05, min_value=0.03, max_value=0.99, step=0.01
    )
    k = st.number_input(
        "Correlation window (k days)", value=7, min_value=1, max_value=30
    )
    size = st.number_input("Sample size (last `size` days)", value=500, min_value=100)

    tech = technical.AssetScraper()
    pro = fundamental.ProtocolScraper()
    dat = fundamental.DataProcessor()

    if st.button("Run Correlation Analysis"):
        tech_filename = DB_DIR / f"{symbol}USDT.csv"
        fun_filename = pro.raw_data_filename(symbol)
        fun_data = dat.process_file(fun_filename)  # List of events with dates
        tech_data = pd.read_csv(tech_filename, parse_dates=["Date"])[-size:]
        tech_data = tech_data.reset_index()

        results = run_correlation_analysis(
            price_df=tech_data, events=fun_data, alpha=alpha, k=k
        )

        # Display results
        st.subheader("Hypothesis A: Events Cause Price Violations")
        st.write("Do events cause significant price changes within k days?")
        display_results(results["hypothesis_A"])

        st.subheader("Hypothesis B: Violations Caused by Events")
        st.write("Are price violations preceded by events within k days?")
        display_results(results["hypothesis_B"])

        # Visualize timeline
        st.subheader("Events and Violations Timeline")
        plot_timeline(results["price_df"])


def agents_section():
    st.header("🤖 Agent Evolution Simulation")

    with st.expander("**Gene template guide**", expanded=False):
        st.json(gene.PARAM_SPACE)

    # Section 1: Simulation Parameters
    with st.expander("⚙️ Simulation Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            init_agents = st.number_input("Initial Agents", 1, 1000, 1)
            init_cash = st.number_input("Starting Cash", 100, 1_000_000, 100)
            iter_num = st.number_input("Skip number", 10, 1_000_000, 100)
        with col2:
            elite_threshold = st.slider(
                "Elite Threshold",
                1,
                100,
                20,
                help="Percentage of top performers to keep",
            )
            mutation_rate = st.slider("Mutation Rate", 0.0, 1.0, 0.3)
            rent = st.checkbox("Enable rent", True)
        with col3:
            max_generations = st.number_input("Max Generations", 1, 100000, 50000)
            stop_condition = st.selectbox(
                "Stop Condition", ["Population Size", "Generations"]
            )
            auto_cross = st.checkbox("Auto Cross", True)

    # Section 2: Population Initialization
    if "manager" not in st.session_state:
        if st.button("🧬 Generate Initial Population"):
            st.session_state.manager = core.generate_population_manager(
                size=init_agents,
                state_size=len(core._get_state(*core._select_dataset())),
                initial_cash=init_cash,
            )
            st.session_state.logs = []
            st.success(f"Created {init_agents} agents with ${init_cash:,.2f} each")

    # Section 3: Simulation Controls
    if "manager" in st.session_state:
        current_gen = len(st.session_state.logs)
        control_cols = st.columns([1, 1, 1, 2])
        with control_cols[0]:
            if st.button("⏩ Step Forward"):
                log = core.simulation(st.session_state.manager, current_gen, rent)
                st.session_state.logs.append(log)

        with control_cols[1]:
            if st.button(f"⏩⏩ {iter_num} Steps"):
                progress = st.progress(0)
                for i in tqdm(range(iter_num)):
                    if (len(st.session_state.manager.agents) / init_agents) <= (
                        elite_threshold / 100
                    ):
                        break
                    log = core.simulation(st.session_state.manager, current_gen, rent)
                    st.session_state.logs.append(log)
                    current_gen += 1
                    progress.progress(i / iter_num)

        with control_cols[2]:
            if st.button("🏃♂️ Run Full Simulation"):
                # random variable to auto cross when necessary
                e = scipy.stats.expon(0, 1)

                progress = st.progress(0)
                current_gen = len(st.session_state.logs)

                while (
                    current_gen < max_generations
                    and len(st.session_state.manager.agents) > 2
                ):
                    log = core.simulation(st.session_state.manager, current_gen, rent)
                    st.session_state.logs.append(log)
                    current_gen += 1
                    progress.progress(current_gen / max_generations)

                    # As the individuals of the species perish
                    # the probability of auto_cross triggering increases
                    alpha = 1 - len(st.session_state.manager.agents) / init_agents
                    interval = e.interval(alpha)
                    val = e.rvs()

                    if (
                        log["dead"]
                        and auto_cross
                        and (val > interval[0] and val < interval[1])
                    ):
                        agents = st.session_state.manager.agents
                        children = gene.biased_crossover(
                            agents, init_agents, init_cash, mutation_rate
                        )
                        for child in children:
                            st.session_state.manager.report(child)

                    if auto_cross and (len(st.session_state.manager.agents) <= 2):
                        print(
                            "WARNING - The species has gone extinct..."
                            "using a deux ex machina to continue the epxeriment"
                        )
                        # prevent extinction
                        for _ in range(3 - len(st.session_state.manager.agents)):
                            st.session_state.manager.report(
                                core.Agent(init_money=init_cash)
                            )
                    if not (current_gen % 1000):
                        print(
                            "INFO - Current DNA:",
                            st.session_state.manager.agents[0].brain.dna,
                        )
        if st.button("Cross"):
            agents = st.session_state.manager.agents
            children = gene.biased_crossover(
                agents, init_agents, init_cash, mutation_rate
            )
            for child in children:
                st.session_state.manager.report(child)
        if st.button("Export Agent"):
            agent = st.session_state.manager.agents[0]
            save_agent(agent)

        # Section 4: Visualization
        st.subheader("📈 Population Dynamics")

        # Real-time metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Current Population", len(st.session_state.manager.agents))
        with metric_cols[1]:
            avg_money = st.session_state.manager.avg_wealth()
            st.metric("Average Capital", f"{avg_money:.2f}")

        with metric_cols[3]:
            st.metric("Generation", len(st.session_state.logs))

        # Evolution history plot
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()

        gens = [log["generation"] for log in st.session_state.logs]
        pop = [log["population"] for log in st.session_state.logs]
        wealth = [log["avg_wealth"] for log in st.session_state.logs]

        ax1.plot(gens, pop, "b-", label="Population")
        ax2.plot(gens, wealth, "r-", label="Wealth")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Population", color="b")
        ax2.set_ylabel("Average Wealth", color="r")
        st.pyplot(fig)

        # Elite agent inspection
        st.subheader("🏆 Top Performers")
        top_agents = sorted(
            st.session_state.manager.agents,
            key=lambda x: x.money,
            reverse=True,
        )[:5]

        for i, agent in enumerate(top_agents):
            with st.expander(f"{agent.name} - ${agent.money:,.2f}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Genes**")
                    st.json(tuple(agent.brain.dna))
                with col2:
                    st.write("**Trading Stats**")

        # Gene statistics
        st.subheader("🧬 Population Gene Distribution")
        gene_df = pd.DataFrame(
            [agent.brain.dna for agent in st.session_state.manager.agents]
        )
        if st.session_state.manager.agents:
            st.dataframe(gene_df.describe())

    else:
        st.warning("Initialize population first!")


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
        [
            "CSV Validation",
            "CI Evaluation",
            "Policy Experimentation",
            "Agents",
            "Fundamental",
            "Utils",
        ],
    )

    if section == "CSV Validation":
        st.header("1. CSV Validation & Distribution Test")

        resampling = st.checkbox("Ordered resampling", False)
        size = st.number_input("Sample size (last n if not resampled)", value=350)

        csv_paths = list(DB_DIR.glob("*.csv")) + list(
            (Path.home() / "Downloads").glob("*.csv")
        )

        if update_required(csv_paths) or update_required(
            fundamental.ProtocolScraper().get_files()
        ):
            print("INFO - Data is out-of-date")
            with st.spinner("Updating..."):
                try:
                    client.update()
                    fundamental.update()
                except Exception as exc:
                    print("ERROR - Update failed")

        filename = st.selectbox("Select CSV", csv_paths)

        if st.button("Estimate parameters"):
            uploaded_file = open(filename)
            prices = load_csv(uploaded_file)
            st.session_state.prices = prices
            st.session_state.symbol = filename.name.split(".")[-2]

            uploaded_file.close()

            price_sample = prices
            if resampling:
                start = random.randint(0, max(0, len(price_sample) - (size + 1)))
                end = min(start + size, len(price_sample))
            else:
                end = len(price_sample)
                start = max(end - size, 0)

            price_sample = price_sample[start:end]
            st.session_state.prices = price_sample

            with st.expander("Step 1: Estimate Parameters", expanded=True):
                args = estimate_parameters(price_sample)
                mu_hat = args[0]
                sigma_hat = args[1]
                skew_hat = args[2]
                st.markdown(
                    f"""
                **Estimated Parameters for n={len(price_sample)}**:
                - $\hat{{\mu}}$ = {mu_hat:.6f}
                - $\hat{{\sigma}}$ = {sigma_hat:.6f}
                - $\hat{{\\alpha}}$ = {skew_hat:.6f}
                """
                )

            with st.expander(
                "Step 2: Verify distribution with new parameters", expanded=True
            ):
                res = verify_dist(price_sample)
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
            T = st.number_input("Days (T)", value=7)
            aleph = st.number_input("Error (alpha)", value=0.66)
            plot_graph = st.checkbox("Plot graph", False)
        with col2:
            num_paths = st.number_input("Paths", value=1000)
            lookback_days = st.number_input("Lookback days", value=240)
            dynamic_estimation = st.checkbox(
                "Dynamic estimation (moving median)", False
            )

        if st.button("Run CI Simulation"):
            prices = st.session_state.prices
            if "prices" in st.session_state and st.session_state.prices is not None:
                if not dynamic_estimation:
                    res = stock_price_ci(prices, T, aleph, num_paths)
                else:
                    res = stock_price_ci(prices[-lookback_days:], T, aleph, num_paths)

                confidence = res["confidence_level"]

                mc_low, mc_high = res["monte_carlo_ci"]
                mc_low = round(float(mc_low), 2)
                mc_high = round(float(mc_high), 2)

                # Visualization
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(res["future_prices"], bins=50, density=True, alpha=0.7)
                ax.axvline(mc_low, color="g", linestyle=":", label="Monte Carlo CI")
                ax.axvline(mc_high, color="g", linestyle=":")
                ax.set_title(f"Stock Price Distribution After {T} Days")
                ax.set_xlabel("Price")
                ax.set_ylabel("Density")
                ax.legend()

                with st.expander("CI Plot", expanded=True):
                    st.pyplot(fig)
                with st.expander("CI Stats", expanded=True):
                    st.markdown(
                        f"""
                    **Expected interval**:  
                    - Confidence = {confidence}  
                    - lower, upper (M-C) = {mc_low, mc_high}
                    """
                    )

        if st.button("Validate cofidence interval"):
            prices = st.session_state.prices
            res = validate_ci_coverage(
                prices,
                window_size=T,
                alpha=aleph,
                n_sim=num_paths,
                dynamic_estimation=dynamic_estimation,
                lookback_days=lookback_days,
            )
            coverage = res["coverage_score"]
            expected = res["expected_coverage"]
            violations = res["violations"]
            violation_rate = res["violation_rate"]
            total = res["total_windows"]
            wins = res["successful_windows"]

            # Plot results
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Coverage plot
            ax1.axhline(expected, color="r", linestyle="--", label="Expected Coverage")
            ax1.bar(["Actual"], [res["coverage_score"]], label=f"Actual (n={total})")
            ax1.set_ylim(0, 1)
            ax1.set_title(f"CI Coverage ({T}-day windows)")
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

            with st.expander("Expectation VS Reality", expanded=True):
                st.pyplot(fig)

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

            with st.expander("Violations", expanded=False):
                st.json(violations)

            if not violations:
                print("WARNING - No violations found. Avoiding division by zero")
                violations = [{"day": 0, "lower": 0, "upper": 0, "price": 0}]
            clusters = find_clusters(violations)

            if clusters:
                big = len(max(clusters, key=len))
                small = len(min(clusters, key=len))
            else:
                big = 0
                small = 0

            st.markdown(
                f"""
                **Volatility clusters**
                - biggest: {big}
                - smallest: {small}
                - mean: {sum(map(len, clusters))/len(clusters):.2f}
                - clustered: {sum(map(len, clusters))/len(violations)*100:.2f}%
                """
            )

            if plot_graph:
                st_plot_violations(prices, violations)

    elif section == "Policy Experimentation":
        prices = st.session_state.prices

        number_of_days = len(prices)
        risk_val = 0.66

        number_of_days = st.number_input(
            "Number of days",
            value=number_of_days,
            min_value=0,
            max_value=len(prices),
        )
        days_to_verify = st.number_input("Days to verify", value=7, min_value=1)
        risk = st.number_input("Risk", value=risk_val)

        general_policy = _configure_algorithm(
            st, [policy.general_policy], "General policy"
        )
        selected_policy = _configure_algorithm(st, policy.POLICIES, "Select policy")

        start_state = policy.STATE.copy()

        best_risk = risk
        history = st.session_state.history

        # indicators
        forecast = policy.policy_agent(
            prices, len(prices) - 1, best_risk, history, load=True
        )["action"]
        rsi = policy.policy_rsi(prices, len(prices) - 1, best_risk, history)["action"]
        expected_value = policy.policy_expected_value(
            prices, len(prices) - 1, best_risk, history
        )["action"]
        ci = stock_price_ci(prices, days_to_verify, best_risk, 100000)["ci"]
        hlc = prices.max(), prices.min(), prices[-1]

        symbol = st.session_state.symbol

        chat_section(
            risk=best_risk,
            auto_close=days_to_verify,  # auto_close == days_to_verify
            forecast=forecast,
            rsi=rsi,
            ci=ci,
            expected_value=expected_value,
            hlc=hlc,
            user_query="General forecast",
        )

        res = selected_policy(prices, len(prices) - 1, best_risk, history)

        forecast = res["action"]
        if forecast > 0:
            forecast = "BUY"
        elif forecast < 0:
            forecast = "SELL"
        else:
            forecast = "WAIT"
        st.markdown(
            f"""
            **Today's forecast for {symbol}**: {forecast}
            - Price: {hlc[2]}
            - High: {hlc[0]}
            - Low: {hlc[1]}
            - CI: ({ci[0]:.2f}, {ci[1]:.2f})
        """
        )

        if st.sidebar.button("Back-Test Policy"):
            state, logs = general_policy(
                prices[:number_of_days],
                state=start_state,
                risk=risk,
                policy=selected_policy,
            )
            _show_policy(start_state, state, logs, prices)
            st.session_state.history = logs["action"]

            policy_validation_page(logs, prices, days_to_verify)
    elif section == "Agents":
        agents_section()
    elif section == "Fundamental":
        fundamentals_section()
    elif section == "Utils":
        T = st.number_input("Days (T)", value=350)

        if st.button("Generate Prices"):
            prices = st.session_state.prices
            sim_prices = simulate_prices(prices, len(prices) - 1)
            df = pd.DataFrame({"Close": sim_prices[0]})

            num = 0
            file_fmt = "Downloads/gen{num}.csv"
            file = Path.home() / file_fmt.format(num=num)
            while file.exists():
                num += 1
                file = Path.home() / file_fmt.format(num=num)
            df.to_csv(file)


if __name__ == "__main__":
    if "prices" not in st.session_state:
        st.session_state.symbol = ""
        st.session_state.prices = None
        st.session_state.history = []
    main()
