from functools import partial
from joblib import Parallel, delayed
import time

from scipy.stats import laplace
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from meta import _configure_algorithm, load_csv
from utils import estimate_parameters, mc_ci
from policy import POLICIES


class AnalysisSuite:
    def __init__(self):
        self._init_session_state()

    def _init_session_state(self):
        """Initialize all session state variables"""
        defaults = {
            "current_idx": 120,
            "auto_play": False,
            "show_graphs": True,
            "show_stats": True,
            "show_logs": True,
            "cache": {},
            "log": [],
            "ci": None,
            "selected_policy": None,
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _control_panel(self):
        """Create interactive control panel"""
        col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
        with col1:
            if st.button("⏪ Previous Day"):
                st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
        with col2:
            if st.button("Next Day ▶️") and not st.session_state.auto_play:
                st.session_state.current_idx += 1
        with col3:
            st.session_state.auto_play = st.button("🚀 Auto Play")
        with col4:
            st.session_state.current_idx = st.number_input(
                "Jump to Day",
                min_value=0,
                max_value=len(st.session_state.prices) - 1,
                value=st.session_state.current_idx,
            )

    def _configuration_panel(self):
        """Configuration sidebar"""
        with st.sidebar.expander("⚙️ Analysis Settings", expanded=True):
            st.session_state.windows = st.multiselect(
                "Time Windows", [7, 15, 27, 37], default=[7, 15, 27, 37]
            )
            st.session_state.ci = _configure_algorithm(st, [mc_ci])

            # Algorithm configuration
            st.session_state.selected_policy = _configure_algorithm(st, POLICIES)

        with st.sidebar.expander("🎚 Performance Settings"):
            st.session_state.parallel_workers = st.slider("Parallel Workers", 1, 8, 4)
            st.session_state.show_graphs = st.checkbox("Show Graphs", True)
            st.session_state.show_stats = st.checkbox("Show Statistics", True)
            st.session_state.show_logs = st.checkbox("Show Logs", True)

    def _plot_laplace_distributions(self, params):
        """Plot multiple Laplace distributions"""
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)
        axes = [fig.add_subplot(gs[i // 2, i % 2]) for i in range(4)]

        for ax, (window, (mu, sigma)) in zip(axes, params.items()):
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            y = laplace.pdf(x, loc=mu, scale=sigma)
            ax.plot(x, y, label=f"{window} days")
            ax.set_title(f"Laplace({mu:.4f}, {sigma:.4f}) - {window} days")
            ax.legend()

        return fig

    def _update_analysis(self, prices, idx):
        """Parallelized analysis update"""
        if idx in st.session_state.cache:
            return st.session_state.cache[idx]

        def process_window(estimator, window):
            start = max(0, idx - window)
            window_prices = prices[start : idx + 1]
            mu, sigma = estimate_parameters(window_prices)
            ci = estimator(prices[idx], mu, sigma)
            return window, (mu, sigma), ci

        results = Parallel(n_jobs=st.session_state.parallel_workers)(
            delayed(process_window)(st.session_state.ci, window)
            for window in st.session_state.windows
        )

        stats = {w: p for w, p, _ in results}
        # The CI function returns (prices, top, bottom)
        cis = {w: (c[1], c[2]) for w, _, c in results}

        st.session_state.cache[idx] = (stats, cis)

        log = st.session_state.log

        change = 0.0 if idx == 0 else prices[idx] - prices[idx - 1]
        log_msg = f"Price: *{prices[idx]}*; Change: *{change}*\n\n"
        for k, v in stats.items():
            log_msg += f"{k}-day Laplace: {round(v[0], 4)}\n\n"
        log_msg += "\n\n\n"

        log.append(
            log_msg,
        )
        return stats, cis

    def _price_chart(self, prices, idx, cis):
        """Interactive price chart with CI bands"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(prices, label="Price")
        ax.plot(idx, prices[idx], "ro", markersize=10, label="Current Position")

        for window, ci in cis.items():
            ax.fill_between(
                range(idx, idx + window), ci[0], ci[1], alpha=0.1, label=f"{window}d CI"
            )

        ax.set_title("Price Movement with Confidence Intervals")
        ax.legend()
        return fig

    def run(self, uploaded_file):
        """Main analysis suite execution"""
        # Initialize data
        if "prices" not in st.session_state:
            with st.spinner("Loading data..."):
                st.session_state.prices = load_csv(uploaded_file)

        self._configuration_panel()
        self._control_panel()

        # Main analysis
        with st.spinner("Computing distributions..."):
            stats, cis = self._update_analysis(
                st.session_state.prices, st.session_state.current_idx
            )

        # Visualization
        if st.session_state.show_graphs:
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(self._plot_laplace_distributions(stats))
            with col2:
                st.pyplot(
                    self._price_chart(
                        st.session_state.prices, st.session_state.current_idx, cis
                    )
                )

        # Statistics
        if st.session_state.show_stats:
            st.subheader("Statistical Summary")
            cols = st.columns(len(stats))
            for (window, (mu, sigma)), col in zip(stats.items(), cols):
                with col:
                    st.metric(f"{window} Days", f"μ={mu:.4f}, σ={sigma:.4f}")

        # Logs
        if st.session_state.show_logs:
            with st.expander("📜 Event Logs", expanded=True):
                for entry in reversed(
                    st.session_state.log[-20:]
                ):  # Show last 20 entries
                    st.write(entry)

        # Auto-play logic
        if st.session_state.auto_play:
            time.sleep(0.5)  # Adjust speed here
            st.session_state.current_idx += 1
            st.rerun()


# Usage example
def main():
    st.title("Market Dynamics Analysis Suite")
    uploaded_file = st.file_uploader("Upload market data", type=["csv"])

    if uploaded_file:
        suite = AnalysisSuite()
        suite.run(uploaded_file)


if __name__ == "__main__":
    main()
