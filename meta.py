from pathlib import Path
import functools
import inspect

from scipy.stats import laplace, gennorm

import pandas as pd

DIST = gennorm
#DIST = laplace
DIST_NAME = "gennorm"
#DIST_NAME = "laplace"

BASE_DIR = Path(__file__).parent

DB_DIR = BASE_DIR / "datasets"

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
                f"{param} {' of ' + algo if algo else ''}",
                value=default,
                step=1 if isinstance(default, int) else 0.00001,
                format="%d" if isinstance(default, int) else "%0.5f",
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


def _configure_algorithm(st, algorithms: list, msg="Select alorithm"):
    """
    Allows us to configure an algorithm using Streamlit's GUI tools.
    We select an algorithm from `algorithms` by clicking on one of the radio
    buttons in the app and configure any optional parameters (bool, int, float supported)

    Returns a partial function
    """
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


def load_csv(uploaded_file):
    """
    Load CSV file and preprocess. Receives a file descriptor so
    you must open and close the file yourself.
    """
    if uploaded_file:
        # since we might have read it already
        uploaded_file.seek(0)
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

        return prices.values
    return None
