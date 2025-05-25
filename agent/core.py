from pathlib import Path
import pickle
import random
import numpy as np

from meta import load_csv, DB_DIR
from utils import get_log_ratios, calculate_rsi
from agent.brain import Brain
from agent.evaluate import fitness, evaluate
from agent.gene import _random_gene
from agent.names import historical_figures
from agent.config import EVAL_WINDOW

MODEL_PATH = "/tmp/{name}.pickle"


def save_object(obj, path=None):
    path = path or Path(MODEL_PATH.format(name=obj.__class__.__name__.lower()))

    with open(path, "wb") as file:
        pickle.dump(obj, file)
    return obj


def load_object(cls_name, path=None):
    path = path or Path(MODEL_PATH.format(name=cls_name.lower()))

    with open(path, "rb") as file:
        obj = pickle.load(file)
    return obj


def load_agent(path=MODEL_PATH):
    return load_object("Agent")


def save_agent(agent, path=MODEL_PATH):
    return save_object(agent)


class Agent:
    def __init__(self, brain=None, init_money=100, dataset=None, index=None):
        self.name = random.choice(historical_figures)

        self.money = init_money

        if dataset is None or index is None:
            # no use-case for this
            if dataset or index:
                print("WARNING - Generating automatic index and dataset for agent")
            dataset, index = _select_dataset()

        self.dataset = dataset
        self.index = index

        self.manager = None
        self.brain = brain or Brain(_random_gene(), len(_get_state(dataset, index)))
        self.brain.agent = self

    def decide(self, information: "np.array"):
        return self.brain.decide(information)

    def remember(self, action):
        return self.manager.remember(self, action)

    def evaluate(self, action):
        return self.manager.evaluate(self, action)

    def get_state(self):
        return _get_state(self.dataset, self.index)

    def advance(self):
        self.index += 1
        if self.index > len(self.dataset):
            dataset, index = _select_dataset()
            self.dataset = dataset
            self.index = index


# XXX maybe do lru cache once we get too much data
class Manager:
    def __init__(self, agents, data_size):
        self.personnel = {}
        self.agents = []
        self.data_size = data_size

        for agent in agents:
            self.report(agent)

    def kill(self, agent):
        self.agents.remove(agent)
        del self.personnel[agent]

    def report(self, agent):
        agent.manager = self
        self.personnel[agent] = {}
        self.agents.append(agent)

    def remember(self, agent, action: "np.array"):
        index = agent.index
        dataset = agent.dataset

        if len(action.shape) > 1:
            raise Exception(
                f"Invalid dimension for parameter for the fitness function for {action}"
            )

        if action.shape[0] == self.data_size:
            # special case
            # it's just the status
            # there is nothing to evaluate
            evaluation = 0
        else:
            # the last index is the thesis of the rule
            # or higher-order rule
            evaluation = fitness(dataset, index, action[-1])

        action_signature = tuple(action)

        act = action_signature in self.personnel[agent]
        if act:
            metadata = self.personnel[agent][action_signature]
            cum = self.personnel[agent][action_signature]["cum"] = (
                metadata["evaluation"] + evaluation
            )
            self.personnel[agent][action_signature]["count"] += 1
            cnt = self.personnel[agent][action_signature]["count"]

            print("INFO - Count", cnt)

            self.personnel[agent][action_signature]["evaluation"] = cum / cnt
        else:
            self.personnel[agent][action_signature] = {
                "evaluation": evaluation,
                "cum": evaluation,
                "count": 1,
            }
        self.personnel[agent][action_signature]["day"] = index
        self.personnel[agent][action_signature]["dataset"] = dataset

        self.personnel[agent][action_signature]["evaluation"] = evaluation

        # for debug/test purposes
        # don't cheat during backtest!
        return evaluation

    def evaluate(self, agent, action):
        action_signature = tuple(action)
        if not action_signature in self.personnel[agent]:
            return 0, 1

        day = self.personnel[agent][action_signature]["day"]
        act_dataset = self.personnel[agent][action_signature]["dataset"]
        # current
        dataset = agent.dataset
        index = agent.index

        # compares dataset's references
        if day + EVAL_WINDOW > index and (id(dataset) == id(act_dataset)):
            # lookahead!
            # A way to avoid is is never allowing
            # the agent to predict something that
            # will happen in less than 5 days
            return 0, 1

        self.personnel[agent][action_signature]["evaluation"]

        return (
            self.personnel[agent][action_signature]["evaluation"],
            self.personnel[agent][action_signature]["count"],
        )

    # utility
    def avg_wealth(self):
        return np.mean([a.money for a in self.agents])


def generate_population_manager(size, state_size, initial_cash):
    agent_swarm = [
        Agent(Brain(_random_gene(), state_size), initial_cash) for _ in range(size)
    ]

    return Manager(agent_swarm, state_size)


def _select_dataset():
    file_path = random.choice(list(DB_DIR.glob("*.csv")))
    fd = open(file_path)
    real_prices = load_csv(fd)
    fd.close()

    start = random.randint(0, len(real_prices) // 2)

    return real_prices, start


def _get_prices(dataset, index, days=EVAL_WINDOW):
    """
    Returns a numpy array with the prices for the last `days` days.
    """
    start, end = max(0, index - (EVAL_WINDOW + 1)), index

    rt = get_log_ratios(dataset[start:end])
    res = np.concatenate((np.zeros(EVAL_WINDOW - len(rt)), rt))

    return res


def _get_rsi(dataset, index, days=None):
    days = days or []
    return np.array(
        [
            (calculate_rsi(dataset, index, period=day)) // 20
            #(calculate_rsi(dataset, index, period=day)) // 10
            for day in days
        ]
    )


def _get_state(dataset, index):
    return np.concatenate(
        (
            # _get_prices(dataset, index),
            [],
            _get_rsi(dataset, index, [7, 14, 30, 90, 360]),
        )
    )


def simulation(manager, generation, rent=True, horizon=300, era=1000):
    """
    Simulate one iteration for each agent.

    The following parameters force the agents to be successful if they want to survive.
    `rent` substracts an amount from the agent's "account"

    `horizon` the rest starts growing linearly and reaches the global maxima after `horizon`
        iterations

    `era` resets the rent every `era` iterations to avoid killing too many agents
    """
    # XXX add tests for manachhhher and brain
    decision_log = []
    dead = []
    for agent in manager.agents:
        state = agent.get_state()

        act = agent.decide(state)

        agent.money += fitness(agent.dataset, agent.index, act)
        if rent:
            agent.money -= min((generation % 1000) / 300, 1)
        if agent.money < 0:
            dead.append(agent)

        agent.advance()

    for agent in dead:
        manager.kill(agent)

    return {
        "population": len(manager.personnel),
        "generation": generation,
        "avg_wealth": manager.avg_wealth(),
        "dead": len(dead),
    }
