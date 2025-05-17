import random
import numpy as np

from meta import load_csv, DB_DIR
from utils import get_log_ratios
from agent.brain import Brain
from agent.evaluate import fitness, evaluate
from agent.gene import _random_gene
from agent.names import historical_figures

EVAL_WINDOW = 5


class Agent:
    def __init__(self, brain, init_money=100):
        self.name = random.choice(historical_figures)
        self.brain = brain
        self.brain.agent = self

        self.money = init_money

        dataset, index = _select_dataset()

        self.dataset = dataset
        self.index = index

        self.manager = None

    def decide(self, environment: "np.array"):
        return self.brain.decide(environment)

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
            cum = self.personnel[agent][action_signature]["cum"] = (
                act["evaluation"] + evaluation
            )
            self.personnel[agent][action_signature]["count"] += 1
            cnt = self.personnel[agent][action_signature]["count"]

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


def _get_state(dataset, index) -> "np.array":
    start, end = max(0, index - (EVAL_WINDOW * 2 + 1)), index

    rt = get_log_ratios(dataset[start:end])
    res = np.concatenate((np.zeros(EVAL_WINDOW * 2 - len(rt)), rt))

    return res


def simulation(manager, generation):
    # XXX add tests for manachhhher and brain
    decision_log = []
    dead = []
    for agent in manager.agents:
        state = agent.get_state()

        act = agent.decide(state)

        agent.money += fitness(agent.dataset, agent.index, act)
        agent.money -= min(generation / 200, 1)
        if agent.money < 0:
            dead.append(agent)

        agent.advance()

    for agent in dead:
        manager.kill(agent)

    return {
        "population": len(manager.personnel),
        "generation": generation,
        "avg_wealth": manager.avg_wealth(),
    }
