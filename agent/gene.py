import random

import numpy as np

from agent.brain import Brain
from agent.config import PARAM_SPACE


def _random_gene(param_space=PARAM_SPACE):
    def handle_int(lower, upper):
        u = np.random.uniform(lower, upper)
        return u

    return np.array(
        [handle_int(lower, upper) for (lower, upper) in param_space.values()]
    )


def mutate(dna, mutation_rate=1):
    # Apply Gaussian noise to random genes
    mask = np.random.rand(*dna.shape) < mutation_rate
    dna[mask] += dna[mask] * np.random.normal(0, 0.1, size=np.sum(mask))

    return dna


def biased_crossover(parents, init_agents, init_money):
    """Evolutionary crossover with cognitive experience transfer"""
    if len(parents) < 2:
        return []
    # Calculate selection probabilities based on capital
    capitals = np.array([a.money for a in parents])
    selection_probs = capitals / np.sum(capitals)

    children = []
    while len(children) + len(parents) < init_agents:
        # Select parents with capital bias
        p1, p2 = np.random.choice(parents, 2, p=selection_probs, replace=False)

        # Create child with blended DNA
        # give more importance to parents with more money (more successful)
        child_dna = (p1.brain.dna * p1.money + p2.brain.dna * p2.money) / (
            p1.money + p2.money
        )
        # XXX wrong mutation rate
        child_dna = mutate(child_dna, mutation_rate=np.std([p1.money, p2.money]))

        # Inherit hierarchical cognitive structures
        child_brain = Brain(genes=child_dna, size=p1.brain.size)
        child = p1.__class__(brain=child_brain, init_money=init_money)

        # Experience transfer using fitness-weighted memory
        for parent in [p1, p2]:
            for mem in parent.brain.memory:
                if random.random() < 0.2:
                    child_brain.comprehend(mem)

        # partially initialized blah blah blah
        children.append(child)

    return children
