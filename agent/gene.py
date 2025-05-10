# we don't set something like "max number of rules"
# i.e. cognitive capacity here or memory because
#
# location, lambda
PARAM_SPACE = {
    # abstraction power
    "pca_components": (3, 1),
    "pca_error_tolerance": (0.5, 0.1),
    # regardless of the level
    # how much information can the
    # individual can retain?
    "memory": (10, 4),
    "long_term_memory": (30, 12),
    "similarity_threshold": (0.3, 0.1),
}

# in order
PARAMS = {v: i for i, v in enumerate(PARAM_SPACE.values())}


class GenePool:
    def __init__(self, population_size, param_space=PARAM_SPACE):
        # Genes as vectors in parameter space
        # Example param_space: {'mu': (-0.1, 0.1), 'b': (0.01, 0.2)}
        self.genes = np.array(
            [self._random_gene(param_space) for _ in range(population_size)]
        )

    def _random_gene(self, param_space):
        def handle_int(location, lmd):
            e = np.random.exponential(lmd)
            if isinstance(location, int):
                e = int(e)
            return location + e

        return np.array(
            [handle_int(location, lmd) for (location, lmd) in param_space.values()]
        )

    def mutate(self, mutation_rate=0.05):
        # Apply Gaussian noise to random genes
        mask = np.random.rand(*self.genes.shape) < mutation_rate
        self.genes[mask] += np.random.normal(0, 0.1, size=np.sum(mask))

    def crossover(self, parent_a, parent_b):
        pass
