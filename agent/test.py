import unittest

import numpy as np

from agent.brain import Brain
from agent.gene import _random_gene, PARAMS
from agent.core import Manager, Agent, _get_state, _select_dataset, EVAL_WINDOW


class TestBrain(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_memory(self):
        brain = Brain(_random_gene(), size=2)
        items = np.array([[1, 2], [2, 3], [3, 4]])

        for item in items:
            brain.remember(item)

        self.assertIn(np.array([1, 2]), brain.memory)

    def test_fill_memory(self):
        brain = Brain(_random_gene(), size=2)

        items = np.array(list([i, i**2] for i in range(1, len(brain.memory) + 2)))

        c = 0
        while not brain.memory_full:
            brain.remember(items[c])
            c += 1

        brain.remember(np.array([9, 6]))

        self.assertEqual(brain.memory[0][0], 9.0)

    def test_not_enough_data(self):
        brain = Brain(_random_gene(), size=6)
        brain.remember([1, 1, 1, 1, 1, 1])
        brain.comprehend([1, 1, 1, 1, 1, 1])

        self.assertEqual(brain._long_term_raw[0][0], 0)

    def test_comprehend(self):
        gene = _random_gene()
        gene[PARAMS["pca_components"]] = 1
        brain = Brain(gene, size=2)
        items = np.array(list([i, i**2] for i in range(1, len(brain.memory) + 2)))

        c = 0
        while not brain.memory_full:
            brain.remember(items[c])
            c += 1

        last_square = [len(brain.memory) + 2, (len(brain.memory) + 2) ** 2]

        # fit for the first time
        self.assertFalse(brain.comprehend(last_square))

        last_square = [len(brain.memory) + 3, (len(brain.memory) + 3) ** 2]

        self.assertFalse(brain.comprehend(last_square))

        # outlier
        self.assertTrue(brain.comprehend([-26, -84]))

    def test_compare(self):
        gene = _random_gene()
        gene[PARAMS["pca_components"]] = 1
        brain = Brain(gene, size=2)
        items = np.array(list([i, i**2] for i in range(1, len(brain.memory) + 2)))

        c = 0
        while not brain.memory_full:
            brain.remember(items[c])
            c += 1

        last_square = [len(brain.memory) + 2, (len(brain.memory) + 2) ** 2]

        # fit for the first time
        self.assertFalse(brain.comprehend(last_square))

        last_square = [len(brain.memory) + 3, (len(brain.memory) + 3) ** 2]

        res = brain.compare(np.array(last_square))
        # The closest value should be the biggest square generated
        self.assertTrue((brain.memory[res[1][0]] == brain.memory[0]).all())


class TestManager(unittest.TestCase):
    def setUp(self):
        self.size = 2
        self.agent = Agent(Brain(_random_gene(), size=self.size))

    def tearDown(self):
        pass

    def test_remember(self):
        act = np.array([1, 1])
        manager = Manager([self.agent], self.size)
        manager.remember(self.agent, act)

        self.assertEqual(
            manager.personnel[self.agent]["actions"][tuple(act)]["evaluation"], 0
        )

    def test_evaluate(self):
        act = np.array([1, 1, 1])
        manager = Manager([self.agent], self.size)
        manager.remember(self.agent, act)

        eva = manager.personnel[self.agent]["actions"][tuple(act)]["evaluation"]

        self.assertNotEqual(eva, 0)
        # lookahead detected
        self.assertEqual(round(manager.evaluate(self.agent, act), 5), 0)

        for _ in range(EVAL_WINDOW + 1):
            manager.advance(self.agent)

        self.assertEqual(round(eva, 5), round(manager.evaluate(self.agent, act), 5))


class TestCore(unittest.TestCase):
    def test_state(self):
        state = _get_state(*_select_dataset())
        self.assertEqual(
            len(state),
            EVAL_WINDOW,
        )
