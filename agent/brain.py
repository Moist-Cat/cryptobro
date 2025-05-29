from math import log
from collections import deque
import random

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics.pairwise import cosine_similarity

from agent.config import PARAMS


def penalize_young(count, desired_count=5):
    """
    Penalise events that haven't happened a lot by
    softening its effects.
    I.E. If we get a strong buy signal only once,
     don't pay attention to it.

     The algorithm is log_{desired_count} because

     >>> penalize_young(5, desired_count=5)
     1.0
    """
    if count > desired_count**2:
        return 2
    return log(count) / log(desired_count)


def calculate_reconstruction_error(pca, vector_centered):

    projected = pca.transform(
        vector_centered.reshape(1, -1)
        if len(vector_centered.shape) == 1
        else vector_centered
    )
    reconstructed = pca.inverse_transform(projected)
    error = np.linalg.norm(vector_centered - reconstructed) ** 2

    return error


class Brain:
    def __init__(
        self,
        genes: "np.array",
        size,
        layers=None,
        parent: "Brain" = None,
        child: "Brain" = None,
        agent=None,
    ):
        self.dna = genes
        self.size = size
        self.memory = np.zeros((int(self.dna[PARAMS["memory"]]), size))
        self._short_term_free = 0
        # We record how old is the memory to be able to
        # see which memory could be less relevant
        self.memory_age = np.zeros(int(self.dna[PARAMS["memory"]]))
        self.current_index = 0

        self.long_term_memory = None

        self.parent = parent
        self.child = child
        self.agent = agent

        self._cot = []
        self.MAX_CHILDREN = layers or 1

    def discard_least_important(self, memory):
        """
        Discard the least significant item from memory given it's full
        """
        minima = float("inf")
        minima_idx = 0
        for idx, v in enumerate(memory):
            # refcount
            _, count = self.agent.evaluate(v)
            age = self.current_index - self.memory_age[idx]
            # +1 for each time we iterate the whole moemory
            relevancy = count - (age / len(memory))
            if relevancy < minima:
                minima_idx = idx
                minima = relevancy

        #print(f"{minima=}")

        return minima_idx

    @property
    def memory_full(self):
        return self.memory[-1].any()

    def remember(self, information):
        self.memory[self._short_term_free] = information
        self.memory_age[self._short_term_free] = self.current_index

        if self.memory_full:
            self._short_term_free = self.discard_least_important(
                self.memory,
            )
        else:
            self._short_term_free += 1
            if self._short_term_free >= len(self.memory):
                self._short_term_free = 0
        self.current_index += 1

    def comprehend(self, information):
        """
          We use PCA to 'comprehend' the information as a whole
        instead of simply memorizing.

         Returns true if the vector was "learned" and false
        if it was "understood" using the existent orthogonal basis
        """

        # hack to avoid redundancy during the first iterations
        vector = (self.memory == information).prod(axis=1)
        if not vector.any() and not self.memory_full:
            self.remember(information)

        if not self.memory_full:
            # We still don't have enough information
            # to make a proper analysis
            # That is, there is no reason
            # to make costly abstractions
            # and use our long term memory
            # if our short-term memory
            # is not full
            return False

        # our short-term memory is full
        # and we forgot something
        # it's time to really understand
        # the information presented to us
        if not hasattr(self.long_term_memory, "components_"):
            # empty, great
            print("INFO - Memory full. Feeding PCA with the memory")

        self.long_term_memory = PCA(
            n_components=min(int(self.dna[PARAMS["pca_components"]]), self.size)
        )
        self.long_term_memory.fit(self.memory)

        # avoid adding information that looks like what we already have
        error = self.compare(information)
        similarity_scores = error[0]
        if not similarity_scores.any():
            return False
        similarity = similarity_scores.max()
        # i.e. new information
        # if similarity > 0.99:
        #    return False

        #print(self.memory)

        self.remember(information)

        # XXX we ingnore outliers now
        # Outlier detected
        # Here might take different approaches to fit new data
        # 1. Incremental SVD
        # 2. Re-train with long-term memory
        # 3. Re-calculate covariance matrix
        # 4. Any of the above with weights to prioritize new
        #  information

        # repeated code
        return True

    def compare(self, information):
        abstract_information = self.long_term_memory.transform(
            information.reshape(1, -1)
        )
        abstract_memory = self.long_term_memory.transform(self.memory)
        sim_scores = cosine_similarity(abstract_information, abstract_memory).reshape(
            -1, 1
        )

        # top-k
        top_k = min(max(self.dna[PARAMS["top_k"]], 0), 1)
        top_vectors = int(top_k * len(self.memory))
        # get the k biggest vector
        # Is top_vectors ever bigger than sim_scores?
        # Spoiler: no
        top_k_vector = np.argsort(sim_scores)[-top_vectors]

        # truncated-top-k algo
        similarity_threshold = max(top_k_vector, self.dna[PARAMS["similarity_threshold"]])
        # top-k algo
        #similarity_threshold = top_k_vector
        # truncated algo
        #similarity_threshold = self.dna[PARAMS["similarity_threshold"]] # XXX old algo

        valid_mask = sim_scores >= similarity_threshold
        valid_indices = np.flatnonzero(valid_mask)

        # force to have enough information to make an informed decision
        print(len(valid_indices), top_vectors)
        if valid_indices.size > 0 and (len(valid_indices) >= top_vectors):
            closest_indices = valid_indices
            closest_scores = sim_scores[closest_indices]
            # The cluster is the raw memory
            # we need it to evaluate the results later
            cluster = self.memory[closest_indices]
            # update memory age
            self.memory_age[valid_indices] = self.current_index
        else:
            closest_indices = np.array([], dtype=int)
            closest_scores = np.array([])
            cluster = np.array([])

        return closest_scores, closest_indices, cluster

    @property
    def _max_depth_reached(self):
        """
        Meaning this is the last layer
        """
        children = 0
        curr = self.child
        while curr:
            children += 1
            curr = curr.child

        return children >= self.MAX_CHILDREN

    def _experiment(self):
        if not self._max_depth_reached:
            return 1
            # return random.choice((-1, 1))
        return 0

    def evaluate(self, cluster, sim_scores):
        """
        Evaluate results from lower layers.
        """
        # sim scores shouldn't be negative
        # set priorities
        if not cluster.any():
            return 0

        evaluations = np.zeros(len(cluster))
        # weight = 1 / len(cluster)
        weight = 1
        for i in range(len(cluster)):
            evaluation, count = self.agent.evaluate(cluster[i])
            # evaluations[i] = (
            #    evaluation * weight * penalize_young(count + 1, desired_count=30)
            # )
            evaluations[i] = evaluation
        # take the closeness into consideration
        # return np.dot(evaluations, sim_scores)[0]
        #
        # k-nn
        # if (evaluations > 0).all():
        #    return 1
        # elif (evaluations < 0).all():
        #    return -1
        # else:
        # print(cluster)
        # print(evaluations)
        mini = evaluations.min()
        maxim = evaluations.max()
        # med = np.median(evaluations)
        med = np.mean(evaluations)

        # We might have too little information to make
        # an informed decision
        if maxim == mini:
            return np.sign(maxim)
        elif np.sign(maxim) == np.sign(mini):
            return np.sign(maxim)
        elif self._max_depth_reached:
            # zero is in the interval
            neg = -mini / (maxim - mini)
            pos = 1 - neg
            if neg > pos:
                return -(neg - 0.5) * 2
            return (pos - 0.5) * 2
        return np.sign(med)

    def create(self, hypothesis, evaluation):
        """
        Generate an hyphothesis (higher-dimension information) based on the information
         currently available, generating new knowledge.
        """
        if evaluation == 0 and not self._max_depth_reached:
            # random if we have no information
            # evaluation = random.choice((1, -1))
            evaluation = 1
        # create rule based on current event
        rule = np.concatenate(
            (
                hypothesis,
                np.array([evaluation]),  # thesis
            )
        )

        return rule

    def think(self, information):
        """
        Upwards backpropagation to think of an action.
        """
        if not self.memory_full:
            return [self._experiment()]

        closest_scores, closest_indices, cluster = self.compare(information)
        evaluation = self.evaluate(cluster, closest_scores)

        # if not cluster.any():
        #    # no idea
        #    return [self._experiment()]

        # print(evaluation, cluster, closest_scores)
        # rule = self.create(cluster.mean(axis=0), evaluation)
        rule = self.create(information, evaluation)
        parent_feedback = []
        if not self.parent:
            self.parent = self._create_parent(rule)
            result = rule[-1]
        else:
            parent_feedback = self.parent.think(rule)

        if self.parent:
            self.parent.comprehend(rule)
        # log action for future reference
        self.agent.remember(rule)

        return parent_feedback + [rule[-1]]

    def _create_parent(self, seed):
        if self._max_depth_reached:
            # avoid adding too many levels of nesting
            return None
        brain = Brain(genes=self.dna, size=len(seed), agent=self.agent, child=self)
        brain.remember(seed)
        self.agent

        return brain

    def _evaluate_cot(self, cot):
        # return np.array(cot).mean()
        p = np.array(cot).prod()
        print(cot)
        return p
        # if not p:
        #    return np.array(cot).mean()
        # return p

    @property
    def _is_dull(self):
        return len(self.memory) == 0 or (
            self.long_term_memory is not None
            and min(int(self.dna[PARAMS["pca_components"]]), self.size) == 0
        )

    def decide(self, information):
        if self._is_dull:
            return 0
        chain_of_thought = self.think(information)

        self._cot = chain_of_thought

        if any(chain_of_thought):
            res = self._evaluate_cot(chain_of_thought)
        else:
            res = 0

        # update knowledge base with new data
        self.comprehend(information)

        if not not self._max_depth_reached:
            return 0

        return res
