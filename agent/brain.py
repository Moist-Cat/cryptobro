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


        self.long_term_memory = PCA(
            n_components=min(int(self.dna[PARAMS["pca_components"]]), self.size)
        )
        self._long_term_raw = np.zeros(
            (int(self.dna[PARAMS["long_term_memory"]]), size)
        )
        self._long_term_free = 0

        self._reconstruction_error = None

        self.parent = parent
        self.child = child
        self.agent = agent

    def discard_least_important(self, memory):
        """
        Discard the least significant item from memory given it's full
        """
        minima = float("inf")
        minima_idx = self._long_term_free
        for idx, v in enumerate(memory):
            # refcount
            try:
                _, count = self.agent.evaluate(v)
            except:
                breakpoint()
            age = self.current_index - self.memory_age[idx]
            # +1 for each time we iterate the whole moemory
            relevancy = count - (age//len(memory))
            if relevancy < minima:
                minima_idx = idx
                minima = count

        print(f"{minima=}")

        return minima_idx

    @property
    def memory_full(self):
        if len(self.memory) > 100:
            return self.memory[100].any()
        return self.memory_overflow

    @property
    def memory_overflow(self):
        return self.memory[-1].any()

    @property
    def long_term_overflow(self):
        return self._long_term_raw[-1].any()

    def remember(self, information):
        self.memory[self._short_term_free] = information
        self.memory_age[self._short_term_free] = self.current_index

        if self.memory_overflow:
           self._short_term_free = self.discard_least_important(
                self.memory,
            )
        else:
            self._short_term_free += 1
            if self._short_term_free >= len(self.memory):
                self._short_term_free = 0
        self.current_index += 1

    def remember_long_term(self, information):
        self._long_term_raw[self._long_term_free] = information

        # In principle, the long_term_memory should never overflow
        self._long_term_free += 1
        if self._long_term_free >= len(self._long_term_raw):
            self._long_term_free = 0

    def comprehend(self, information):
        """
          We use PCA to 'comprehend' the information as a whole
        instead of simply memorizing.

         Returns true if the vector was "learned" and false
        if it was "understood" using the existent orthogonal basis
        """
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
            try:
                self.long_term_memory.fit(self.memory)
            except Exception as exc:
                raise
            max_mem = min(len(self._long_term_raw), len(self.memory))
            for i in range(0, max_mem):
                self.remember_long_term(self.memory[i])

        # New vector (ensure it's centered using the training mean!)
        vector_centered = information - self.long_term_memory.mean_
        error = calculate_reconstruction_error(self.long_term_memory, vector_centered)

        # print(error, self._reconstruction_error)
        if not self._reconstruction_error:
            self._reconstruction_error = np.percentile(
                calculate_reconstruction_error(
                    self.long_term_memory, self._long_term_raw
                ),
                self.dna[PARAMS["pca_error_tolerance"]],
            )

        # small errors (i.e. 5 percentile) are not relevant
        # because that means the vector was fully comprehended!
        if error < self._reconstruction_error:
            return False

        # Outlier detected
        # Here might take different approaches to fit new data
        # 1. Incremental SVD
        # 2. Re-train with long-term memory
        # 3. Re-calculate covariance matrix
        # 4. Any of the above with weights to prioritize new
        #  information

        self.remember_long_term(information)

        # repeated code
        self.long_term_memory = PCA(
            n_components=min(int(self.dna[PARAMS["pca_components"]]), self.size)
        )
        self.long_term_memory.fit(self._long_term_raw)
        # invalidate cache
        self._reconstruction_error = None

        return True

    def compare(self, information):
        abstract_information = self.long_term_memory.transform(
            information.reshape(1, -1)
        )
        abstract_memory = self.long_term_memory.transform(self.memory)
        sim_scores = cosine_similarity(abstract_information, abstract_memory).reshape(
            -1, 1
        )

        similarity_threshold = self.dna[PARAMS["similarity_threshold"]]

        valid_mask = sim_scores >= similarity_threshold
        valid_indices = np.flatnonzero(valid_mask)

        if valid_indices.size > 0:
            # Get scores for valid entries and sort descendingly
            # sorted_idx = np.argsort(-sim_scores[valid_indices])
            # sorted_idx = np.argsort(-sim_scores[valid_indices
            # closest_indices = valid_indices[sorted_idx]
            closest_indices = valid_indices
            closest_scores = sim_scores[closest_indices]
            # The cluster is the raw memory
            # we need it to evaluate the results later
            cluster = self.memory[closest_indices]
        else:
            closest_indices = np.array([], dtype=int)
            closest_scores = np.array([])
            cluster = np.array([])

        return closest_scores, closest_indices, cluster

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
            #evaluations[i] = (
            #    evaluation * weight * penalize_young(count, desired_count=5)
            #)
            evaluations[i] = evaluation

        # take the closeness into consideration
        # return np.dot(evaluations, sim_scores)[0]
        #
        # k-nn
        #print(evaluations)
        if (evaluations > 0).all():
            return 1
        elif (evaluations < 0).all():
            return -1
        else:
            return 0
        # print(evaluations)
        # return evaluations.mean()

    def create(self, hypothesis, evaluation):
        """
        Generate an hyphothesis (higher-dimension information) based on the information
         currently available, generating new knowledge.
        """
        if evaluation == 0 and not self.child:
            # random if we have no information
            # evaluation = 1
            evaluation = random.choice((1, -1))
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
            return [0]

        closest_scores, closest_indices, cluster = self.compare(information)
        evaluation = self.evaluate(cluster, closest_scores)

        if not cluster.any():
            # no idea
            return [0]

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
        #if self.child and self.child.child:
        if self.child:
            # avoid adding too many levels of nesting
            return None
        brain = Brain(genes=self.dna, size=len(seed), agent=self.agent, child=self)
        brain.remember(seed)
        self.agent

        return brain

    def _evaluate_cot(self, cot):
        # return np.array(cot).mean()
        print(cot)
        p = np.array(cot).prod()
        print(p)
        return p
        # if not p:
        #    return np.array(cot).mean()
        # return p

    def decide(self, information):
        chain_of_thought = self.think(information)

        if any(chain_of_thought):
            res = self._evaluate_cot(chain_of_thought)
        else:
            res = 0

        # update knowledge base with new data
        self.comprehend(information)

        return res
