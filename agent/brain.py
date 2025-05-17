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

        self.long_term_memory = PCA(
            n_components=int(self.dna[PARAMS["pca_components"]])
        )
        self._long_term_raw = np.zeros(
            (int(self.dna[PARAMS["long_term_memory"]]), size)
        )
        self._long_term_free = 0

        self._reconstruction_error = None

        self.parent = parent
        self.child = child
        self.agent = agent

    @property
    def memory_full(self):
        return self.memory[-1].any()

    def remember(self, information):
        if int(self.dna[PARAMS["memory"]]) == self._short_term_free:
            self._short_term_free = 0

        self.memory[self._short_term_free] = information
        self._short_term_free += 1

    def remember_long_term(self, information):
        if int(self.dna[PARAMS["long_term_memory"]]) == self._long_term_free:
            self._long_term_free = 0

        self._long_term_raw[self._long_term_free] = information
        self._long_term_free += 1

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
            except:
                breakpoint()
            max_mem = min(len(self._long_term_raw), len(self.memory))
            for i in range(0, max_mem):
                self.remember_long_term(self.memory[i])

        # New vector (ensure it's centered using the training mean!)
        vector_centered = information - self.long_term_memory.mean_
        error = calculate_reconstruction_error(self.long_term_memory, vector_centered)

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
            n_components=int(self.dna[PARAMS["pca_components"]])
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
        # i.e. calculate loss function
        # set priorities
        if not cluster.any():
            return 0

        evaluations = np.zeros(len(cluster))
        weight = 1 / len(cluster)
        for i in range(len(cluster)):
            evaluation, count = self.agent.evaluate(cluster[i])
            evaluations[i] = evaluation * weight * penalize_young(count)

        # take the closeness into consideration
        #return np.dot(evaluations, sim_scores)[0]
        return evaluations.mean()

    def create(self, information):
        """
        Generate an hyphothesis (higher-dimension information) based on the information
         currently available, generating new knowledge.
        """
        closest_scores, closest_indices, cluster = self.compare(information)
        evaluation = self.evaluate(cluster, closest_scores)
        # print(f"{evaluation=} ; {self.size=}")
        if evaluation == 0:
            # random if we have no information
            evaluation = random.random() * 2 - 1
        # create rule based on current event
        rule = np.concatenate(
            (
                information,
                np.array([evaluation]),
            )
        )

        return rule

    def think(self, information):
        """
        Upwards backpropagation to think of an action.
        """
        if not self.memory_full:
            return []

        rule = self.create(information)
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
        if self.child and self.child.child:
            # avoid adding too many levels of nesting
            return None
        brain = Brain(genes=self.dna, size=len(seed), agent=self.agent, child=self)
        brain.remember(seed)
        self.agent

        return brain

    def _evaluate_cot(self, cot):
        # return np.array(cot).prod()
        return np.array(cot).mean()

    def decide(self, information):
        chain_of_thought = self.think(information)

        if any(chain_of_thought):
            res = self._evaluate_cot(chain_of_thought)
        else:
            res = 0

        # update knowledge base with new data
        self.comprehend(information)

        return res
