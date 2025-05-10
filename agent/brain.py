import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.metrics import cosine_similarity

from gene import PARAMS
from collections import deque


def calculate_reconstruction_error(pca, vector):
    projected = pca.transform(vector_centered.reshape(1, -1))
    reconstructed = pca.inverse_transform(projected)
    error = np.linalg.norm(vector_centered - reconstructed) ** 2

    return error


class Brain:
    def __init__(self, genes: "numpy.array", size, upper: "Brain"=None):
        self.dna = genes
        self.memory = numpy.zeros((size, self.dna[PARAMS["memory"]]))
        self._short_term_free = 0

        self.long_term_memory = PCA(n_components=self.dna[PARAMS["pca_components"]])
        self._long_term_raw = numpy.zeros((size, self.dna[PARAMS["long_term_memory"]]))
        self._long_term_free = 0

        self._reconstruction_error = None

    def remember(self, information):
        if self.dna[PARAMS["memory"]] == self._short_term_free:
            self._short_term_free = 0

        self.memory[self._short_term_free](information)

    @property
    def memory_full(self):
        return self.memory[-1].any()

    def remember_long_term(self, information):
        if self.dna[PARAMS["long_term_memory"]] == self._long_term_free:
            self._long_term_free = 0

        self._long_term_raw[self._short_term_free](information)

    def comprehend(self, information):
        """
          We use PCA to 'comprehend' the information as a whole
        instead of simply memorizing.
        """
        memory_before = len(self.memory)
        self.remember(information)
        memory_after = len(self.memory)

        if memory_before != memory_after:
            # We still don't have enough information
            # to make a proper analysis
            # That is, there is no reason
            # to make costly abstractions
            # and use our long term memory
            # if our short-term memory
            # is not full
            return
        # our short-term memory is full
        # and we forgot something
        # it's time to really understand
        # the information presented to us
        if not hasattr(p, "components_"):
            # empty, great
            self.long_term_memory.fit(self.memory)
            max_mem = min(len(self._long_term_raw), len(self.memory))
            for i in range(0, max_mem):
                self.remember_long_term(self.memory[i])

        # New vector (ensure it's centered using the training mean!)
        vector_centered = information - self.long_term_memory.mean_
        error = calculate_reconstruction_error(vector_centered, self.long_term_memory)

        if not self._reconstruction_error:
            self._reconstruction_error = np.percentile(
                calculate_reconstruction_error(self._long_term_raw),
                self.dna["pca_error_tolerance"],
            )

        # small errors (i.e. 5 percentile) are not relevant
        # because that means the vector was fully comprehended!
        if error < self._reconstruction_error:
            return

        # Outlier detected
        # Here might take different approaches to fit new data
        # 1. Incremental SVD
        # 2. Re-train with long-term memory
        # 3. Re-calculate covariance matrix
        # 4. Any of the above with weights to prioritize new
        #  information

        remember_long_term(information)

        # repeated code
        self.long_term_memory = PCA(n_components=self.dna[PARAMS["pca_components"]])
        self.long_term_memory.fit(self._long_term_raw)
        # invalidate cache
        self._reconstruction_error = None

    def compare(self, information):
        abstract_information = self.long_term_memory.transform(information)
        abstract_memory = self.long_term_memory.transform(self.memory)
        sim_scores = cosine_similarity(abstract_information, abstract_memory)

        return sim_scores, ()

    def evaluate(self, cluster, sim_score):
        """
        Evaluate results from lower layers.
        """
        # i.e. calculate loss function
        # set priorities
        pass

    def create(self, cluster, sim_scores):
        # generate an hyphothesis (higher-dimension information) based on the information
        # currently available, generating new knowledge
        cluster = self.evaluate(information, sim_scores[index])

    def decide(self, information):
        # with the new information, make a decision
        self.comprehend(information)
        sim_scores, cluster = self.compare(information)
        self.create(cluster, sim_scores)
