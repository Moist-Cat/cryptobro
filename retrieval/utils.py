from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

DIMENSIONS = 100


def not_garbage(t):
    return not (t.strip() == "" or t in (",", ".", ";", ":"))


def preprocess_token(t):
    return (
        t.lower()
        .replace("(", "")
        .replace("\\", "")
        .replace("/", "")
        .replace("`", "")
        .replace("=", "")
        .replace(":", "")
        .replace(",", "")
        .replace("#", "")
        .replace("@", "")
    )


def not_tentatively_garbage(t):
    text = t
    return not (len(text) <= 5)


class Model:
    """
    The general process is:
        fit:
            document -> preprocess -> transform to vector -> svd
        predict:
            fit query -> calculate similarity
    """

    # whitelist
    # AND
    FILTERED = [not_garbage, not_tentatively_garbage]

    def __init__(self):
        """
        Inicializa el modelo de recuperación de información.
        """
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.svd = TruncatedSVD(
            n_components=DIMENSIONS,
        )

        self.lsi = self.svd

        self.documents = []
        self.doc_ids = []

        self.doc_vectors = None

    def _preprocess_text(self, text: str) -> str:
        # nothing for now
        text = text.split(" ")
        res = ""
        for t in text:
            mask = [f(t) for f in self.FILTERED]
            if all(mask):
                res += preprocess_token(t)
        return res

    def _preprocess_query(self, query_text: str):
        return self._preprocess_text(query_text)

    def _query_to_lsi(self, query_text: str):
        """Convert a query to its LSI representation"""
        query_tfidf = self.vectorizer.transform([query_text])
        return self.lsi.transform(query_tfidf)

    def encode(self, query_text: str):
        return self._query_to_lsi(query_text)

    def fit(self, dataset: "List[Document]"):
        if not dataset:
            raise ValueError("This dataset has no defined queries")

        # Process documents
        self.documents = []
        self.doc_ids = []

        for doc in dataset:
            self.doc_ids.append(doc.doc_id)
            self.documents.append(self._preprocess_text(doc.text))

        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)

        # Fit TF-IDF and LSI
        self.doc_vectors = self.lsi.fit_transform(self.tfidf_matrix)

    def predict(self, query_text, top_k=3):
        """
        Predict relvant documents for one query
        """
        print("INFO - Predicting data for query of len", len(query_text))

        query_text = self._preprocess_query(query_text)
        query_vec = self._query_to_lsi(query_text)

        sim_scores = cosine_similarity(query_vec, self.doc_vectors)[0]

        top_results = [
            (float(sim_scores[i]), self.doc_ids[i]) for i in range(len(sim_scores))
        ]

        top_results.sort(reverse=True)

        matched_docs = [r[1] for r in top_results]

        print("INFO - Returning", len(matched_docs), "docs")

        return matched_docs[:top_k]


class Document:
    """
    Dummy class to match the interface
    """

    def __init__(self, identifier, text):
        self.doc_id = identifier
        self.text = text

    def __str__(self):
        return f"<Document ({self.doc_id=}, {self.text=})>"

    __repr__ = __str__
