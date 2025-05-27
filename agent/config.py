EVAL_WINDOW = 7

PARAM_SPACE = {
    # abstraction power
    # "pca_components": (6, 6),
    # "pca_components": (2, 10),
    "pca_components": (1, 3),
    # "memory": (20, 20),
    # "memory": (5, 100),
    "memory": (40, 60),
    # XXX top-k
    # "similarity_threshold": (0.999, 0.999),
    # "similarity_threshold": (0.0, 1.0),
    "similarity_threshold": (0.1, 0.3),
}

# in order
PARAMS = {v: i for i, v in enumerate(PARAM_SPACE.keys())}
