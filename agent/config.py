EVAL_WINDOW = 7

PARAM_SPACE = {
    # abstraction power
    "pca_components": (3, 0),
    # "pca_components": (4, 1),
    # Doesn't matter at all LOL (for 3 comp)
    "pca_error_tolerance": (0.9, 0),
    # regardless of the level
    # how much information can the
    # individual can retain?
    # "memory": (10, 4),
    # "memory": (20, 8),
    "memory": (20, 0),
    # "memory": (5, 1),
    # "long_term_memory": (30, 12),
    # "long_term_memory": (60, 24),
    "long_term_memory": (100, 0),
    # "long_term_memory": (10, 1),
    # "similarity_threshold": (0.3, 0.1),
    # "similarity_threshold": (0.95, 0.0),
    "similarity_threshold": (0.95, 0.0),
}

# in order
PARAMS = {v: i for i, v in enumerate(PARAM_SPACE.keys())}
