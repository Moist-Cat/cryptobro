EVAL_WINDOW = 7

# 1-layer
# 1
# 50
# 0.2

# 2-layer
[3.9718686900687628,94.13383874396523,0.7745745259822914]
[3.849471955132901,176.4184268699369,0.632716453709777]
[2.5165012774856943,137.9934736915067,0.41931479882617473]

# 2-layer

PARAM_SPACE = {
    # abstraction power
    #"pca_components": (3, 3),
    "pca_components": (2, 7),
    #"pca_components": (1, 3),
    #"memory": (176, 176),
    "memory": (5, 300),
    #"memory": (40, 60),
    # XXX top-k
    #"similarity_threshold": (0.63, 0.63),
    "similarity_threshold": (0.0, 1.0),
    #"similarity_threshold": (0.1, 0.3),
}

# in order
PARAMS = {v: i for i, v in enumerate(PARAM_SPACE.keys())}
