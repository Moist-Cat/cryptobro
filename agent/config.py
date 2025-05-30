EVAL_WINDOW = 7

# 1-layer
# sim-s
# 1
# 50
# 0.2

# top-k
[5.416451342162675, 73.30474736180807, 0.1428035133588303, 0.3571300047341119]
[6.100056324111041338, 139.21008707058098, 0.026232150104103975, 0.1514585281521243]

[1.3761325850072001,136.85538138564704,0,0.22755069944673167]

# 2-layer
# sim-s
[3.9718686900687628, 94.13383874396523, 0.7745745259822914]
[3.849471955132901, 176.4184268699369, 0.632716453709777]
[2.5165012774856943, 137.9934736915067, 0.41931479882617473]

# 2-layer

PARAM_SPACE = {
    # abstraction power
    "pca_components": (1, 1),
    #"pca_components": (2, 7),
    # "pca_components": (1, 3),
    "memory": (136, 136),
    #"memory": (5, 100),
    # "memory": (40, 60),
    # "similarity_threshold": (0.63, 0.63),
    #"similarity_threshold": (0.0, 1.0),
    # XXX disabled for now to avoid extreme drawdowns
    "similarity_threshold": (0.0, 0.0),
    # percent of the memory (ceil)
    # The rationale is that we need a general ratio to
    # match it with the memory
    # Yoinking an arbitrary number of vectors isn't easily genera-
    # lizable
    "top_k": (0.22, 0.22)
    #"top_k": (0.0, 1.0),
    # "similarity_threshold": (0.1, 0.3),
}

# in order
PARAMS = {v: i for i, v in enumerate(PARAM_SPACE.keys())}
