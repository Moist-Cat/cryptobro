from brain import Brain

class Agent:
    def __init__(self, brain):
        self.brain = brain

    def interact(self, environment: "np.array"):
        self.brain.comprehend(environment)
        self.brain.create()

        return self.brain.decide(environment)
