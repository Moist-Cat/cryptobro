class Beliefs:
    """Agent's beliefs about price dynamics"""
    def __init__(self, mu=0.0, b=0.1, initial_price=100.0):
        # Laplace parameters for log returns
        self.mu = mu          # Location parameter
        self.b = b            # Scale parameter
        self.initial_price = initial_price
