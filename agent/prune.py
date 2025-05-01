class Policy:
    """Trading policy (Intentions filter)"""
    def __init__(self, lower_percentile=33, upper_percentile=67):
        self.lower_p = lower_percentile / 100
        self.upper_p = upper_percentile / 100
        
    def get_bounds(self, current_price, mu, b):
        """Calculate dynamic bounds using Laplace quantiles"""
        lower = current_price * np.exp(laplace.ppf(self.lower_p, loc=mu, scale=b))
        upper = current_price * np.exp(laplace.ppf(self.upper_p, loc=mu, scale=b))
        return lower, upper
