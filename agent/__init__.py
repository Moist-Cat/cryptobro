class Agent:
    """BDI Agent with configurable components"""
    def __init__(self, beliefs, policy, initial_cash=0):
        self.beliefs = beliefs
        self.policy = policy
        self.portfolio = {'cash': initial_cash, 'assets': 0}
        self.price_history = [beliefs.initial_price]
        
    def decide_action(self, current_price):
        """BDI decision-making pipeline"""
        lower, upper = self.policy.get_bounds(
            current_price, self.beliefs.mu, self.beliefs.b
        )
        
        if current_price < lower:
            return 'buy'
        elif current_price > upper:
            return 'sell'
        return 'wait'
    
    def execute(self, action, current_price):
        """Execute trading action"""
        if action == 'buy' and self.portfolio['cash'] > 0:
            # Buy with all available cash
            bought = self.portfolio['cash'] / current_price
            self.portfolio['assets'] += bought
            self.portfolio['cash'] = 0
            
        elif action == 'sell' and self.portfolio['assets'] > 0:
            # Sell all assets
            proceeds = self.portfolio['assets'] * current_price
            self.portfolio['cash'] += proceeds
            self.portfolio['assets'] = 0
            
    def portfolio_value(self, current_price):
        return self.portfolio['cash'] + self.portfolio['assets'] * current_price

class MarketSimulator:
    """Price simulator with Laplace dynamics"""
    def __init__(self, beliefs):
        self.beliefs = beliefs
        self.current_price = beliefs.initial_price
        
    def step(self):
        """Generate next price using Laplace log-returns"""
        log_return = laplace.rvs(loc=self.beliefs.mu, scale=self.beliefs.b)
        self.current_price *= np.exp(log_return)
        return self.current_price

def simulate(agent, simulator, n_steps=100):
    """Run trading simulation"""
    results = {
        'prices': [simulator.current_price],
        'portfolio': [agent.portfolio_value(simulator.current_price)]
    }
    
    for _ in range(n_steps):
        current_price = simulator.step()
        action = agent.decide_action(current_price)
        agent.execute(action, current_price)
        
        # Record state
        results['prices'].append(current_price)
        results['portfolio'].append(agent.portfolio_value(current_price))
    
    return results
