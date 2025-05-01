def test_strategy(n_simulations=1000):
    """Compare against buy-and-hold strategy"""
    returns_active = []
    returns_bnh = []
    
    for _ in range(n_simulations):
        # Active strategy
        agent = Agent(beliefs, policy)
        sim = MarketSimulator(beliefs)
        results = simulate(agent, sim)
        returns_active.append(results['portfolio'][-1]/10000 - 1)
        
        # Buy-and-hold
        bnh_return = sim.current_price / beliefs.initial_price
        returns_bnh.append(bnh_return - 1)
    
    print(f"Active strategy: {np.mean(returns_active):.2%} vs B&H: {np.mean(returns_bnh):.2%}")
