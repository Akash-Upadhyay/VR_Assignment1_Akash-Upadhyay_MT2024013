import numpy as np

def infinite_horizon_parking(G, T, p, max_j=100, tol=1e-6):
    """Solves the infinite horizon parking problem using value iteration."""
    
    V = np.zeros(max_j + 1)  # Initialize V(j) with zeros
    threshold = max_j  # Initialize threshold to a large value
    
    while True:
        V_old = V.copy()
        
        for j in range(max_j):
            # Compute expected cost for both actions
            cost_garage = G + V[0]  # Garage resets streak
            cost_street = p[j] * T + V[min(j+1, max_j)]  # Staying on street
            
            # Update value function
            V[j] = min(cost_garage, cost_street)
        
        # Check for convergence
        if np.max(np.abs(V - V_old)) < tol:
            break

    # Find the threshold j* where it's optimal to switch to garage
    for j in range(max_j):
        if G + V[0] <= p[j] * T + V[j+1]:
            threshold = j
            break

    return V, threshold

# Example Usage
G = 5  # Garage cost
T = 50  # Ticket fine
p = np.linspace(0.01, 1, 100)  # Probabilities increasing with j

V, j_star = infinite_horizon_parking(G, T, p)
print(f"Optimal threshold j* = {j_star}")
