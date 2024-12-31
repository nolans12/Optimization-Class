from import_libraries import *

## Want to formulate and solve the optimziation problem of maximizing expected value of portfolio.

class Asset:
    def __init__(self, v, c, sigma):
        self.v = v
        self.c = c
        self.sigma = sigma


total_Sigma = np.array([[0.09, 0, 0, 0], [0, 6.25, 0, -4], [0, 0, 4, 0], [0, -4, 0, 4]])

# Make a list of the assets available
asset1 = Asset(v = 1.7, c = 1, sigma = 0.3)
asset2 = Asset(v = 6, c = 2, sigma = 2.5)
asset3 = Asset(v = 4, c = 3, sigma = 2)
asset4 = Asset(v = 1, c = 1.5, sigma = 2)

assets = [asset1, asset2, asset3, asset4]

max_investment = 1000

# Now, optimization problem:
import pulp

prob = pulp.LpProblem("Portfolio Optimization", pulp.LpMaximize)

# Define the variables, for each asset, the amount of money to invest in it
share1 = pulp.LpVariable('share1', lowBound=0)
share2 = pulp.LpVariable('share2', lowBound=0)
share3 = pulp.LpVariable('share3', lowBound=0)
share4 = pulp.LpVariable('share4', lowBound=0)

# The objective is to maximize the expected profit
# Value - Cost
prob += pulp.lpSum((asset.v - asset.c)*share for asset, share in zip(assets, [share1, share2, share3, share4]))

# Define the constraints, the total amount of money invested is 1000
prob += share1*asset1.c + share2*asset2.c + share3*asset3.c + share4*asset4.c <= max_investment

# Solve the problem
prob.solve()

print("Amount of shares in each asset:")
print("Asset 1:", pulp.value(share1))
print("Asset 2:", pulp.value(share2))
print("Asset 3:", pulp.value(share3))
print("Asset 4:", pulp.value(share4))
print("Expected Profit:", pulp.value(prob.objective))
