from import_libraries import *
from pyomo.environ import *
from pyomo.environ import sqrt

class Asset:
    def __init__(self, v, c, sigma):
        self.v = v
        self.c = c
        self.sigma = sigma

total_Sigma = np.array([[0.09, 0, 0, 0], [0, 6.25, 0, -4], [0, 0, 4, 0], [0, -4, 0, 4]])

# Make a list of the assets available
asset1 = Asset(v=1.7, c=1, sigma=0.3)
asset2 = Asset(v=6, c=2, sigma=2.5)
asset3 = Asset(v=4, c=3, sigma=2)
asset4 = Asset(v=1, c=1.5, sigma=2)

assets = [asset1, asset2, asset3, asset4]

max_investment = 1000

# Create a Pyomo model
model = ConcreteModel()

# Define the variables, for each asset, the amount of money to invest in it
model.shares = Var(range(len(assets)), domain=NonNegativeReals)

# The objective is to maximize the expected profit
def objective_rule(model):
    return sum((assets[i].v - assets[i].c) * model.shares[i] for i in range(len(assets)))

model.objective = Objective(rule=objective_rule, sense=maximize)

# The total amount of money invested is 1000
def investment_constraint_rule(model):
    return sum(assets[i].c * model.shares[i] for i in range(len(assets))) <= max_investment

model.investment_constraint = Constraint(rule=investment_constraint_rule)
    
# Now a constraint that sigma portfolio is less than 1/2 profit
def sigma_constraint_rule(model):
    # To calculate the sigma, use formula given
    epsilon = 1e-6  
    shares_array = np.array([model.shares[i] for i in range(len(assets))])
    sigma_portfolio = sqrt(shares_array.T @ total_Sigma @ shares_array + epsilon)
    return sigma_portfolio <= (0.5 * objective_rule(model))

model.sigma_constraint = Constraint(rule=sigma_constraint_rule)

# Solve the problem using a nonlinear solver
solver = SolverFactory('ipopt')
solver.solve(model)
print("Amount of shares in each asset:")
for i in range(len(assets)):
    shares = model.shares[i].value
    cost = shares * assets[i].c
    print(f"Asset {i+1}: {shares} shares (Total spent: ${cost:.2f})")
print("Expected Profit:", model.objective())

shares_array = np.array([model.shares[i].value for i in range(len(assets))])
calculated_sigma = sqrt(shares_array.T @ total_Sigma @ shares_array)
print("Portfolio Sigma:", calculated_sigma)
