from import_libraries import *
from scipy.optimize import minimize

## Basic Optimal Control Problem

# We have the following dynamics equations:

# - x_k+1 = x_k + 30 * dt
# - y_k+1 = y_k + 30 * dt * sin(u)
# - car is traveling at a constant 30 m/s

# - u between -0.1 and 1 radians every 0.2 seconds
# - plans 2 seconds into the future

# after 30 meters (x direction), need to move 1 m to the left of road

# objective is to stay as close to the center of road as possible, while avoiding obstacle

## is the objective to stay as close to the center of road after the 2 seconds? given constant u for that long?


def objective(u, eval_points = 10):
    """
    Objective function
    - u: control input (list of control inputs)
    - Returns the cost of a control input u
    """
    cost = 0
    y = 2
    for i in range(eval_points):
        y += 30 * 0.2 * np.sin(u[i])
        cost += (y - 2)**2
    return cost

def g(u):
    """
    Constraint function
    - u: control input (list of control inputs)
    - Returns the matrix of g values for each input control u
    """
    # Compute the y values of the car given the control input u
    x = np.zeros(len(u))
    y = np.zeros(len(u)) + 2 
    for i in range(len(u)):
        x[i] = x[i-1] + 30 * 0.2
        y[i] = y[i-1] + 30 * 0.2 * np.sin(u[i])

    # Now, check the constraints
    g_mat = np.zeros((5, len(u)))
    for i in range(len(u)):
        g_mat[0, i] = - u[i] - 0.1
        g_mat[1, i] = u[i] - 0.1
        g_mat[2, i] = - y[i]
        g_mat[3, i] = y[i] - 4
        if x[i] >= 30:
            g_mat[4, i] = -y[i] + 3
        else:
            g_mat[4, i] = 0
    return g_mat

def g_bar(u, lam, rho):
    """
    Bar function for the constraints
    Ensuring only positive values
    """
    g_vals = g(u)
    g_new = np.zeros(g_vals.shape)
    height, width = g_vals.shape
    for i in range(height):
        for j in range(width):
            if g_vals[i, j] >= lam[i, j]/rho:
                g_new[i, j] = g_vals[i, j]
            else:
                g_new[i, j] = lam[i, j]/rho
    return g_new

def augmented_lagrange_method(f, g, x, k_max, rho=1, gamma=2):
    """
    Augmented Lagrange Method
    - f: objective function
    - g: constraint function
    - x: initial vector of control inputs
    - k_max: number of iterations
    - rho: initial penalty scalar
    - gamma: penalty multiplier
    """

    lam = np.zeros((len(g(x)), len(x))) # Lagrange multipliers for each constraint
    for k in range(k_max):
        def p(x):
            g_vals = g_bar(x, lam, rho) # Use the bar func which ensures g(x) is either 0 if active/inactive otherwise positive if violated
            penalty = 0
            # Calculate the penalty for every single constraint at every single time step
            for u in range(len(x)): # for each time step (control)
                for c in range(len(g(x))): # for each constraint
                    penalty += rho/2 * g_vals[c, u]**2 - lam[c, u] * g_vals[c, u]
            return penalty
        
        result = minimize(lambda x: f(x) + p(x), x)
        x = result.x
        
        lam -= rho * g_bar(x, lam, rho)
        rho *= gamma
    return x, lam

# Get optimal control input
u, lam = augmented_lagrange_method(objective, g, np.zeros(10), 25)
print(u)

# Now plot the path of the car following this control input
x = np.zeros(len(u))
y = np.zeros(len(u)) + 2
for i in range(len(u)):
    x[i] = x[i-1] + 30 * 0.2
    y[i] = y[i-1] + 30 * 0.2 * np.sin(u[i])

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot the vehicle path
axs[0].plot(x, y, 'k-', linewidth=2, label='Vehicle Path')
axs[0].set_title('Solution to the optimal control problem')
axs[0].set_xlabel('X Position (m)')
axs[0].set_ylabel('Y Position (m)')
axs[0].set_ylim(0, 4)
axs[0].grid(True)
axs[0].legend()

# Plot the control input u
axs[1].scatter(np.arange(len(u))*0.2, u, color='b', s=50, label='Control Input u')
axs[1].set_title('Control Input over Time')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Control Input (radians)')
axs[1].set_xticks(np.arange(0, len(u)*0.2, 0.2))
axs[1].grid(True)
axs[1].legend()

# Adjust layout and show the plots
plt.tight_layout()
plt.show()


# Format and print the matrix
print("\nLagrange Multipliers, for each constraint at every control input:\n")
for row in lam:
    formatted_row = ["{:10.2e}".format(val) for val in row]
    print(" ".join(formatted_row))