from import_libraries import *


# ## Question 1:

# # Have funciton f(x1, x2) = x1^4 + 3x1^3 + 3x1^2 - 6x1x2 - 2x2.

# # First make the function:
# def f(x1, x2):
#     return x1**4 + 3*x1**3 + 3*x2**2 - 6*x1*x2 - 2*x2

# # Now, make a contour plot of the function over the range [-10, 10] for both x1 and x2
# x1 = np.linspace(-5, 5, 100)
# x2 = np.linspace(-5, 5, 100)

# X1, X2 = np.meshgrid(x1, x2)
# Z = f(X1, X2)

# plt.contour(X1, X2, Z, 1000)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title("Contour plot of the function")

# # Also now label estimated crit points
# plt.plot(-1/4, 1/12, 'ro', label='(-1/4, 1/12)')
# plt.plot(-1 + np.sqrt(3), -2/3 + np.sqrt(3), 'go', label='(-1 + sqrt(3), -2/3 + sqrt(3))')
# plt.plot(-1 - np.sqrt(3), -2/3 - np.sqrt(3), 'bo', label='(-1 - sqrt(3), -2/3 - sqrt(3))')
# plt.legend()
# plt.show()


## Question 2:
# Minimize rosenbrock function

# # First make the function:
# def rosenbrock(X, a=1, b=5):
#     x1 = X[..., 0]
#     x2 = X[..., 1]
#     return (a - x1)**2 + b*(x2 - x1**2)**2

# # Use -2, 2 for x1 and x2
# x1 = np.linspace(-2, 2, 100)
# x2 = np.linspace(-2, 2, 100)

# X1, X2 = np.meshgrid(x1, x2)
# Z = rosenbrock(X1, X2)

# Z_log = np.log(Z + 1)

# contour = plt.contour(X1, X2, Z_log, 10)
# plt.xlabel('x1')
# plt.ylabel('x2')
# plt.title("Rosenbrock function")

# # Add a color bar to represent the scale of potential
# plt.colorbar(contour, label='Log Scale Potential')

# plt.show()


## LETS TRY BASIC LINE SEARCH GRADIENT DESCENT, WHERE THE STEP SIZE IS OPTIMIZED USING BACKTRACKING

def basic_line_search_gradient_descent(func, x0, tol=1e-5, showPlot = False):

    if showPlot:
        x = np.linspace(-10, 10, 250)
        X1, X2 = np.meshgrid(x, x)
        X = np.stack([X1, X2], axis=-1)
        Z = func(X)
        Z_log = np.log(Z + 1)

        plt.ion()
        fig, ax = plt.subplots()
        contour = ax.contour(X1, X2, Z_log, 10)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("Line Search Gradient Descent With Backtracking")
        plt.colorbar(contour, label='Log Scale Potential')
        point, = ax.plot([], [], 'k.')
        ax.plot(x0[0], x0[1], 'rx', markersize=5)

    # First, we will get the gradient of the function at the starting point
    grad = jax.grad(func)(x0)

    # Now, we will keep iterating until the gradient is less than the tolerance
    i = 0
    while True and i < 5000:
        i += 1

        # Get the direction of the gradient
        direction = -grad

        alpha = 0.1
        decay = 0.9
        
        # Now, run alpha = alpha * decay until armijo condition is met
        while func(x0 + alpha * direction) > func(x0) + 1e-4 * alpha * np.dot(grad, direction):
            alpha = alpha * decay

        # Now, update the x0 value
        x0 = x0 + alpha * direction

        # Now, get the gradient at the new point
        grad = jax.grad(func)(x0)

        # Check if the gradient is less than the tolerance
        if np.linalg.norm(grad) < tol:
            break

        # print the gradient norm
        if i % 10 == 0:
            print(f'Iteration {i}, gradient norm: {np.linalg.norm(grad)}')

        if showPlot:
            # Also nwo update the title of the plot with the iteration
            ax.set_title(f"Iteration {i}")
            point.set_xdata(np.append(point.get_xdata(), x0[0]))
            point.set_ydata(np.append(point.get_ydata(), x0[1]))
            plt.draw()
            plt.pause(0.1)

    if showPlot:
        plt.ioff()
        plt.show()

    return x0


## NEXT, LETS USE ADAM

class Adam:
    def __init__(self, x, alpha=1, betaV=0.9, betaS=0.999, epsilon=1e-8):
        self.alpha = alpha # Learning rate
        self.betaV = betaV 
        self.betaS = betaS
        self.epsilon = epsilon
        self.k = 0 # Step counter
        self.v = np.zeros(len(x)) # First moment estimate
        self.s = np.zeros(len(x)) # Second moment estimate

    def step(self, func, x):
        grad = jax.grad(func)(x)
        self.v = self.betaV * self.v + (1 - self.betaV) * grad
        self.s = self.betaS * self.s + (1 - self.betaS) * grad**2
        self.k += 1
        v_hat = self.v / (1 - self.betaV**self.k)
        s_hat = self.s / (1 - self.betaS**self.k)
        return x - (self.alpha * v_hat) / (np.sqrt(s_hat) + self.epsilon) 

    def solve(self, func, x, tol=1e-4, showPlot = False):
        if showPlot:
            x_plot = np.linspace(-10, 25, 100)
            X1, X2 = np.meshgrid(x_plot, x_plot)
            X = np.stack([X1, X2], axis=-1)
            Z = func(X)
            Z_log = np.log(Z + 1)

            plt.ion()
            fig, ax = plt.subplots()
            contour = ax.contour(X1, X2, Z_log, 20)
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.title("Line Search Gradient Descent With Backtracking")
            plt.colorbar(contour, label='Log Scale Potential')
            point, = ax.plot([], [], 'k.')
            ax.plot(x[0], x[1], 'rx', markersize=5)
            
        while True:
            x = self.step(func, x)
            grad = jax.grad(func)(x)
            # Print norm of gradient
            print(f'Gradient norm: {np.linalg.norm(grad)}')
            if np.linalg.norm(grad) < tol:
                break

            # Update plot
            if showPlot:
                # Also nwo update the title of the plot with the iteration
                ax.set_title(f"Iteration {self.k}")
                point.set_xdata(np.append(point.get_xdata(), x[0]))
                point.set_ydata(np.append(point.get_ydata(), x[1]))
                plt.draw()
                plt.pause(0.05)

        if showPlot:
            plt.ioff()
            plt.show()

        return x


# def rosenbrock(X, a=1, b=5):
#     x1 = X[..., 0]
#     x2 = X[..., 1]
#     return (a - x1)**2 + b*(x2 - x1**2)**2

# # Now, we will try to minimize the rosenbrock function using the basic line search gradient descent
# x0 = np.array([-1.2, 1])

# # x_star = basic_line_search_gradient_descent(rosenbrock, x0, tol = 10**-4, showPlot=True)

# adam_optimizer = Adam(x0, alpha = 0.1)
# x_star = adam_optimizer.solve(rosenbrock, x0, tol = 10**-4, showPlot = True)

# print(f'The minimum of the rosenbrock function is at: {x_star}')

# # Use the Branin function instead
def branin(X):
    x1 = X[..., 0]
    x2 = X[..., 1]
    a = 1
    b = 5.1 / (4 * jnp.pi**2)
    c = 5 / jnp.pi
    r = 6
    s = 10
    t = 1 / (8 * jnp.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * jnp.cos(x1) + s

x0 = np.array([7.5, 15.0])

adam_optimizer = Adam(x0, alpha = 0.5)
x_star = adam_optimizer.solve(branin, x0, tol = 10**-4, showPlot = True)

# print(f'The minimum of the rosenbrock function is at: {x_star}')

## Question 3:
# Solving the spring minimization problem

# Parameters for potential energy calculation
ls = [1, 1, 2]
ks = [3, 12, 94]
ps = [[0, 0, 0], [1, 0, 0], [0, 0, 2]]

# Potential energy function
def potential_energy(X, w=30, ls=jnp.array(ls), ks=jnp.array(ks), ps=jnp.array(ps)):
    x = X[..., 0]
    y = X[..., 1]
    z = X[..., 2]
    
    distances = jnp.sqrt((x - ps[:, 0])**2 + (y - ps[:, 1])**2 + (z - ps[:, 2])**2)
    spring_energy = 0.5 * ks * (distances - ls)**2
    total_spring_energy = jnp.sum(spring_energy)
    
    return total_spring_energy + w * y

# Initial point
x0 = np.array([0, 5.0, 0.0])

# Initialize and run the Adam optimizer
adam_optimizer = Adam(x0, alpha=0.5)
x_star = adam_optimizer.solve(potential_energy, x0, tol=1e-4)

print(f'The minimum of the spring potential energy function is at: {x_star}')
print(f'The minimum potential energy is: {potential_energy(x_star)}')
