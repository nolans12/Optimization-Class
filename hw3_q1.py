from import_libraries import *


## Question 1:

# Implement two derivative free optimization algorithms.
# One can be copied from class the other from scratch

# Solve the following optimization 

def objective_function(x):
    """
    Implements the optimization function f(x,y) = (4 - 2.1x² + x⁴/3)x² + xy + (-4 + 4y²)y² + 0.1y
    Args:
        x: numpy array of shape (2,) containing [x, y] coordinates
    Returns:
        float: function value at point (x,y)
    """
    x, y = x[0], x[1]
    
    term1 = (4 - 2.1*x**2 + (x**4)/3) * x**2
    term2 = x*y
    term3 = (-4 + 4*y**2) * y**2
    term4 = 0.1*y
    
    return term1 + term2 + term3 + term4

# By hand, simulated annealing

def simulated_annealing(func, temp_schedule, domain, x0 = None, showPlot = False, max_iter = 1000):
    """
    Simulated Annealing Algorithm

    - func to optimize
    - temp schedule for annealing
    - domain of the problem (n_dim x 2) where each row is a min and max bound for a dimension
    - initial point x0 (if None, sample from domain)
    - showPlot (boolean) whether to show plots animating
    """

    if x0 is None:
        # Sample x0 from domain randomly between min and max bounds for each dimension
        x0 = np.random.uniform(domain[:, 0], domain[:, 1], size=domain.shape[0])

    x = x0
    best_x = x0
    best_val = func(x0)

    if showPlot:
        plt.ion()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Create meshgrid for contour plot
        x_grid = np.linspace(domain[0,0], domain[0,1], 100)
        y_grid = np.linspace(domain[1,0], domain[1,1], 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = func([X, Y])
        Z_log = np.log(Z - np.min(Z) + 1)
        
        # Plot contours
        contour = ax.contour(X, Y, Z_log, 20)
        plt.colorbar(contour, label='log(f(x,y))')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Initialize point plot
        accepted, = ax.plot([], [], 'ks', fillstyle='none', label='Accepted')
        rejected, = ax.plot([], [], 'kx', label='Rejected')
        best_point, = ax.plot([best_x[0]], [best_x[1]], 'gs', label='Current Best', markersize=5)
        current_point, = ax.plot([x[0]], [x[1]], 'bs', label='Current Point', markersize=5)
        ax.plot(x0[0], x0[1], 'rs', label='Start', markersize=5)
        ax.legend(bbox_to_anchor=(1.25, 1), loc='upper left')
        plt.tight_layout()

    # Now loop until max_iter or tol is reached
    for k in range(max_iter):

        # Random new sample
        x_new = x + np.random.uniform(domain[:, 0], domain[:, 1], size=domain.shape[0])

        # The delta
        y_delta = func(x_new) - func(x)

        # Update temp
        t = temp_schedule(k)

        # Piece wise acceptance prob
        p = 1 if y_delta < 0 else np.exp(-y_delta/t)

        if np.random.uniform() <= p:
            # Accept the new sample
            x = x_new
            
            # Update best if needed
            current_val = func(x)
            if current_val < best_val:
                best_x = x.copy()
                best_val = current_val
            
            # Update plot
            if showPlot:
                ax.set_title(f'Iteration {k}, T={t:.2f}')
                accepted.set_xdata(np.append(accepted.get_xdata(), x[0]))
                accepted.set_ydata(np.append(accepted.get_ydata(), x[1]))
                best_point.set_xdata([best_x[0]])
                best_point.set_ydata([best_x[1]])
                current_point.set_xdata([x[0]])
                current_point.set_ydata([x[1]])
                plt.draw()
                plt.pause(0.1)
        else:
            # Rejected point
            if showPlot:
                ax.set_title(f'Iteration {k}, T={t:.2f}')
                rejected.set_xdata(np.append(rejected.get_xdata(), x_new[0]))
                rejected.set_ydata(np.append(rejected.get_ydata(), x_new[1]))
                plt.draw()
                plt.pause(0.01)

    # Return the best point found
    return best_x, best_val

def temp_schedule(k, t0 = 1000):
    # Use fast annealing, t_k = t'/k
    return t0/(k + 1)

## Now use particle swarm optimization

# make a particle struct
class Particle:
    def __init__(self, bounds = np.array([[-4, 4], [-2, 2]])):
        # randomly initialize x within bounds
        self.x = np.random.uniform(bounds[:, 0], bounds[:, 1], size=bounds.shape[0])
        self.v = np.zeros(bounds.shape[0])
        self.best_x = self.x.copy()

def particle_swarm_optimization(f, population, max_iter = 100, w = 1, c1 = 1, c2 = 1, bounds = np.array([[-4, 4], [-2, 2]]), max_func_calls = 10000, showPlot=False):
    """
    Particle Swarm Optimization

    - f: function to optimize
    - population: list of Particle objects
    - max_iter: maximum number of iterations
    - w: inertia weight
    - c1 and c2, momentum coefficients
    - bounds: array of [min, max] bounds for each dimension
    - showPlot: whether to show animation of particles
    """

    # Number of states  
    n = len(population[0].x)

    # Initialize best values and function evaluation counter
    x_best, y_best = population[0].x, float('inf')
    num_evals = 0
    
    for particle in population:
        y = f(particle.x)
        num_evals += 1
        if y < y_best:
            x_best = particle.x
            y_best = y

    # Setup plot if requested
    if showPlot:
        plt.ion()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        
        # Create contour plot
        x = np.linspace(bounds[0,0], bounds[0,1], 100)
        y = np.linspace(bounds[1,0], bounds[1,1], 100)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(len(x)):
            for j in range(len(y)):
                Z[i,j] = f(np.array([X[i,j], Y[i,j]]))
        
        Z_log = np.log(Z - np.min(Z) + 1)
        # Create log contour plot
        ax.contour(X, Y, Z_log, levels=20)
        
        # Plot particles
        particles_plot, = ax.plot([p.x[0] for p in population], 
                                [p.x[1] for p in population], 
                                'ro', label='Particles')
        best_point, = ax.plot([x_best[0]], [x_best[1]], 'g*', 
                             markersize=15, label='Global Best')
        
        ax.set_xlim(bounds[0])
        ax.set_ylim(bounds[1])
        ax.legend(loc='upper right', fontsize='small')
        plt.tight_layout()
        plt.draw()

    # Now loop until max_iter
    for k in range(max_iter):

        if num_evals >= max_func_calls:
            break

        for particle in population:
            
            # Random sample 0-1 * n array
            r1 = np.random.rand(n)
            r2 = np.random.rand(n)

            # Update velocity and position
            particle.v = w*particle.v + c1*r1*(particle.best_x - particle.x) + c2*r2*(x_best - particle.x)
            particle.x += particle.v

            y = f(particle.x)
            num_evals += 1
            
            # Global best
            if y < y_best:
                x_best = particle.x.copy()
                y_best = y

            # Local best
            if y < f(particle.best_x):
                num_evals += 1
                particle.best_x = particle.x.copy()
                
        # Update plot
        if showPlot:  # Update every iteration
            ax.set_title(f'Iteration {k} (Evals: {num_evals})')
            particles_plot.set_xdata([p.x[0] for p in population])
            particles_plot.set_ydata([p.x[1] for p in population])
            best_point.set_xdata([x_best[0]])
            best_point.set_ydata([x_best[1]])
            plt.draw()
            plt.pause(0.1)

    if showPlot:
        plt.ioff()
        
    return population, y_best


## Call particale swarm and show it

population = [Particle() for _ in range(50)]
pop_final, y_best = particle_swarm_optimization(objective_function, population, w=0.25, c1=1, c2=1, max_iter=100, showPlot=True)

# Output the final value of the function and where its at
print(f'Final value of function: {y_best}')
x,y = pop_final[0].x
print(f'Final x value: {pop_final[0].x}')

print(x**3)



# ## Monte carlo we want to compare PSO to SA

# # We will cap both of them at 1,000 function evaluations, and run each 100 times, recording the best value found each time

# import time

# domain = np.array([[-4, 4], [-2, 2]])

# sa_best_vals = []
# pso_best_vals = []
# sa_times = []
# pso_times = []

# for _ in range(100):
#     # Run SA and time it
#     start_time = time.time()
#     x_sa, y_sa = simulated_annealing(objective_function, temp_schedule, domain, showPlot=False, max_iter=1000)
#     sa_time = time.time() - start_time
#     sa_best_vals.append(y_sa)
#     sa_times.append(sa_time)

#     # Reset and run PSO with timing
#     start_time = time.time()
#     population = [Particle() for _ in range(50)]
#     population, y_pso = particle_swarm_optimization(objective_function, population, max_iter=100, max_func_calls=1000, showPlot=False)
#     pso_time = time.time() - start_time
#     pso_best_vals.append(y_pso)
#     pso_times.append(pso_time)

# print(f"SA best value: {np.mean(sa_best_vals)}, PSO best value: {np.mean(pso_best_vals)}")
# print(f"Average runtime - SA: {np.mean(sa_times):.3f}s, PSO: {np.mean(pso_times):.3f}s")

# # Make subplots comparing performance and runtime
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# # Performance comparison
# ax1.boxplot([sa_best_vals, pso_best_vals], labels=['Simulated Annealing', 'Particle Swarm Optimization'])
# ax1.set_ylabel('Best Value Found')
# ax1.set_title('Distribution of Best Values Found\nAfter 1000 Function Calls (100 MC runs)')

# # Runtime comparison 
# ax2.boxplot([sa_times, pso_times], labels=['Simulated Annealing', 'Particle Swarm Optimization'])
# ax2.set_ylabel('Runtime (seconds)')
# ax2.set_title('Distribution of Algorithm Runtimes\n(100 runs)')

# plt.tight_layout()
# plt.show()
