from import_libraries import *

## Particle Swarm Optimization

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

        if k == 5:
            test = 1
        if k == 25:
            test = 2

    if showPlot:
        plt.ioff()
        
    return population, y_best


def f(x):
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

    ## Also add the penalty method
        # For y >= x^3
    penalty = 1000
    p = 0 # num of constraints breaking
    if not y >= x**3:
        p += 1
    
    return term1 + term2 + term3 + term4 + penalty * p

bounds = np.array([[-4, 4], [-2, 2]])

population = [Particle(bounds) for _ in range(50)]
pop_final, y_best = particle_swarm_optimization(f, population, w=0.75, c1=1, c2=1, max_iter=100, showPlot=True)

# Output the final value of the function and where its at
print(f'Final value of function: {y_best}')
print(f'Final x value: {pop_final[0].x}')


