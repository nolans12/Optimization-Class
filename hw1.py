from import_libraries import *

## Questions 4:

# Will focus on optimizing f(x) = x^3 + x*cos(3x) + sin(8 x^2) / 2. over [-1, 1]

# First make the function:
def f(x):
    return x**3 + x*jnp.cos(3*x) + jnp.sin(8*x**2)/2.


# Now plot the function and numerically find the minimum using min of 1000 points

x = jnp.linspace(-1, 1, 100)

# Now, is the function lipschitz continous? We need to symbolically diff the function and plot it over the same interval

# Use jax toolbox, to do numerical differentiation
f_prime = jax.grad(f)

# Now plot the derivative of the function
deriv = []
for i in x:
    deriv.append(f_prime(i))

plt.plot(x, deriv, 'r')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.title("Derivative of the function")
plt.show()

# Now, get the max value of the derivative over the interval [-1, 1]
max_deriv = np.max(np.abs(deriv))
max_deriv_index = np.argmax(np.abs(deriv))  # Get the index of the max derivative
max_deriv_x = x[max_deriv_index]  # Get the x value at the max derivative

print(f'The maximum value of the derivative over the interval [-1, 1] is: {max_deriv} at x = {max_deriv_x}')


# Try to use golden section search to find the critical points of the function
def golden_find_a_critical_point(func, a, b, tol=1e-5):
    """
    Inputs are the function we can call, and the bounds we are searching over, a and c, where a < c
    """

    golden = (np.sqrt(5) - 1) / 2

    # Now, we will keep iterating until the difference between the two points is less than the tolerance
    i = 0
    while True and i < 1000:
        i += 1

        # Perform the golden section search
        p1 = b - golden * (b - a)
        p2 = a + golden * (b - a)

        # Have a < p1 < p2 < b
        
        f1, f2 = func(p1), func(p2)

        if f1 < f2:
            # If f1 is less than f2, then know the crit point 
            b = p2
        else: # f1 > f2
            a = p1

        # Check if the difference is less than the tolerance
        if np.abs(a - b) < tol:
            critical = (a + b) / 2
            return critical


# Now, to find all the critical points, will just run quad_search_find_a_critical_point 1000 times
# And keep a bin of all the critical points, removing those within 1e-3 of each other
critical_points = []
bracket = 1
bounds = [-1, 1]
for i in range(1000):
    # Randomly pick a bracket
    sample = np.random.uniform(bounds[0], bounds[1])
    a = sample - bracket
    b = sample + bracket

    # Now, find the critical point
    critical = golden_find_a_critical_point(f, a, b)

    # Append the critical point to the list
    critical_points.append(critical)

# Now also find the crit points of the -funciton
def f_inv(x):
    return -f(x)

for i in range(1000):
    # Randomly pick a bracket
    sample = np.random.uniform(bounds[0], bounds[1])
    a = sample - bracket
    b = sample + bracket

    # Now, find the critical point
    critical = golden_find_a_critical_point(f_inv, a, b)

    # Append the critical point to the list
    critical_points.append(critical)

# Now, remove the critical points that are within 1e-3 of each other
critical_points = np.array(critical_points)
# Sort the critical points
critical_points = np.sort(critical_points)
# Now, remove the critical points that are within 1e-3 of each other
critical_points = critical_points[np.append([True], np.diff(critical_points) > 1e-3)]
# Also remove any critical points that are outside the bounds
critical_points = critical_points[(critical_points > bounds[0]) & (critical_points < bounds[1])]

# # Plot the function:
plt.plot(x, f(x))

# Plot the critical points with a big scatter marker to show them
plt.scatter(critical_points, f(critical_points), s=50, c='r')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Using Golden Section Search to find critical points')
plt.show()


## Now, use the shubert-piyavskii to find the global minimum on the bound -1, 1

# Define a struct for the x and function value of a point
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def get_sp_intersection(A, B, lc):
    """
    Gets the intersection points input given lipschitz constant
    """
    t = ((A.y - B.y) - lc * (A.x - B.x)) / (2 * lc)
    return Point(A.x + t, A.y - t * lc)


def shubert_piyavskii(func, bounds, lc, n, showPlot=False, tol=1e-4):
    """
    Inputs:  
        - func: The function we want to minimize.
        - bounds: The search interval as [min, max].
        - lc: The Lipschitz constant of the function.
        - n: The number of iterations to do the algorithm.
        - showPlot: If True, will plot the sawtooth bound at each iteration.
        - tol: The tolerance for the algorithm.
    """
    
    # Get the bounds a and b 
    a = bounds[0]
    b = bounds[1]

    # Start at mid point
    m = (a + b) / 2

    # Initialize the points on the function
    A = Point(a, func(a))
    B = Point(b, func(b))
    M = Point(m, func(m))

    # Get the initial 5 intersection points: contains 3 points on the function and 2 intersections!
    pts = [A, get_sp_intersection(A, M, lc), M, get_sp_intersection(M, B, lc), B]

    diff = np.inf
    iter_count = 0

    if showPlot:
        plt.figure()

    for n in range(n):

        if diff < tol:
            break

        iter_count += 1

        # We want to expand up from the minimum point, get the index of the min point
        i = pts.index(min(pts, key=lambda x: x.y))

        # Get the function value at the min point and check if it's within tolerance
        P = Point(pts[i].x, func(pts[i].x))
        diff = P.y - pts[i].y  # Difference between function value and SP intersect

        # Get the next two intersection points
        P_prev = get_sp_intersection(pts[i - 1], P, lc)  # Back one
        P_next = get_sp_intersection(P, pts[i + 1], lc)  # Forward one

        # Delete the min point and add the two new intersection points
        pts.pop(i)
        pts.insert(i, P_prev)
        pts.insert(i+1, P)
        pts.insert(i+2, P_next)

        # Plot the current sawtooth bound if requested
        if showPlot:
            plt.clf()  # Clear the plot for the next iteration
            # Plot the function
            x_vals = np.linspace(a, b, 1000)
            y_vals = [func(x) for x in x_vals]
            plt.plot(x_vals, y_vals, label="Function", color="blue")

            # Plot the sawtooth bound
            saw_x = [p.x for p in pts]
            saw_y = [p.y for p in pts]
            plt.plot(saw_x, saw_y, label=f"Iteration {iter_count}", color="red", marker="o", markersize = 2)

            # Highlight the current minimum
            min_pt = min(pts, key=lambda x: x.y)
            plt.scatter(min_pt.x, min_pt.y, color="green", zorder=5, label="Current Min")

            plt.title(f"Shubert-Piyavskii, Iteration {iter_count} - Diff: {diff:.6f}")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.pause(0.1)  # Pause to visually update the plot

    if showPlot:
        plt.show()

    # Return the minimum point found
    return min(pts, key=lambda x: x.y)


# Run the shubert-piyavskii algorithm
global_min = shubert_piyavskii(f, [-1, 1], 7.5, 50, showPlot=True)