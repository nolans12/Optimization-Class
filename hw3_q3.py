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


class Car:
    def __init__(self, x = 0, y = 0, v = 30, u_domain = np.array([[-0.1, 0.1]]), plan_time = 2, control_time = 0.2):
        """
        Initialize starting position of car
        """
        self.x = x
        self.y = y
        self.v = v # constant velocity
        self.u = None # control input
        self.u_domain = u_domain # domain of control values possible
        self.control_time = control_time # time step for applying control
        self.plan_time = plan_time # time step for planning control

        self.path = []

    def move(self, u):
        """
        EOM of car
        """
        # Break control_time into 100 smaller steps
        dt = self.control_time / 100
        for _ in range(100):
            self.x = self.x + self.v * dt
            self.y = self.y + self.v * dt * np.sin(u)
            self.path.append((self.x, self.y))

    def plan(self, u, dt):
        """
        Plan the next position of the car given a control input and plan time
        """
        x_new = self.x + self.v * dt
        y_new = self.y + self.v * dt * np.sin(u)
        return x_new, y_new
    
    def objective(self, u, constraints, eval_points = 100, penalty = 1000):
        """
        Objective function

        - Get sum of sq deviation from center of road over plan time at eval_points number of discretization points
        - constraints: list of constraint functions to evaluate
        """
        # Get x,y values of planned trajectory
        times = np.linspace(0, self.plan_time, eval_points)
        planned_points = [self.plan(u, dt) for dt in times]
        x_vals = np.array([p[0] for p in planned_points])
        y_vals = np.array([p[1] for p in planned_points])
            
        total_violation = 0
        # Evaluate each constraint at each point
        for constraint in constraints:
            constraint_vals = np.array([constraint(x, y) for x,y in zip(x_vals, y_vals)])
            # Sum up any constraint violations (positive values)
            total_violation += np.sum(np.maximum(0, constraint_vals))
            
        return np.sum((y_vals - 0)**2) + penalty * total_violation
    
    def optimize(self, constraints, total_time = 2, showCar = False, showHeatmap = False):
        """
        Optimizes the cars control input given the constraints for a total time
        """

        # Set up the plots if showing car
        if showCar:
            plt.ion()
            fig, (ax, ax_control) = plt.subplots(2, 1, height_ratios=[3, 1], figsize=(8, 8))
            
            # Configure main plot
            ax.set_xlim(-5, 160)
            ax.set_ylim(-2, 2)
            ax.scatter([], [], c='blue', label='Car')
            ax.axhline(y=0, color='k', linestyle='-', label='Road')
            ax.vlines(x=30, ymin=-3, ymax=1, color='r', label='Obstacle')
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
            ax.grid(True)
            ax.legend()

            # Configure control plot
            ax_control.set_xlim(0, total_time)
            ax_control.set_ylim(-0.15, 0.15)
            ax_control.set_xlabel('Time (s)')
            ax_control.set_ylabel('Control Input (rad)')
            ax_control.grid(True)
            
            # Initialize control history plot
            control_times = []
            control_values = []
            control_line, = ax_control.plot([], [], 'b-', label='Control Input')
            ax_control.legend()

            # Add time text at top of plot
            time_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

            # Track the colorbar object
            colorbar = None
            
        # Get the number of control steps
        num_steps = int(total_time / self.control_time)

        # Now, loop through the control steps and for each, get an optimal control input
        for i in range(num_steps):
            
            ## Get the best control input, use optimization technique here
            u_opt, NaN = simulated_annealing(lambda u: self.objective(u, constraints), temp_schedule, self.u_domain)

            if showCar:
                # Get the best trajectory for visualization
                times = np.linspace(0, self.plan_time, 100)
                planned_points = [self.plan(u_opt, dt) for dt in times]
                x_vals = np.array([p[0] for p in planned_points])
                y_vals = np.array([p[1] for p in planned_points])
                
                ax.scatter(self.x, self.y, c='blue')
                
                # Update time text
                current_time = i * self.control_time
                time_text.set_text(f'Time: {current_time:.1f}s')
                
                # Update control plot
                control_times.append(current_time)
                control_values.append(u_opt)
                control_line.set_data(control_times, control_values)
                
                if showHeatmap:
                    # Create grid for heatmap
                    x_grid = np.linspace(self.x, self.x + 60, 100)
                    y_grid = np.linspace(-2, 2, 100)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = np.zeros_like(X)

                    # Calculate objective values for each point
                    for ii in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            total_violation = 0
                            if constraints is not None:
                                if not isinstance(constraints, list):
                                    constraints = [constraints]
                                for constraint in constraints:
                                    total_violation += max(0, constraint(X[ii,j], Y[ii,j]))
                            Z[ii,j] = (Y[ii,j] - 0)**2 + 1000 * total_violation

                    # Plot heatmap with alpha for transparency and capped colorbar
                    Z = np.clip(Z, 0, 10)  # Cap values at 100
                    heatmap = ax.imshow(Z, extent=[self.x, self.x + 60, -2, 2],
                                      aspect='auto', origin='lower', alpha=0.5,
                                      cmap='Reds', vmin=0, vmax=10)
                    if i == 0:
                        colorbar = plt.colorbar(heatmap, label='Objective Value')

                # Plot the best trajectory
                ax.plot(x_vals, y_vals, 'g-', alpha=0.9)

                plt.draw()
                plt.pause(0.25)

                # Clear old trajectory
                for line in ax.lines[1:]:  # Keep the first line (road centerline)
                    line.remove()
                if showHeatmap:
                    heatmap.remove()
        
            self.u = u_opt
            self.move(self.u)

        if showCar:
            # plot the ending path as a dashed black line, also label this
            x_vals = np.array([p[0] for p in self.path])
            y_vals = np.array([p[1] for p in self.path])
            ax.plot(x_vals, y_vals, 'k--', alpha=0.9, label='Final Path')
            ax.legend()
            plt.ioff()
            plt.show()

        # Return the path of the car
        return self.path

        
    def optimize_old(self, constraints, total_time = 2, showCar = False, showHeatmap = False):
        """
        Optimizes the cars control input given the constraints for a total time
        """
        # Get the number of control steps
        num_steps = int(total_time / self.control_time)
        # Set up the plot if showing car
        if showCar:
            plt.ion()
            fig, ax = plt.subplots()
            ax.set_xlim(-5, 150)
            ax.set_ylim(-2, 2)
            ax.scatter([], [], c='blue', label='Car')
            ax.axhline(y=0, color='k', linestyle='-', label='Road')
            # Add wall at x=30
            ax.vlines(x=30, ymin=-3, ymax=1, color='r', label='Obstacle')
            # Add wall at x=100
            # ax.vlines(x=100, ymin=3, ymax=-1, color='r', label='Obstacle')
            ax.set_xlabel('x position')
            ax.set_ylabel('y position')
            ax.grid(True)
            ax.legend()

            # Add time text at top of plot
            time_text = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center')

            # Track the colorbar object
            colorbar = None

        # Now, loop through the control steps and for each, get an optimal control input
        for i in range(num_steps):
            # Try each control value and find the best one
            control_options = np.linspace(-0.1, 0.1, 20)
            best_objective = float('inf')
            best_control = 0
            best_trajectory = None

            # Plot shooting paths for each control option
            if showCar:
                ax.scatter(self.x, self.y, c='blue')
                
                # Update time text
                current_time = i * self.control_time
                time_text.set_text(f'Time: {current_time:.1f}s')
                
                if showHeatmap:

                    # Create grid for heatmap
                    x_grid = np.linspace(self.x, self.x + 60, 100)
                    y_grid = np.linspace(-2, 2, 100)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    Z = np.zeros_like(X)

                    # Calculate objective values for each point
                    for ii in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            total_violation = 0
                            if constraints is not None:
                                if not isinstance(constraints, list):
                                    constraints = [constraints]
                                for constraint in constraints:
                                    total_violation += max(0, constraint(X[ii,j], Y[ii,j]))
                            Z[ii,j] = (Y[ii,j] - 0)**2 + 1000 * total_violation

                    # Plot heatmap with alpha for transparency and capped colorbar
                    Z = np.clip(Z, 0, 10)  # Cap values at 100
                    heatmap = ax.imshow(Z, extent=[self.x, self.x + 60, -2, 2],
                                      aspect='auto', origin='lower', alpha=0.5,
                                      cmap='Reds', vmin=0, vmax=10)
                    if i == 0:
                        colorbar = plt.colorbar(heatmap, label='Objective Value')

                for u in control_options:
                    # Get predicted trajectory points
                    times = np.linspace(0, self.plan_time, 100)
                    planned_points = [self.plan(u, dt) for dt in times]
                    x_vals = [p[0] for p in planned_points]
                    y_vals = [p[1] for p in planned_points]
                    # Plot predicted trajectory with dashed black lines
                    ax.plot(x_vals, y_vals, 'k--', alpha=0.3)
                    
                    obj_value = self.objective(u, constraints=constraints)
                    if obj_value < best_objective:
                        best_objective = obj_value
                        best_control = u
                        best_trajectory = (x_vals, y_vals)
                plt.pause(0.5)
                
                # Plot the best trajectory
                if best_trajectory:
                    ax.plot(best_trajectory[0], best_trajectory[1], 'g-', alpha=0.9)
                
                plt.draw()
                plt.pause(0.25)
                test = 1
                
                # Clear all trajectories and heatmap
                for line in ax.lines[1:]:  # Keep the first line (road centerline)
                    line.remove()
                if showHeatmap:
                    heatmap.remove()
            else:
                # If not showing car, still need to find best control
                for u in control_options:
                    obj_value = self.objective(u, constraints=constraints)
                    if obj_value < best_objective:
                        best_objective = obj_value
                        best_control = u
            
            self.u = best_control
            self.move(self.u)

        if showCar:
            plt.ioff()
            plt.show()

# define the constraint, we want to be +1 y position at 30 meters x position
def constraint(x, y):
    # If we are within 0.1 meters x of 30. 
    if np.abs(x - 30) < 1:
        # Constraint is that y must be greater than 1
        if y < 1:
            return 1
        else:
            return 0
    else:
        return 0
    
def constraint2(x, y):
    # If we are within 0.1 meters x of 30. 
    if np.abs(x - 100) < 1:
        # Constraint is that y must be greater than 1
        if y > -1:
            return 1
        else:
            return 0
    else:
        return 0

## Annealing
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


# Now, optimize the car driing for 3 seconds
car = Car()
path = car.optimize(constraints = [constraint], total_time = 5, showCar = True)