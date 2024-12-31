# Solve the system of equations:

import numpy as np
from scipy.optimize import fsolve
def equations(vars):
    x1, x2, lagrange = vars

    eq1 = 6*x1 + x2 + lagrange
    eq2 = x1 + lagrange + 1
    eq3 = x1 + x2 - 12

    return [eq1, eq2, eq3]

solution = fsolve(equations, [1, 1, 1])
print(solution)
print(equations(solution))