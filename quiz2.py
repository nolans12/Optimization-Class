from import_libraries import *


## Compute the direcitonal derivative @ [1,1,1] in dir [ 1,2,3]

# First make the function:
def f(X):
    x1 = X[..., 0]
    x2 = X[..., 1]
    x3 = X[..., 2]
    return x1 + 2*x2 + x3**3

# Use jax for the grad
grad_f = jax.grad(f)

# Now compute the gradient at [1,1,1]
X = jnp.array([1.0,1.0,1.0])

grad = grad_f(X)

print(f"The gradient at [1,1,1] is {grad}")

# Now compute the directional derivative in the direction [1,2,3]
dir = jnp.array([1.0,2.0,3.0])
# Normalize the direction
dir = dir / jnp.linalg.norm(dir)


dir_deriv = jnp.dot(grad, dir)

print(f"The directional derivative in the direction [1,2,3] is {dir_deriv}")





