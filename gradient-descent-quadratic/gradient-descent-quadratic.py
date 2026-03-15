import numpy as np

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Minimize f(x) = ax^2 + bx + c using gradient descent.
    """
    
    x = float(x0)

    for _ in range(steps):
        grad = 2 * a * x + b
        x = x - lr * grad

    return float(x)