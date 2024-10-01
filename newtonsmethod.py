from backtracking import backtracking as bt
import numpy as np

def newtonsmeth(x, f, grad, hess):
    epsilon = 0.0001
    x_vals = [x]
    lambda_sqrd = np.transpose(grad(x)) * (1/hess(x)) * grad(x)
    while lambda_sqrd/2 > epsilon:
        delta_x = -1 * (1/hess(x)) * grad(x)
        lambda_sqrd = np.transpose(grad(x)) * (1/hess(x)) * grad(x)
        t = bt(x, delta_x, f, grad)

        x = x + t * delta_x
        x_vals.append(x)

    return x_vals
