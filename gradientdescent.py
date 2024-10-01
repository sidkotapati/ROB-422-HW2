from backtracking import backtracking as bt
import numpy as np

def grad_descent(x, f, grad):
    epsilon = 0.0001
    x_vals = [x]
    #terminate conditions
    while (abs(grad(x))) >= epsilon:
        delta_x = -1*grad(x)
        t = bt(x, delta_x, f, grad)
        x = x + t*delta_x
        x_vals.append(x)

    return x_vals


