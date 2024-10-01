import numpy as np
import random

def stochastic_gradient_descent(i, x, grad, iterations):
    random.seed(101)

    x_vals = [x]
    t = 1
    for z in range(iterations):
        rand_func = random.randint(1,i)
        
        delta_x = -1*grad(x, rand_func)
        x = x + t*delta_x
        x_vals.append(x)

    return x_vals