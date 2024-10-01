import numpy as np


def backtracking(x, delta_x, f, grad):
    alpha  = 0.1
    beta = 0.6

    t = 1
    while f(x + t* delta_x) > f(x) + alpha*t*grad(x) * delta_x:
        t = beta * t

    return t
