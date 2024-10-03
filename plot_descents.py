from gradientdescent import grad_descent
from newtonsmethod import newtons_meth
import matplotlib.pyplot as plt
import numpy as np
import math

def f(x): 
    return (math.e)**(0.5*x + 1) + (math.e)**(-0.5 * x - 0.5) + 5*x

def grad_f(x):
    return  5 - 0.303265*(math.e)**(-0.5*x) + 1.35914*(math.e)**(0.5*x)

def hess_f(x):
    return ((math.e)**(-0.5*x))*(0.151633 + 0.67957*(math.e)**x)

if __name__ == "__main__":
    x0 = 5

    x_grad = grad_descent(x0, f, grad_f)
    x_nm = newtons_meth(x0, f, grad_f, hess_f)

    # Generate values for the objective function over a range
    x_range = np.linspace(-10, 10, 400)
    y_values = [f(x) for x in x_range]

    # Create the first plot: Objective function and optimization method points
    plt.figure(figsize=(10, 5))
    plt.plot(x_range, y_values, color='black', label='Objective Function f(x)', linewidth=2)

    # Plot points from Gradient Descent
    gd_points = [f(x) for x in x_grad]
    plt.plot(x_grad, gd_points, 'ro-', label='Gradient Descent', markersize=5)

    # Plot points from Newton's Method
    nm_points = [f(x) for x in x_nm]
    plt.plot(x_nm, nm_points, 'mo-', label='Newton\'s Method', markersize=5)

    plt.title('Objective Function and Optimization Methods')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid()
    plt.show()

    # Create the second plot: f(x(i)) vs. iteration i
    plt.figure(figsize=(10, 5))
    
    # Values of f at each iteration for Gradient Descent
    gd_values = [f(x) for x in x_grad]
    plt.plot(gd_values, 'ro-', label='f(x) - Gradient Descent', markersize=5)

    # Values of f at each iteration for Newton's Method
    nm_values = [f(x) for x in x_nm]
    plt.plot(nm_values, 'mo-', label='f(x) - Newton\'s Method', markersize=5)

    plt.title('Objective Function Values vs. Iteration')
    plt.xlabel('Iteration i')
    plt.ylabel('f(x(i))')
    plt.legend()
    plt.grid()
    plt.show()

    