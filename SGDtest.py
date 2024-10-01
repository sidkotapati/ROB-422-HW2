import numpy as np
import matplotlib.pyplot as plt
import time
import random

from sgd import stochastic_gradient_descent
from gradientdescent import grad_descent
from newtonsmethod import newtons_meth

maxi = 10000 #this is the number of functions

def fi(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (np.exp(coef1*x + 0.1) + np.exp(-coef1*x - 0.5) - coef2*x)/(maxi/100)

def fiprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    coef2 = 1 + (6-1)*i/maxi
    return (coef1*np.exp(coef1*x + 0.1) -coef1*np.exp(-coef1*x - 0.5) - coef2)/(maxi/100)

def fiprimeprime(x,i):
    coef1 = 0.01 + (0.5-0.01)*i/maxi
    #coef2 = 1 + (6-1)*i/maxi
    return (coef1*coef1*np.exp(coef1*x + 0.1) +coef1*coef1*np.exp(-coef1*x - 0.5))/(maxi/100)


def fsum(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fi(x,i)
    return sum

def fsumprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprime(x,i)
    return sum

def fsumprimeprime(x):
    sum = 0
    for i in range(0,maxi):
       sum = sum + fiprimeprime(x,i)
    return sum

if __name__ == "__main__":
    #this is just to see the function, you don't have to use this plotting code
    # xvals = np.arange(-10, 10, 0.01) # Grid of 0.01 spacing from -10 to 10
    # yvals = fsum(xvals) # Evaluate function on xvals
    # plt.figure()
    # plt.plot(xvals, yvals) # Create line plot with yvals against xvals

    #this is the timing code you should use
    #start = time.time()
    print("Hello world!")
    #YOUR ALGORITHM HERE#

    x0 = -5


    start_sgd = time.time()
    x_star_sgd = stochastic_gradient_descent(10000, x0, fiprime, 1000)  # Assuming maxi = 10000
    end_sgd = time.time()
    sgd_time = end_sgd - start_sgd

    # Measure Gradient Descent runtime
    start_gd = time.time()
    x_star_gd = grad_descent(x0, fsum, fsumprime)
    end_gd = time.time()
    gd_time = end_gd - start_gd

    # Measure Newton's Method runtime
    start_newton = time.time()
    x_star_newton = newtons_meth(x0, fsum, fsumprime, fsumprimeprime)
    end_newton = time.time()
    newton_time = end_newton - start_newton

    # Print the runtimes
    print(f"Stochastic Gradient Descent Runtime: {sgd_time} seconds")
    print(f"Gradient Descent Runtime: {gd_time} seconds")
    print(f"Newton's Method Runtime: {newton_time} seconds")

    # Calculate fsum(x*) for each method
    fsum_sgd = fsum(x_star_sgd[-1])
    fsum_gd = fsum(x_star_gd[-1])
    fsum_newton = fsum(x_star_newton[-1])

    # Print the fsum(x*) results
    print(f"SGD fsum(x*): {fsum_sgd}")
    print(f"Gradient Descent fsum(x*): {fsum_gd}")
    print(f"Newton's Method fsum(x*): {fsum_newton}")

        #end = time.time()
        #print("Time: ", end - start)

    #plt.show() #show the plot

