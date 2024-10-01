from sgd import stochastic_gradient_descent as sgd
import numpy as np
import matplotlib.pyplot as plt
from SGDtest import fsum, fiprime

if __name__ == "__main__":
    print("takes about 20-30 seconds to run, Im so sorry")
    x0 = -5  # Starting point
    nFunctions = 10000  # Number of fi functions

    # Run Stochastic Gradient Descent
    x_vals = sgd(nFunctions, x0, fiprime, 1000)

    # Evaluate fsum for each x value
    #fsum_vals = [fsum(x_vals[0])]

    fsum_vals = [fsum(x) for x in x_vals]
    # for i in range(len(x_vals)-1):
    #     fsum_vals.append(fsum(x_vals[i + 1]))


    # Plot fsum(x(i)) vs i
    plt.figure()
    plt.plot(fsum_vals, color='blue', label='fsum(x(i))')
    plt.xlabel('Iteration')
    plt.ylabel('fsum(x)')
    plt.title('Stochastic Gradient Descent: fsum(x(i)) vs. Iteration')
    plt.legend()
    plt.grid()
    plt.show()