import numpy as np
from sgd import stochastic_gradient_descent as sgd
from SGDtest import fsum, fiprime

# Parameters
nFunctions = 10000  # Number of fi functions
x0 = -5  # Initial starting point
iterations_1000 = 1000
iterations_750 = 750
num_runs = 30

# Function to run SGD and compute fsum(x*) for each run
def run_sgd(nFunctions, iterations, x0):
    fsum_results = []
    for _ in range(num_runs):
        x_vals = sgd(nFunctions, x0, fiprime, iterations)  # Run SGD
        fsum_val = fsum(x_vals[-1])  # Compute fsum at final x (x*)
        fsum_results.append(fsum_val)
    return fsum_results

if __name__ == "__main__":
    # Run SGD for 1000 iterations and collect results
    fsum_1000 = run_sgd(nFunctions, iterations_1000, x0)
    mean_1000 = np.mean(fsum_1000)
    variance_1000 = np.var(fsum_1000)

    # Run SGD for 750 iterations and collect results
    fsum_750 = run_sgd(nFunctions, iterations_750, x0)
    mean_750 = np.mean(fsum_750)
    variance_750 = np.var(fsum_750)

    # Print comparison of results
    print(f"Results for 1000 iterations: Mean = {mean_1000}, Variance = {variance_1000}")
    print(f"Results for 750 iterations: Mean = {mean_750}, Variance = {variance_750}")
