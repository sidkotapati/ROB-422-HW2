# Import packages.
import cvxpy as cp
import numpy as np


#the optimization function c:
c = np.array([2, 1]).T

a = np.asmatrix([[ 0.7071, -0.7071,  0.7071, -0.7071],
 [0.7071,  0.7071, -0.7071, -0.7071]])
b = np.array([1.5, 1.5,1.0 ,1.0 ])
# Define and solve the CVXPY problem.
x = cp.Variable(2)
prob = cp.Problem(cp.Minimize(c.transpose()@x), [a @ x <= b])
prob.solve()

# Print result.
print("\nThe optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution is")
print(prob.constraints[0].dual_value)