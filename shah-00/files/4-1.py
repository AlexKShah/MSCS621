# 4-1.py
# Alex Shah
# Homework 0

from scipy.optimize import fmin

def g(x): return -3.0*x**2+24*x-30

maximum = print(fmin(lambda x: -g(x), 0.0))

# Output:
# Optimization terminated successfully.
#         Current function value: -18.000000
#         Iterations: 29
#         Function evaluations: 58
# [ 4.] <-- Answer

