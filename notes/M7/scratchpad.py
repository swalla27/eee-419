from scipy.special import factorial
import matplotlib.pyplot as plt
import numpy as np
import sys

n = 1000
y = np.zeros(n)

def implement_function(n: float, m: float):

    result = 0
    for x in np.arange(n, n-m, step=-1):
        result += 1/x
    return result

for m in range(1, n):
    y[m] = 1 / (n-m)
    

plt.plot(y)
plt.show()