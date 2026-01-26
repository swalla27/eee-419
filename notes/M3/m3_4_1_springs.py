# Example 6.2
# author: allee updated by sdm
# Calculate motion of masses separated by springs

import numpy as np                 # import math/array functions
import matplotlib.pyplot as plt    # plotting functions
from numpy.linalg import solve     # and the matrix solver

# constants
N = 26                             # Number of masses
C = 1.0                            # harmonic force constant
M = 1.0                            # mass of the masses
K = 6.0                            # spring constant
OMEGA = 2.0                        # harmonic force frequency

alpha = 2*K-M*OMEGA*OMEGA          # entry coefficient

# Set up arrays
# First, the array with the coefficients

coeff = np.eye(N,N)*alpha + np.eye(N,N,k=1)*(-K) + np.eye(N,N,k=-1)*(-K) 
coeff[0,0] -= K
coeff[N-1,N-1] -= K

np.set_printoptions(linewidth=120) # make it so rows on one line!
print(coeff)

# Now set up the answers, where the top is C and all else is 0
answer = np.zeros(N,float)         # initialize to 0
answer[0] = C

print(answer)                      # print so we can see it

x = solve(coeff,answer)            # now, solve coeff dot x = v!
print(x)                           # print it...

plt.plot(x,marker='x')                         # and plot it
plt.xlabel("mass number - 1")
plt.ylabel("vibration amplitude")
plt.title("26 Masses with Springs Between Them")
plt.show()
