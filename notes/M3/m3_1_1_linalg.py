# Example linear algebra problem
# author: hartin updated by sdm

import numpy as np                 # import math and arrays
from numpy.linalg import solve     # import matrix solver

ANS = 5.0                          # even answer value
N = 4                              # number of equations

# create the coefficient matrix
coeff = np.full([N,N],-1.0,float)  # initialize to all -1 (most common value!)
coeff[0,0] = 4.0                   # put in other values
coeff[1,1] = 3.0
coeff[2,2] = 3.0
coeff[3,3] = 4.0
coeff[1,2] = 0.0
coeff[2,1] = 0.0
print(coeff)                       # print the matrix

# create the result matrax
result = np.zeros(N,float)         # initialize to 0
result[0] = ANS                    # put in other values
result[2] = ANS
print(result)                      # print the matrix

answer = solve(coeff,result)       # get the answers
print(answer)
