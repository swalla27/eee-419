import numpy as np

matrix = 5*np.eye(5,5,k=0) + 3*np.eye(5,5,k=1) - 2*np.eye(5,5,k=-1)

print(matrix)