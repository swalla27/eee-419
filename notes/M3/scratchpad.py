import numpy as np

N = 3
A = np.eye(N,N)*2 - np.eye(N,N,k=1)-np.eye(N,N,k=-1)
eig_val,eig_vec = np.linalg.eigh(A)
print(eig_val)
print(eig_vec)