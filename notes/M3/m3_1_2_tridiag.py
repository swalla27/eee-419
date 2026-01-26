# Example for filling the diagnonals of a matrix
# author: sdm

import numpy as np         # import matrix functions

# Define the matrix
N = 5                      # dimensions
DIAG = 5                   # value for the main diagonal
UPPER = 3                  # value for the upper diagonal
LOWER = -2                 # value for the lower diagonal

# create a matrix and fill it with values... the hard way
hard_way = np.zeros([N,N],float)      # initialize to all zero
for i in range(1,N-1):                # for all the rows...
    hard_way[i,i] = DIAG              # the diagonal values
    hard_way[i,i+1] = UPPER           # the upper diagonal
    hard_way[i,i-1] = LOWER           # the lower diagonal

# now fix up the first and last row of the hard_way
hard_way[0,0] = DIAG
hard_way[N-1,N-1] = DIAG
hard_way[0,1] = UPPER
hard_way[N-1,N-2] = LOWER

print("\n\n\n")        # some blank lines for separation...
print(hard_way)
print("\n\n\n")        # some blank lines for separation...

# Now, the easy way using eye...
# The np.eye function creates a matrix with 1s on the diagonal and 0s elsewhere.
# However, the k input can be used to modify which "diagonal" is used.
# If k=0, or is not specified, then the main diagonal gets the ones.
# If k>0, then the diagonal k spots above the main diagonal gets the ones.
# If k<0, then the diagonal k spots below the main diagonal gets the ones.
# The first argument is the number of rows.
# If not specified, the number of columns is the same as the number of rows.
# Otherwise, the number of columns is the second entry.
# dtype can also be specified - the default is float.

easy_way = np.eye(N,N)*DIAG + np.eye(N,N,k=1)*UPPER + np.eye(N,N,k=-1)*LOWER

print(easy_way)
print("\n\n\n")        # some blank lines for separation...

