# example tri() calls

# The tri function creates an array filled with 1s in the shape of a
# triangle. If k is 0, then the diagonal and below are all 1s, the rest 0s.
# If k=-1, the diagonal is 0 but all below the diagonal are 1. If k=-2,
# then the entries below the diagonal are also 0, and so on.
# If k=1, then the entries above the diagonal are 1, etc.

# get the package
import numpy as np

print('main diagonal')
mat = np.tri(5,5)
print(mat)

print('one above')
mat = np.tri(5,5,1)
print(mat)

print('one below')
mat = np.tri(5,5,-1)
print(mat)

print('top right')
mat = np.tri(5,5,-1).T
print(mat)


