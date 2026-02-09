# covariance example
# author SDM

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data

iris = pd.read_csv('m5_iris.csv')       # read in the data

# create the covariance
# take the absolute value since large negative are as useful as large positive
cov = iris.cov().abs()
print(cov)
input()

# set the covariance on the diagonal or lower triangle to zero,
# so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)

# We clear the diagonal since the covariance with itself is always 1.

# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.
print(cov.values.shape)

# Note this will be element by element multiplication
cov *= np.tri(*cov.values.shape, k=-1).T
print(cov)
input()

# now unstack it so we can sort things
# note that zeros indicate no covariance OR that we cleared below the
# diagonal. Note that cov_unstack is a pandas series.
cov_unstack = cov.unstack()
print(cov_unstack)
print(type(cov_unstack))
input()

# Sort values in descending order
cov_unstack.sort_values(inplace=True,ascending=False)
print(cov_unstack)
input()

# Now just print the top values
print(cov_unstack.head(5))
input()

# Get the covariance with type
with_type = cov_unstack.get(key="type")
print(with_type)
