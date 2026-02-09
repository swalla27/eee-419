# correlation example
# author SDM

import numpy as np                      # needed for arrays and math
import pandas as pd                     # needed to read the data

iris = pd.read_csv('m5_iris.csv')       # read in the data

# create the correlation
# take the absolute value since large negative are as useful as large positive
corr = iris.corr().abs()
print(corr)
input()

# set the correlations on the diagonal or lower triangle to zero,
# so they will not be reported as the highest ones.
# (The diagonal is always 1; the matrix is symmetric about the diagonal.)

# We clear the diagonal since the correlation with itself is always 1.

# Note the * in front of the argument in tri. That's because shape returns
# a tuple and * unrolls it so they become separate arguments.
print(corr.values.shape)

# Note this will be element by element multiplication
corr *= np.tri(*corr.values.shape, k=-1).T
print(corr)
input()

# now unstack it so we can sort things
# note that zeros indicate no correlation OR that we cleared below the
# diagonal. Note that corr_unstack is a pandas series.
corr_unstack = corr.unstack()
print(corr_unstack)
print(type(corr_unstack))
input()

# Sort values in descending order
corr_unstack.sort_values(inplace=True,ascending=False)
print(corr_unstack)
input()

# Now just print the top values
print(corr_unstack.head(5))
input()

# Get the correlations with type
with_type = corr_unstack.get(key="type")
print(with_type)
