# examples for dealing with missing data

import pandas as pd                          # package for reading data frames
import numpy as np                           # needed for "not a number"
from sklearn.impute import SimpleImputer as simp   # learning package

df = pd.read_csv("m5_2_2_imputing.txt")          # read the data frame
print("original data")                       # and show how it looks
print(df,"\n")

print("Find NULL values")
print(df.isnull(),"\n")                      # T if entry is missing or NaN

print("Find and sum NULL values")
print(df.isnull().sum(),"\n")

input()

df_drop_row =df.dropna(axis=0)               # axis 0 is ROWS, or samples
print("dropped rows with NULL")
print(df_drop_row,"\n")

df_drop_col =df.dropna(axis=1)               # axis 1 is COLUMNS, or features
print("dropped columns with NULL")
print(df_drop_col,"\n")

df_drop_thresh4 =df.dropna(thresh=4)         # how many values must be present
print("drop threshold 4")
print(df_drop_thresh4,"\n")

df_drop_thresh3 =df.dropna(thresh=3)
print("drop threshold 3")
print(df_drop_thresh3,"\n")

df_drop_col_c =df.dropna(subset=['C'])       # restrict analysis to a feature
print("drop only if NULL in column C")
print(df_drop_col_c,"\n")

input()

# can also use median, most_frequent, or constant
# if constant, then also specify fill_value=X.
# default is to produce a copy of the data...
# NOTE: "fit" learns data, or from data
#       "tranform" modifies data
imr = simp(missing_values=np.nan,strategy='mean') # different than text!
imr = imr.fit(df.values)                          # do the analysis
df_imputed = imr.transform(df.values)             # perform the modifications
print("Imputed values")
print(df_imputed,"\n")

print("original data not altered")
print(df,"\n")
