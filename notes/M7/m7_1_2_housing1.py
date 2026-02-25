# Example housing data - use all parameters
# author: d updated by sdm

import matplotlib.pyplot as plt                       # for plotting
import pandas as pd                                   # for data frame
import seaborn as sns                                 # data analysis
import numpy as np                                    # to compute correlation
from mlxtend.plotting import scatterplotmatrix        # a pair plot function

# read the data, assign column names, and print the first 5 entries
# CRIM    - Per capita crime rate
# ZN      - % of residential land zoned for lots over 25k sq ft
# INDUS   - % of non-retail acres
# CHA     - 1 if on a river; 0 otherwise
# NOX     - Nitric Oxide concentration
# RM      - Average number of rooms
# AGE     - % of owner-occupied built before 1940
# DIS     - Weighted distance to 5 business centers
# RAD     - Index of accessibilty to radial highways
# TAX     - Full-value property tax rate
# PTRATIO - Pupil-teacher ratio
# B       - Measure of population of African descent
# LSTAT   - % of lower status of population
# MEDV    - Median value of owner-occupied homes in $1000s
df = pd.read_csv('notes/M7/m7_1_1_housing.data', header=None, sep='\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
              'RAD','TAX','PTRATIO','B','LSTAT','MEDV']
print(df.head())

# use mlxtend to create a pair plot for 5 of the columns
cols = ['LSTAT','INDUS','NOX','RM','MEDV']
scatterplotmatrix(df[cols].values,figsize=(10,8),names=cols,alpha=.5)
plt.tight_layout()        # spread out the charts a bit
plt.show()

# now compute the correlation coefficients and use seaborn to plot a heat map
# annot indicates whether the value should be shown in each square
# annot_kws is a dictionary for how to present the text in the squares
# fmt is for formatting the text in the squares

cm = np.corrcoef(df[cols].values.T)
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',
                 annot_kws={'size':15},
                 yticklabels=cols,xticklabels=cols)
plt.title('Correlation of Housing Features')
plt.show()
