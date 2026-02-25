# Example housing data - use all parameters
# author: d updated by sdm

import matplotlib.pyplot as plt                       # for plotting
import pandas as pd                                   # for data frame
from sklearn.model_selection import train_test_split  # split the data
from sklearn.linear_model import LinearRegression     # algorithm to use
from sklearn.metrics import mean_squared_error        # data analysis
from sklearn.metrics import r2_score                  # data analysis

# read the data, assign column names
df = pd.read_csv('notes/M7/m7_1_1_housing.data', header=None, sep='\s+')
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
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS',
              'RAD','TAX','PTRATIO','B','LSTAT','MEDV']

X = df.iloc[:,:-1].values     # features are all rows and all but last column
y = df['MEDV'].values         # value to predict is the last column

# now do the train/test split
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

# NOTE: LinearRegression works WITHOUT requiring standarization!
slr = LinearRegression()              # instantiate a linear regression tool
slr.fit(X_train,y_train)              # fit the data
y_train_pred = slr.predict(X_train)   # predict the training values
y_test_pred = slr.predict(X_test)     # predict the test values

# plot the train and test residuals vs their predictions.
# The residual is the difference between the predicted and actual values
ax = plt.axes()
ax.set_facecolor('grey')
plt.scatter(y_train_pred, y_train_pred - y_train,
            c='blue', marker='x', label='Training data')
plt.scatter(y_test_pred, y_test_pred - y_test,
            c='orange', marker='+', label='Test data')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0,xmin=-10,xmax=50,lw=2,color='red')
plt.xlim([-10,50])
plt.title('Housing Data Regression Analysis')
plt.show()

# Calculate measures of the performance of the model.
# First, calculate the mean square error of both the train and test set.
# The results indicate that we have overfit!
print('MSE train: %.3f, test: %.3f' % (
    mean_squared_error(y_train,y_train_pred),
    mean_squared_error(y_test,y_test_pred)))

# Now calculate the coefficient of determination, R^2
print('R^2 train: %.3f, test: %.3f' % \
      (r2_score(y_train,y_train_pred),r2_score(y_test,y_test_pred)))
