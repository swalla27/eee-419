# example quad integration
import numpy as np                     # get array functions
from scipy.integrate import quad       # get the integration function
import matplotlib.pyplot as plt        # get plotting functions

POINTS = 10      # number of values of 'a' to use

################################################################################
# Function to be integrated: e^(ax)                                            #
# input:                                                                       #
#    x: the variable to integrate                                              #
#    a: a constant to be used                                                  #
# output:                                                                      #
#    returns the value of the function at                                      #
################################################################################

def my_func(x,a):
    y = np.exp(a*x)
    return y

# initialize arrays to hold the values of a and the integral results
a_vals = np.zeros(POINTS,int)
integrals = np.zeros(POINTS,float)

print("a value   error")

for a_val in range(POINTS):                        # for each of the values
    a_vals[a_val] = a_val                          # set the range value

    # calculate the integral - note two return values!
    integrals[a_val], err = quad(my_func,0,1,args=(a_val))

    print("  ",a_val,"    {:.2e}" .format(err))

# now generate the plot
plt.plot(a_vals,integrals)           # data to plot
plt.xlabel("value of 'a'")           # x-axis
plt.ylabel("e^(ax)")                 # y-axis
plt.title('example 1')               # a title
plt.show()                           # and plot it
