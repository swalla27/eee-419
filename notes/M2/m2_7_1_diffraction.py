# Example 5.11: diffraction around an edge
# allee updated by millman

#  diffraction intensity p 174
#  at some point x and z given lambda
#  
import numpy as np                            # import required math functions
from scipy.integrate import quad              # quadrature integration
import matplotlib.pyplot as plt               # get plotting functions

LOC_PI = np.pi                                # odd name to avoid collision
LAMBDA = 1.                                   # wavelength, in meters
STEP = 0.01                                   # step size to use
START_X = -5                                  # starting value of x
END_X = 5+STEP                                # ending value of x
Z = 3.                                        # distance beyond edge, in meters

################################################################################
# Function to be integrated for the cosine term                                #
# input:                                                                       #
#    t: the value at which to evaluate the function                            #
# output:                                                                      #
#    returns the value of the function at t                                    #
################################################################################

def c_integrand(t):
    return np.cos( 0.5 * LOC_PI * t**2 )
    
################################################################################
# Function to be integrated for the sine term                                  #
# input:                                                                       #
#    t: the value at which to evaluate the function                            #
# output:                                                                      #
#    returns the value of the function at t                                    #
################################################################################

def s_integrand(t):
    return np.sin( 0.5 * LOC_PI * t**2 )

# NOTE: quad(function, lowerbound, upperbound)

################################################################################
# Function to integrate the cosine term; note that the error term is dropped   #
# input:                                                                       #
#    u: the upper bound to which to integrate                                  #
# output:                                                                      #
#    returns the requested integral                                            #
################################################################################

def cos_integ(u):
    res,err = quad(c_integrand, 0, u)
    return res
    
################################################################################
# Function to integrate the sine term; note that the error term is dropped     #
# input:                                                                       #
#    u: the upper bound to which to integrate                                  #
# output:                                                                      #
#    returns the requested integral                                            #
################################################################################

def sin_integ(u):
    res,err = quad(s_integrand, 0, u)
    return res

################################################################################
# Start of the main program...                                                 #
################################################################################

x = np.arange( START_X, END_X, STEP )        # values of x where we evaluate
num_vals = len(x)                            # how many values are there?
intensity = np.zeros(num_vals,float)         # create array to hold results

u = x * np.sqrt( 2. / ( LAMBDA * Z ) )       # note entire array handled!

for index in range(num_vals):                # for each of the x values...

    # now calculate the intensity
    intensity[index] = ( ( 2 * cos_integ(u[index]) + 1 )**2 +
                         ( 2 * sin_integ(u[index]) + 1 )**2 ) / 8

#print(intensity)

# note method of creating a subscript... Use ^ for a superscript
plt.plot(x,intensity)
plt.xlabel("x(m) for Z=3m and wavelength=1m")
plt.ylabel("wave intensity")
plt.title("Intensity as a fraction of I"+r'${_0}$')
plt.show()
