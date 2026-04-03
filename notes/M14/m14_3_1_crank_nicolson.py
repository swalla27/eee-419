# Crank-Nicolson solution of Schrodinger Equation
# author: allee updated by sdm

import numpy as np                     # import packages
import matplotlib.pyplot as plt
from numpy.linalg import solve
from timeit import default_timer as timer  # track performance

# constants
HBAR = 1.055e-34                       # Planck's constant in Js/rad
MELE = 9.109e-31                       # mass of electron in kg
L    = 1e-8                            # width of box in m 
N    = 1000                            # number of spatial steps

# other values
h   = 1e-18                            # s, time step
sig = 1e-10                            # std dev wave function in m
k   = 5e10                             # wave vector in m-1
a   = L/N                              # spacial step
x0  = L/2.0                            # center point of the wave at t0

# initial wave function
psi = np.empty(N+1,complex)            # create a complex array
psinew = np.empty(N+1,complex)         # and one to hold the computed next step
psi[0] = 0.0                           # The end points are 0
psi[N] = 0.0
two_sig2 = 2 * sig * sig               # calculate up front
for ii in range(1,N):                  # Set up the internal points...
    x = ii*a                                     # for each point
    exponent = -(x-x0) * (x-x0) / two_sig2       # calculate the first exponent
    psi[ii] = np.exp(exponent)*np.exp(1j*k*x)    # and then the wave function
 
# The np.eye function creates a matrix with 1s on the diagonal and 0s elsewhere.
# However, the k input can be used to modify which "diagonal" is used.
# If k=0, or is not specified, then the main diagonal gets the ones.
# If k>0, then the diagonal k spots above the main diagonal gets the ones.
# If k<0, then the diagonal k spots below the main diagonal gets the ones.
# The first argument is the number of rows.
# If not specified, the number of columns is the same as the number of rows.
# Otherwise, the number of columns is the second entry.
# dtype can also be specified - the default is float.

# calculate the common factors
com_fact = h * 1j * HBAR / ( 2 * MELE * a * a )
com_fact_half = com_fact / 2

#create A matrix   
A = np.zeros((N+1,N+1),complex)      # start with all 0
a1 = 1.0 + com_fact                  # calculate the factors
a2 = -com_fact_half

# now build the A matrix tridiagonal
A = np.eye(N+1,N+1,k=-1)*a2 + np.eye(N+1,N+1)*a1 + np.eye(N+1,N+1,k=1)*a2
    
#create B matrix
B = np.zeros((N+1,N+1),complex)
b1 = 1.0 - com_fact                  # calculate the factors
b2 = com_fact_half

# now build the B matrix tridiagonal
B = np.eye(N+1,N+1,k=-1)*b2 + np.eye(N+1,N+1)*b1 + np.eye(N+1,N+1,k=1)*b2

# Note that the time step is already included in the matrices so no need
# to deal with time here - each step in the for loop is the equivalent to
# one time step. We'll create a plot every 1000 time steps so we can see
# how the wave evolves. (And, how long it takes to compute!)

start = timer()             # start time of first loop
for step in range(5001):    # calculate the new value of psi at each step
    v = B.dot(psi.T)        # dot product of the transpose of psi with B
    psinew = solve(A,v)     # and get the new psi (solve creates a NEW matrix!)
    psi = psinew            # copy it back

    if ( step % 1000 ) == 0:  # if at a multiple of 1000, generate a plot
        end = timer()                # calculate time taken
        diff = end - start
        print('At step {:4d}'.format(step),'after {:5.2f}'.format(diff),'seconds')

        temp = abs(psi)              # get magnitude of complex numbers
        plt.plot(temp*temp,label="after step "+str(step))     
        plt.xlabel('x')
        plt.ylabel('wave function')
        plt.legend()
        plt.title('Crank-Nicolson Solution')
        plt.show()

        start = timer()              # start time of next loop
