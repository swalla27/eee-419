# Spectral method for solving Schrodinger Equation
# author: allee updated by sdm

import numpy as np                 # arrays and math
import matplotlib.pyplot as plt    # plotting
from numpy.linalg import solve     # solve the matrix
from scipy import fftpack          # Discrete sine transform both dst and idst
from timeit import default_timer as timer  # track performance

# constants
HBAR = 1.055e-34                   # Planck's constant in Js/rad
MELE = 9.109e-31                   # mass of electron in kg
L = 1e-8                           # width of box in m 
N = 1000                           # number of spatial steps

# other values
h    = 1e-18                       # s, time step
sig  = 1e-10                       # std dev wave function in m
k    = 5e10                        # wave vector in m-1
a    = L/N                         # spacial step
x0   = L/2.0                       # center point of wave at t0

# This is the constant inside the sin/cos.
# k (the integer, not the wave vector) is not here - included below
cnst = np.pi * np.pi * HBAR / ( 2.0 * MELE * L * L )

# initial wave function
psi = np.empty(N+1,complex)        # create a complex array
psi[0] = 0.0                       # the end points are 0
psi[N] = 0.0
two_sig2 = 2 * sig * sig           # calculate up front
for ii in range(1,N):              # set up the internal points...
    x = ii*a                                   # for each point
    exponent = -(x-x0) * (x-x0) / two_sig2     # calculate the first exponent
    psi[ii] = np.exp(exponent)*np.exp(1j*k*x)  # and then the wave function
 
realpsi = psi.real                 # extract the real part
imagpsi =  psi.imag                # extract the imaginary part
 
c = np.arange(N+1)                 # this takes the role of 'k', the integer

alpha = fftpack.dst(realpsi)       # discrete sine transform to get real coeff
eta = fftpack.dst(imagpsi)         # discrete sine transform to get imag coeff

# pick times to plot - match times from prior examle
# NOTE: we compute each time step directly; no having to go through time steps!

start = timer()                    # start time of first loop
for step in range(0,5001,1000):    # so count by 1000s...
    t = step*h                     # convert to time

    # create an array for the real coeff and populate it
    coeffr = np.empty(N+1,float)
    coeffr = alpha * np.cos(-cnst * c * c * t) - eta * np.sin(-cnst * c * c * t)

    # perform the inverse discrete sine transform to get the wave function
    # at time t for the real part
    psiratt = np.empty(N+1,float)
    psiratt = fftpack.idst(coeffr)

    # create an array for the imaginary coeff and populate it
    coeffi = np.empty(N+1,float)
    coeffi = alpha * np.sin(-cnst * c * c * t) + eta * np.cos(-cnst * c * c * t)

    # perform the inverse discrete sine transform to get the wave function
    # at time t for the imaginary part
    psiiatt = np.empty(N+1,float)
    psiiatt = fftpack.idst(coeffi)

    # add real part to i * imaginary part
    psiatt = np.empty(N+1,complex)
    psiatt = psiratt + 1j*psiiatt

    temp = abs(psiatt)             # take absolute value (magnitude!)
    end = timer()                  # calculate time taken
    diff = end - start
    print('At step {:4d}'.format(step),'after {:5.2f}'.format(diff),'seconds')

    # plot the result - divide by 1000 as the numbers get large
    plt.plot(temp*temp/1000,label="after step "+str(step))
    plt.title('Spectral Method')
    plt.xlabel('x')
    plt.ylabel('wave function/1000')
    plt.legend()
    plt.show()
    start = timer()                # start time of next loop
