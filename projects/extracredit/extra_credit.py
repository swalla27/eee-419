# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 27 February 2026

# Extra Credit Project

# I did not use AI at all to complete this assignment

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import time
import sys
import os

NUM_BITS = 10_000
NUM_FEAT = 3
FREQ = 1 / NUM_FEAT
OMEGA = 2*np.pi*FREQ
SNR_RAT = 1
SIGMA = np.sqrt(0.5 / SNR_RAT)

rng = np.random.default_rng()
random_bits = rng.integers(0, 2, NUM_BITS)

time_array = np.linspace(0, FREQ, NUM_FEAT)

clean_array = np.zeros([NUM_BITS, NUM_FEAT])
for idx, _ in enumerate(clean_array):
    clean_array[idx] = np.cos(OMEGA*time_array + random_bits[idx]*np.pi)

noise_array = rng.normal(0, SIGMA, [NUM_BITS, NUM_FEAT])
dirty_array = clean_array + noise_array

print(clean_array)
print(dirty_array)

freq_array = np.arange(0.01,0.5,0.01)               # array of frequencies

# Remember that matrix products are done right to left!
# And each entry in J is:
# J = Xt * H * inv( Ht * H ) * Ht * X
# where X is the sample array and Xt its transpose
# H is the array of sine and cos entries for a particular frequency
# inv() is the inverse of the contents
# and * is the dot product

J = []                                      # maximize J to find f 
h = np.zeros((NUM_FEAT,2))                         # create the H matrix
for f in freq_array:                                # for every frequency to try...
    h[:,0] = np.cos(2 * np.pi * f * time_array)      # column 0 gets the cosines
    h[:,1] = np.sin(2 * np.pi * f * time_array)      # column 1 gets the sines
    a = np.dot(h.transpose(),x)             # Ht * X
    b = inv(np.dot(h.transpose(),h))        # inverse of the product Ht * H
    c = np.dot(b,a)                         # dot the above two terms
    d = np.dot(h,c)                         # and with H
    J.append(np.dot(x.transpose(),d))       # and with Xt and into the list

print(J)
indexmax = np.argmax(J)                     # find the index of largest value
f_est = freq_array[indexmax]                        # then get the corresponding freq
print('freq est', f_est, '\tvs actual', FREQ)

h[:,0] = np.cos(2 * np.pi * f_est * time_array)      # set up H with that frequency
h[:,1] = np.sin(2 * np.pi * f_est * time_array)

# apply least squares estimator
# [alpha1 alpha2] = inv(Ht * H) * Ht * X

a = np.dot(h.transpose(),x)             # Ht * X
b = inv(np.dot(h.transpose(),h))        # inverse of Ht * H
c = np.dot(b,a)                         # product of the above

phase_est = np.arctan(-c[1]/c[0])       # compute the estimated phase
print('phase_est', phase_est, '\tvs actual', phi)
