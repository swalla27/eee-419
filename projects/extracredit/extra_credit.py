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
NUM_SAMP = 20
FREQ = 10e6
PERIOD = 1 / FREQ
OMEGA = 2*np.pi*FREQ
SNR_RAT = 1
SIGMA = np.sqrt(0.5 / SNR_RAT)

rng = np.random.default_rng()
random_bits = rng.integers(0, 2, NUM_BITS)

sample_times = np.linspace(0, PERIOD, NUM_SAMP)

clean_array = np.zeros([NUM_BITS, NUM_SAMP])
for idx, _ in enumerate(clean_array):
    clean_array[idx] = np.cos(OMEGA*sample_times + random_bits[idx]*np.pi)

noise_array = rng.normal(0, SIGMA, [NUM_BITS, NUM_SAMP])
dirty_array = clean_array + noise_array
print(clean_array)
print(noise_array)
print(dirty_array)
# sys.exit()

freq_guesses = np.linspace(FREQ/10, FREQ*10, 100)

def estimate_phase(sample_times: np.array, sample_values: np.array, freq_guesses: np.array):
    """This function will estimate the phase of a sampled waveform.\n
       It requires an array of frequency guesses, sample values, and the times at which those samples were taken."""
    
    J = list()
    h = np.zeros((NUM_SAMP, 2))
    for freq in freq_guesses:
        h[:,0] = np.cos(2*np.pi*freq*sample_times)
        h[:,1] = np.sin(2*np.pi*freq*sample_times)
        a = np.dot(h.T, sample_values)
        b = inv(np.dot(h.T, h))
        c = np.dot(b, a)
        d = np.dot(h, c)
        J.append(np.dot(sample_values.T, d))

    idx_max = np.argmax(J)
    f_est = freq_guesses[idx_max]

    h[:, 0] = np.cos(2*np.pi*f_est*sample_times)
    h[:, 1] = np.sin(2*np.pi*f_est*sample_times)

    a = np.dot(h.T, sample_values)
    b = inv(np.dot(h.T, h))
    c = np.dot(b, a)

    d = np.arctan(abs(c[1]/c[0]))
    phase_est = np.where(c[0] > 0, d, np.pi-d)
    return phase_est

def evaluate_estimate(phase_est: float, correct_bit: float):
    """This function accepts a phase estimate and the correct bit, then outputs 
    a boolean stating whether that bit was interpreted correctly at the receiver."""

    if (phase_est > np.pi/2) or (phase_est < -np.pi/2):
        bit_est = 1
    else:
        bit_est = 0
    
    if bit_est == correct_bit:
        return True
    else:
        return False


bit_errors = 0
for idx, sample in enumerate(dirty_array):
    phase_est = estimate_phase(sample_times, sample, freq_guesses)

    if not evaluate_estimate(phase_est, random_bits[idx]):
        bit_errors += 1

ber = bit_errors / random_bits.size
print(ber)