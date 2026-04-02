# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 2 April 2026

# Homework 7

# I did not use AI at all to complete this assignment.

import numpy as np
import sys

np.set_printoptions(precision=15)

MAX_POINTS = 100_000
NUM_ATTEMPTS = 100
precision_values = [10.0**x for x in np.arange(start=-1, stop=-8, step=-1)]
rng = np.random.default_rng()

for precision_value in precision_values:

    successful_attempts = 0
    pi_values = np.zeros(NUM_ATTEMPTS)
    for i in np.arange(start=0, stop=NUM_ATTEMPTS):

        points_inside = 0
        for j in np.arange(start=1, stop=MAX_POINTS+1):
            xy = rng.uniform(low=0, high=1, size=2)
            r = np.linalg.norm(xy)

            if r < 1:
                points_inside += 1
            
            pi_value = 4*points_inside / j
            if abs(np.pi - pi_value) < precision_value:
                successful_attempts += 1
                break
        
        pi_values[i] = pi_value
    
    avg_pi = np.average(pi_values)
    if successful_attempts > 0:
        print(f'{precision_value} success {successful_attempts} times {avg_pi:.15f}')
    else:
        print(f'{precision_value} no success')