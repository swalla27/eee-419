# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 2 April 2026

# Homework 7

# I did not use AI at all to complete this assignment.

import numpy as np
import sys

# Define some constants and the precision used for numpy floating point numbers.
# np.set_printoptions(precision=15)
MAX_POINTS = 10_000
NUM_ATTEMPTS = 100

# Define the precision values we will be using, and initialize the numpy random number generator.
precision_values = [10.0**x for x in np.arange(start=-1, stop=-8, step=-1)]
rng = np.random.default_rng()

# Loop over each precision value in that list we just made.
for precision_value in precision_values:

    # Initialize variables for the number of successful attempts and an array holding all the pi values at this precision.
    successful_attempts = 0
    pi_values = list()

    # Begin to loop over the attempt number. There will be NUM_ATTEMPTS iterations.
    for attempt_number in np.arange(start=0, stop=NUM_ATTEMPTS):

        # Initalize a variable for the number of points inside the circle.
        points_inside = 0

        # Begin to loop over the point number. There are a maximum of MAX_POINTS allowed.
        # This is important because the value of pi must be calculated using point_number and not MAX_POINTS.
        for point_number in np.arange(start=1, stop=MAX_POINTS+1):

            # Generate two random numbers between 0 and 1, then place the magnitude of that vector into a variable r.
            xy = rng.uniform(low=0, high=1, size=2)
            r = np.linalg.norm(xy)

            # If r is less than 1, then we are inside the circle. 
            # In that situation, we want to increment the points_inside variable by one to keep track of this.
            if r < 1:
                points_inside += 1
            
            # Find the estimate for pi based on the current information.
            pi_value = 4*points_inside / point_number

            # I am testing how close the estimate is to the actual value of pi. 
            # This statement evaluates to true when the estimate is sufficient.
            if abs(np.pi - pi_value) < precision_value:

                # If the accuracy of pi is sufficient, no point wasting any more time. 
                # Increment the counter for successful attempts and break the j loop.
                successful_attempts += 1
                pi_values.append(pi_value)
                break
    
    # The variable avg_pi is the average of all my pi approximations for this precision.
    # Print the requested information to the terminal. If there were no successful attempts, then print something slightly different.
    # During testing, I tried this program with MAX_POINTS=100_000, and this improved the accuracy considerably.
    # For that reason, I am fairly confident this program does exactly what I want it to.
    if len(pi_values) > 0:
        avg_pi = sum(pi_values) / len(pi_values)
        print(f'{precision_value} success {successful_attempts} times {avg_pi:.15f}')
    else:
        print(f'{precision_value} no success')
