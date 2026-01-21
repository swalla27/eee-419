# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Homework 1
# I did not use AI at all to complete this assignment
# https://math.libretexts.org/Courses/Mount_Royal_University/Calculus_for_Scientists_II/2%3A_Techniques_of_Integration/2.5%3A_Numerical_Integration_-_Midpoint%2C_Trapezoid%2C_Simpson%27s_rule

########################################################
##### Problem 1: Quad package for finite integrals #####
########################################################

import scipy.integrate as integrate
import numpy as np
import sys

LOWER_BOUND = 3 # The lower bound for the integral
UPPER_BOUND = 10 # The upper bound for the integral
n = 100_000 # The number of divisions used for the custom integration method, standard integral notation
h = (UPPER_BOUND - LOWER_BOUND) / n # Standard integral notation, this is the width of each rectangle

user_response = input('Input a set of 3 numbers between -10 and 10 (float or int): ')
try:
    # Attempt to extract the three coefficients, and if an exception occurs, stop the whole program
    (a, b, c) = user_response.split()
    a = float(a)
    b = float(b)
    c = float(c)
except:
    print('Invalid entry, please try again.')
    sys.exit()

def cubic_function(x: float, a: float, b: float, c: float) -> float:
    # This function will take x, a, b, and c as inputs. f(x) = ax^3 + bx^2 + cx
    # It will output the result of the cubic function itself, without any integration
    
    return (a*x**3) + (b*x**2) + (c*x)

# Integrate using method 1, discarding the error
method1_result, _ = integrate.quad(cubic_function, LOWER_BOUND, UPPER_BOUND, args=(a, b, c)) 

def midpoint_integration():
    # This is a custom function based on midpoint integration
    # I calculate the area of a bunch of rectangles and add them together

    running_sum = 0 # The running soum will store our area until now
    midpoints = np.arange((h/2)+LOWER_BOUND, UPPER_BOUND+(h/2), h)
    # We want the midpoints to run from h/2 above the lower bound until h/2 beneath the upper bound (np.arange stops one short)
    # They should also increment by h, and this array will have the same number of entries as "n"

    for midpoint in midpoints:
        # The width of the rectangle should always be h
        rect_width = h

        # The height of the rectangle will be the cubic function evaluated at that midpoint
        rect_height = cubic_function(midpoint, a, b, c)

        # Now I'm just adding the area of that new rectangle to whatever we had before
        running_sum += rect_width * rect_height

    return running_sum

method2_result = midpoint_integration() # Just changing the name of the variable so it's easy to see what I'm doing

perc_error = abs((method1_result-method2_result)/method1_result)*100 # Calculates the percent error for the custom integration

# Print the results for Problem 1 to the terminal
print(f'Method 1: I1 = {method1_result:.4f}')
print(f'Method 2: I1 = {method2_result:.4f}')
print(f'Percentage error: {perc_error:.4f}%')

#################################################################
##### Problem 2: Substitution method and infinite integrals #####
#################################################################

