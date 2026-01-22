# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Homework 1
# I did not use AI at all to complete this assignment

# I used this reference to understand the procedure behind the midpoint method of definite integrals
# https://math.libretexts.org/Courses/Mount_Royal_University/Calculus_for_Scientists_II/2%3A_Techniques_of_Integration/2.5%3A_Numerical_Integration_-_Midpoint%2C_Trapezoid%2C_Simpson%27s_rule
# I used the integral tables in this second reference when solving the integral in problem 2, it was the 1/a arctan(u/a) one
# https://openstax.org/books/calculus-volume-1/pages/a-table-of-integrals
# I also used this reference when writing my midpoint function because I needed to make it accept an arbitrary number of arguments
# https://www.geeksforgeeks.org/python/args-kwargs-python/

########################################################
##### Problem 1: Quad package for finite integrals #####
########################################################

import scipy.integrate as integrate
import numpy as np
import sys
import time

LOWER_BOUND = 3 # The lower bound for the integral
UPPER_BOUND = 10 # The upper bound for the integral
n = 100_000 # The number of divisions used for the custom integration method, standard integral notation
h = (UPPER_BOUND - LOWER_BOUND) / n # Standard integral notation, this is the width of each rectangle

user_response = input('Input a set of 3 numbers between -10 and 10: ')
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

def midpoint_integration(integrand, *args) -> float:
    # This is a custom function based on midpoint integration
    # I calculate the area of a bunch of rectangles and add them together
    
    # The inputs are the function to be integrated (integrand) and the arguments that function requires (*args)
    # The function will output the definite integral of that function over the specified interval

    running_sum = 0 # The running sum will store the area accumulated so far
    midpoints = np.arange((h/2)+LOWER_BOUND, UPPER_BOUND+(h/2), h)
    # We want the midpoints to run from h/2 above the lower bound until h/2 beneath the upper bound
    # (Note that I used h/2 over the upper bound because np.arange stops one short)
    # They should also increment by h, and this array will have the same number of entries as "n"

    for midpoint in midpoints:
        # The width of the rectangle should always be h
        rect_width = h

        # The height of the rectangle will be the cubic function evaluated at that midpoint
        rect_height = integrand(midpoint, *args)

        # Now I'm just adding the area of that new rectangle to whatever we had before
        running_sum += rect_width * rect_height

    return running_sum # Now that we're done with that for-loop, the running sum is our definite integral

t0 = time.time()
method2_result = midpoint_integration(cubic_function, a, b, c) # Calling the midpoint integration function and storing the result
# This time, I called the midpoint_integration() function using four arguments because the first argument (cubic_function) actually requires those three other arguments
# My functions are set up such that the 2nd, 3rd, and 4th arguments into midpoint_integration() are actually fed into cubic_function() later on
t1 = time.time()

perc_error = abs((method1_result-method2_result)/method1_result)*100 # Calculates the percent error for the custom integration

# Print the results for Problem 1 to the terminal
print('\nPROBLEM ONE')

# This was 0.05 seconds on my machine. This one should be really fast no matter what
print(f'The custom function to evaluate this integral took {t1-t0:.2f} seconds on this machine') 
print(f'Method 1: I1 = {method1_result:.4f}')
print(f'Method 2: I1 = {method2_result:.4f}')
print(f'Percentage error: {perc_error:.4f}%')

#################################################################
##### Problem 2: Substitution method and infinite integrals #####
#################################################################

# I am going to overwrite a few different variables, such as the upper and lower bound
# Creating new variables would be confusing and unnecessary. They do the same things as before
UPPER_BOUND = 100_000
LOWER_BOUND = 1
h = 0.001

def integrand_problem_two(x: float) -> float:
    # This is the integrand for problem 2, which involves a square root in the denominator
    # This python function just mimics that mathematical function from the prompt

    denominator = x*np.sqrt(x-1)
    return 1/denominator

method1_result = np.pi # I solved this integral using u-subtitution and have confirmed the result is pi
t2 = time.time()
method2_result = midpoint_integration(integrand_problem_two) # integrand_problem_two has no other arguments besides x, so I list no extra arguments here
t3 = time.time()

# Print the results for problem two to the terminal
print('\nPROBLEM TWO')

# This was 41.34 seconds on my machine. It might take you longer, depending on your hardware
# I really did not want to increase the accuracy any more, because I'm assuming that more than about a minute is too much
print(f'The custom function to evaluate this integral took {t3-t2:.2f} seconds on this machine') 
print(f'{method1_result:.8f} # I2 Method 1')
print(f'{method2_result:.8f} # I2 Method 2')
print(f'{(method1_result-np.pi):.10f} # Difference Method 1')
print(f'{(method2_result-np.pi):.10f} # Difference Method 2')