# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Homework 1
# I did not use AI at all to complete this assignment

# I used this reference to understand the procedure behind the midpoint method of definite integrals
# https://math.libretexts.org/Courses/Mount_Royal_University/Calculus_for_Scientists_II/2%3A_Techniques_of_Integration/2.5%3A_Numerical_Integration_-_Midpoint%2C_Trapezoid%2C_Simpson%27s_rule
# I used this reference when writing my midpoint function because I needed to make it accept an arbitrary number of arguments
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

#####################
##### Functions #####
#####################

def find_perc_error(solution: float, estimate: float) -> float:
    """The purpose of this function is to find the percent error for a given esimate.\n
       It takes two inputs, which are the analytical solution and the estimate in question.\n
       Naturally, the only output is the percentage error."""

    error = abs((solution-estimate)/solution)*100
    return error

def cubic_function(x: float, a: float, b: float, c: float) -> float:
    """The purpose of this function is to represent the cubic function as a function of x and the coefficients a, b, c.\n
       Its only output is the y-value associated with that combination of input values.\n
       Of course, the function is a*x**3 + b*x**2 + c*x"""
    
    return (a*x**3) + (b*x**2) + (c*x)

def find_cubic_solution(a: float, b: float, c: float, LOWER_BOUND: float, UPPER_BOUND: float) -> float:
    """The purpose of this function is to find the analytical solution for this integral, when the upper and lower bounds are known.\n
       Naturally, it takes three inputs: a, b, and c. Its only output is the analytical solution for those coefficients and bounds.\n
       """
    
    upper_term = (a/4)*UPPER_BOUND**4 + (b/3)*UPPER_BOUND**3 + (c/2)*UPPER_BOUND**2
    lower_term = (a/4)*LOWER_BOUND**4 + (b/3)*LOWER_BOUND**3 + (c/2)*LOWER_BOUND**2
    
    return upper_term - lower_term

def midpoint_integration(integrand, LOWER_BOUND: float, UPPER_BOUND: float, h: float, *args) -> float:
    """The purpose of this function is to evaluate an integral when given an integrand, its required arguments, and bounds.\n
       It uses the midpoint method, which just means that we create a ton of rectangles between the bounds and add their areas.\n
       The function will return the value of this integral over the defined interval."""

    running_sum = 0 # The running sum will store the area accumulated so far
    midpoints = np.arange((h/2)+LOWER_BOUND, UPPER_BOUND+(h/2), h)
    # We want the midpoints to run from h/2 above the lower bound until h/2 beneath the upper bound
    # (Note that I used h/2 over the upper bound because np.arange stops one short)
    # They should also increment by h, and this array will have the same number of entries as "n"

    for midpoint in midpoints:

        rect_width = h # The width of the rectangle should always be h
        rect_height = integrand(midpoint, *args) # The height of the rectangle will be the cubic function evaluated at that midpoint
        running_sum += rect_width * rect_height # Now I'm just adding the area of that new rectangle to whatever we had before

    return running_sum # Now that we're done with that for-loop, the running sum is our definite integral

###########################
##### Solve Problem 1 #####
###########################

# First, I will find the solution to the integral of this particular cubic function using my derived expression
# Then, I will integrate using the quad function from scipy.integrate, and find its associated error
# Finally, I will integrate using my custom midpoint integration function and find its error as well

cubic_solution = find_cubic_solution(a, b, c, LOWER_BOUND, UPPER_BOUND)

method1_result, _ = integrate.quad(cubic_function, LOWER_BOUND, UPPER_BOUND, args=(a, b, c)) 
perc_error_m1 = find_perc_error(cubic_solution, method1_result)

method2_result = midpoint_integration(cubic_function, LOWER_BOUND, UPPER_BOUND, h, a, b, c) 
perc_error_m2 = find_perc_error(cubic_solution, method2_result)

# Print the results for Problem 1 to the terminal
print('\nPROBLEM ONE')
print(f'Solution = {cubic_solution:.2f}')
print(f'Method 1 (scipy.integrate.quad): I1 = {method1_result:.2f}')
print(f'Method 1 Percent Error: {perc_error_m1:.4f} %')
print(f'Method 2 (custom midpoint): I1 = {method2_result:.2f}')
print(f'Method 2 Percent Error: {perc_error_m2:.4f} %')


#################################################################
##### Problem 2: Substitution method and infinite integrals #####
#################################################################

# The u-substitution that I used to verify the solution to this integral is indeed pi
# https://drive.google.com/file/d/1J4fB9gmG9iC_ujHpxxWu7HsU9jIadJto/view?usp=drive_link
# A reference I used for integral tables while performing this derivation
# https://openstax.org/books/calculus-volume-1/pages/a-table-of-integrals

# I am going to overwrite a few variables for this problem, namely h and the upper and lower bounds
UPPER_BOUND = 100_000
LOWER_BOUND = 1
h = 0.001

def integrand_inverse_sqrt(x: float) -> float:
    """The purpose of this function is to represent the integrand in problem two.\n
       This python functions accepts only x and outputs the y-value."""

    denominator = x*np.sqrt(x-1)
    return 1/denominator

# Now I am going to find the result of this integral using methods 1 and 2
# Refer to that google drive link for how I proved this integral evaluates to pi
# I use the scipy.integrate.quad function to evaluate the integral in method 2
method1_result = np.pi
method2_result, _ = integrate.quad(integrand_inverse_sqrt, LOWER_BOUND, UPPER_BOUND)

# Print the results for problem two to the terminal
print('\nPROBLEM TWO')
print(f'{method1_result:.8f} # I2 Method 1')
print(f'{method2_result:.8f} # I2 Method 2')
print(f'{(method1_result-np.pi):.10f} # Difference Method 1')
print(f'{(method2_result-np.pi):.10f} # Difference Method 2')