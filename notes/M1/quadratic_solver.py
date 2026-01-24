# Problem 1:
# Quadratic equation solver + identify whether the roots are complex, real, equal
import numpy as np
import cmath
import sys

def quadratic_solver(a: float, b: float, c: float):
    discriminant = b**2 - (4*a*c)
    if discriminant > 0:
        solution1 = (-b + np.sqrt(discriminant))/(2*a)
        solution2 = (-b - np.sqrt(discriminant))/(2*a)
        print(f'Real Roots: {solution1:.2f} and {solution2:.2f}')
    elif discriminant == 0:
        solution = (-b)/(2*a)
        print(f'One root: {solution}')
    else:
        solution1 = cmath.sqrt(discriminant)
        solution2 = np.conjugate(solution1)
        print(f'Complex roots: {solution1:.2f} and {solution2:.2f}')

quadratic_solver(1, 4, 5)