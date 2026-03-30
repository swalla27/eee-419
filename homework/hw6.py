# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 30 March 2026

# Homework 6

# I did not use AI at all to complete this assignment.

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import sys

#####################
##### Problem 1 #####
#####################

def dydt_prob1(y, t):
    """
    A function describing the differential equation in problem 1, which is y' = cos(t)

    Parameters
    ----------
    y : np.array
        The variable called y in this equation. Its physical meaning is not mentioned.
    t : np.array
        An array of time values, which will be the same for all problems in this homework.

    Returns
    -------
    dydt : np.array
        The derivative of y with respect to time.
    """

    return np.cos(t)

# Define the time vector, which is a constant because it does not change for the entire program.
TIME_VEC = np.linspace(0, 7, 700)

# Use odeint to solve the differential equation for the first problem. The initial condition is y(0) = 1
yvec = odeint(func=dydt_prob1, y0=1, t=TIME_VEC)

# Plot the solution to the first problem and label the graph accordingly.
plt.plot(TIME_VEC, yvec)
plt.xlabel('Time (t)')
plt.ylabel('y')
plt.title('Solution to Problem #1')
plt.grid(True)
plt.show()

#####################
##### Problem 2 #####
#####################

def dydt_prob2(y, t):
    """
    A function describing the differential equation in problem 2, which is y' = -y + (t^2)(e^-2t) + 10

    Parameters
    ----------
    y : np.array
        The variable called y in this equation. Its physical meaning is not mentioned.
    t : np.array
        An array of time values, which will be the same for all problems in this homework.

    Returns
    -------
    dydt : np.array
        The derivative of y with respect to time.
    """

    return -y + t**2*np.exp(-2*t) + 10

# Solve the differential equation mentioned in problem 2, whose initial condition is y(0) = 0
# The time vector does not need to be changed, and remains constant throughout all problems.
yvec = odeint(func=dydt_prob2, y0=0, t=TIME_VEC)

# Plot the solution to the second problem and label the graph accordingly.
plt.plot(TIME_VEC, yvec)
plt.xlabel('Time (t)')
plt.ylabel('y')
plt.title('Solution to Problem #2')
plt.grid(True)
plt.show()

#####################
##### Problem 3 #####
#####################

def system_prob3(r, t):
    """
    A function describing the differential equation mentioned in problem 3, which is y" = -4y' - 4y + 25cos(t) + 25sin(t).
    I solved this problem by splitting it into a system of first order differential equations. The details of this are shown below.

    Parameters
    ----------
    r : np.array
        An array containing the values for w and y, where w is equal to dydt. The 0th entry is w and the 1st is y.
    t : np.array
        An array of time values, which will be the same for all problems in this homework.

    Returns
    -------
    drdt : np.array
        An array with the rate of change in r with respect to time. The 0th entry is dwdt, and the 1st is dydt.
    """

    # Extract w and y from the input vector called r. The 0th entry is w, and the 1st is y.
    w = r[0]
    y = r[1]

    # Begin with y" = -4y' - 4y + 25cos(t) + 25sin(t)
    # Allow w = y'
    # Therefore, w' = -4w - 4y + 25cos(t) + 25sin(t)
    # and y' = w

    # Define the rate of change in w with respect to time.
    dwdt = -4*w - 4*y + 25*np.cos(t) + 25*np.sin(t)

    # Define the rate of change in y with respect to time, which is how we defined w.
    dydt = w

    # The vector called r has a derivative as well, and its 0th entry will be dwdt, 1st dydt.
    # This is what we will return from this function, because that is the format odeint wants.
    drdt = np.array([dwdt, dydt], dtype=float)

    return drdt

# Solve the second order differential equation specified in problem 3 by decomposing it into a system of first order equations.
# odeint will return the vector "r" in this situation, but I needed to transpose it before extracting dydt and y.
# dydt is the 0th entry because I defined w as the first component of r. 
# I know for certain that dwdt is not returned because previous problems returned y, not y'
# Of course, the initial conditions are y(0) = 1 and y'(0) = 1
dydt, y = odeint(func=system_prob3, y0=[1, 1], t=TIME_VEC).T

# Plot both y and dydt to solve the third problem, then label the graph accordingly.
plt.plot(TIME_VEC, y, label='y')
plt.plot(TIME_VEC, dydt, label='dydt')
plt.xlabel('Time (t)')
plt.ylabel('y')
plt.title('Solution to Problem #3')
plt.legend()
plt.grid(True)
plt.show()

# Notice how dydt passes through 0 near t=1.75, at which point y is at a maximum. This is consistent with the story I have outlined above.
# I also plotted dwdt in an earlier version, which helped me to confirm that I have labeled the traces correctly.