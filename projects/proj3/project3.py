# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 19 March 2026

# Project 3

# I did not use AI at all to complete this assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, leastsq
import sys
import os

#############################
##### Functions Block 1 #####
#############################

def compute_diode_current(Vd: np.array, n: float, T: float, Is: float):
    """
    Calculate the diode current given some diode voltages and several parameters.
    
    Parameters
    ----------
    Vd : np.array
        An array containing all of the diode voltages for which we want a current.
    n : float or int
        The diode ideality factor, a number such as 1.7.
    T : float or int
        The temperature of the diode.
    Is : float or int
        The saturation current of the diode.

    Returns
    -------
    diode_current : np.array
        The amount of current flowing through this diode under these conditions.
    """

    Vt = n*K*T/Q
    return Is * (np.exp(Vd/Vt)-1)

def solve_diode(Vd: float, Vs: float, R: float, n: float, T: float, Is: float):
    """
    Find the difference between diode and resistor current for a specific combination of voltages and parameters.
    
    Parameters
    ----------
    Vd : float or int
        A single value for the diode voltage.
    Vs : float or int
        The source voltage for this iteration of the circuit.
    R : float or int
        The value of resistance placed in series with the diode.
    n : float or int
        The diode ideality factor, a number such as 1.7.
    T : float or int
        The temperature of the diode.
    Is : float or int
        The saturation current of the diode.

    Returns
    -------
    error : float or int
        The difference between the diode and resistor currents. This will be zero when the solution is found.
    """

    # First, find the current through the resistor.
    i_resistor = (Vs - Vd)/R

    # Now, find the current through the diode.
    i_diode = compute_diode_current(Vd, n, T, Is)

    # Return the difference between the two numbers.
    return i_diode - i_resistor

def calc_diode_iv(source_voltages: float, R: float, n: float, T: float, Is: float):
    """
    Find the diode voltages and currents given the source voltages and a few other parameters.

    Parameters
    ----------
    source_voltages : np.array
        An array containing all of the source voltages we want to test.
    R : float or int
        The value of resistance placed in series with the diode.
    n : float or int
        The diode ideality factor, a number such as 1.7.
    T : float or int
        The temperature of the diode.
    Is : float or int
        The saturation current of the diode.

    Returns
    -------
    diode_voltages : list
        A list containing all of the diode voltages. Each one maps to a source voltage in the "source_voltages" array.
    diode_currents : list
        A list containing all of the diode currents. Each one maps to a source voltage in the "source_voltages" array.
    """

    # Initialize some variables before we enter a loop.
    prev_sol = 0.60
    diode_voltages = list()
    diode_currents = list()

    # Loop over every source voltage, finding the diode voltage and current for each one.
    for Vs in source_voltages:

        # Solve for the diode voltage at this source voltage.
        Vd = fsolve(
            func=solve_diode,
            x0=prev_sol,
            args=(Vs, R, n, T, Is)
        )

        # Solve for the diode current using Ohm's law.
        Id = (Vs - Vd) / R

        # Save the diode voltage and current to the respective lists.
        diode_voltages.append(Vd)
        diode_currents.append(Id)

        # The next iteration should start its fsolve journey using this diode voltage.
        prev_sol = Vd
    
    # Return the diode voltages and currents.
    return diode_voltages, diode_currents

##############################
##### Complete Problem 1 #####
##############################

# Constants for problem 1.
K = 1.380648e-23
Q = 1.6021766208e-19
is_val = 1e-9
n_val = 1.7
res_val = 11e3
temp = 350

# Solve for the diode voltages and currents in problem 1.
source_voltages = np.arange(0.1, 2.6, step=0.1)
diode_voltages, diode_currents = calc_diode_iv(source_voltages, res_val, n_val, temp, is_val)

# Create the graph for problem 1.
plt.semilogy(source_voltages, diode_currents, label='Idiode vs Source Voltage', color='red')
plt.semilogy(diode_voltages, diode_currents, label='Idiode vs Diode Voltages', color='black')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('Diode Current vs Source Voltage (#1)')
plt.legend()
plt.grid(True)
plt.show()

#############################
##### Functions Block 2 #####
#############################

def read_data(file_path: str):
    """
    Read the data stored in the provided txt file and return two variables: the source voltages and the real currents.
    
    Parameters
    ----------
    file_path : str
        The path to the txt file containing source voltages and diode currents.

    Returns
    -------
    source_voltages : np.array
        An array containing every source voltage in the provided document. Its format should match the example provided.
    real_currents : np.array
        An array containing the measured currents for each source voltage in the other array.
    """

    # Read in the txt file and store its values in two variables for source voltage and measured current.
    col_names = ['Vsource', 'Idiode']
    df = pd.read_csv(file_path, names=col_names, header=None, delimiter=' ')
    source_voltages = df['Vsource'].to_numpy()
    real_currents = df['Idiode'].to_numpy()

    # Return the source voltages and measured currents.
    return source_voltages, real_currents

def norm_error(real: np.array, calc: np.array):
    """
    Calculate the normalized error for two numpy arrays.

    Parameters
    ----------
    real : np.array
        The currents measured in the lab.
    calc : np.array
        The calculated currents in this iteration of the program.
    
    Returns
    -------
    abs_err : np.array
        The normalized error for this combination of input arrays.
    """

    abs_err = (real - calc) / (real + calc + 1e-15)
    return abs_err

def abs_error(real: np.array, calc: np.array):
    """
    Calculate the absolute error for two numpy arrays.

    Parameters
    ----------
    real : np.array
        The currents measured in the lab.
    calc : np.array
        The calculated currents in this iteration of the program.
    
    Returns
    -------
    abs_err : np.array
        The absolute error for this combination of input arrays.
    """

    norm_err = (real - calc)
    return norm_err

def opt_r(r_value,
          ide_value,phi_value,area,temp,source_voltages,real_currents):
    """
    Optimize for the resistance value.

    Parameters
    ----------
    r_value : float or int
        The current prediction for the value of resistance in series with the diode.
    ide_value : float or int
        The current prediction for the ideality value of the diode in this circuit.
    phi_value : float or int
        The current prediction for the phi value of the diode in this circuit.
    area : float or int
        This is a constant defined in the problem statement, corresponding to the area of the diode.
    temp : float or int
        The temperature of the diode. This will affect the current through the device.
    source_voltages : np.array
        An array containing every source voltage in the provided document. Its format should match the example provided.
    real_currents : np.array
        An array containing the measured currents for each source voltage in the other array.
    
    Returns
    -------
    residuals : np.array
        An array containing the residuals for this combination of inputs. When the solution is found, this will approach zero.

    """

    ####################################
    ##### This section is repeated #####
    ####################################

    est_v = np.zeros_like(source_voltages)               # an array to hold the diode voltages
    calc_currents = np.zeros_like(source_voltages)       # an array to hold the diode currents
    prev_v = 0.6                                         # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )

    for index in range(len(source_voltages)):
        prev_v = fsolve(solve_diode,prev_v,
				(source_voltages[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    calc_currents = compute_diode_current(est_v,ide_value,temp,is_value)

    ################################
    ##### End repeated section #####
    ################################

    # Calculate the residuals and return them.
    return abs_error(real_currents, calc_currents)

def opt_n(ide_value,
          r_value,phi_value,area,temp,source_voltages,real_currents):
    """
    Optimize for the ideality value.
    
    Parameters
    ----------
    ide_value : float or int
        The current prediction for the ideality value of the diode in this circuit.
    r_value : float or int
        The current prediction for the value of resistance in series with the diode.
    phi_value : float or int
        The current prediction for the phi value of the diode in this circuit.
    area : float or int
        This is a constant defined in the problem statement, corresponding to the area of the diode.
    temp : float or int
        The temperature of the diode. This will affect the current through the device.
    source_voltages : np.array
        An array containing every source voltage in the provided document. Its format should match the example provided.
    real_currents : np.array
        An array containing the measured currents for each source voltage in the other array.
    
    Returns
    -------
    residuals : np.array
        An array containing the residuals for this combination of inputs. When the solution is found, this will approach zero.

    """

    ####################################
    ##### This section is repeated #####
    ####################################

    est_v = np.zeros_like(source_voltages)               # an array to hold the diode voltages
    calc_currents = np.zeros_like(source_voltages)       # an array to hold the diode currents
    prev_v = 0.6                                         # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )

    for index in range(len(source_voltages)):
        prev_v = fsolve(solve_diode,prev_v,
				(source_voltages[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    calc_currents = compute_diode_current(est_v,ide_value,temp,is_value)

    ################################
    ##### End repeated section #####
    ################################

    # Calculate the residuals and return them.
    return norm_error(real_currents, calc_currents)

def opt_phi(phi_value,
          r_value,ide_value,area,temp,source_voltages,real_currents):
    """
    Optimize for the phi value.
    
    Parameters
    ----------
    phi_value : float or int
        The current prediction for the phi value of the diode in this circuit.
    r_value : float or int
        The current prediction for the value of resistance in series with the diode.
    ide_value : float or int
        The current prediction for the ideality value of the diode in this circuit.
    area : float or int
        This is a constant defined in the problem statement, corresponding to the area of the diode.
    temp : float or int
        The temperature of the diode. This will affect the current through the device.
    source_voltages : np.array
        An array containing every source voltage in the provided document. Its format should match the example provided.
    real_currents : np.array
        An array containing the measured currents for each source voltage in the other array.
    
    Returns
    -------
    residuals : np.array
        An array containing the residuals for this combination of inputs. When the solution is found, this will approach zero.

    """

    ####################################
    ##### This section is repeated #####
    ####################################

    est_v = np.zeros_like(source_voltages)               # an array to hold the diode voltages
    calc_currents = np.zeros_like(source_voltages)       # an array to hold the diode currents
    prev_v = 0.6                                         # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )

    for index in range(len(source_voltages)):
        prev_v = fsolve(solve_diode,prev_v,
				(source_voltages[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    calc_currents = compute_diode_current(est_v,ide_value,temp,is_value)

    ################################
    ##### End repeated section #####
    ################################

    # Calculate the residuals and return them.
    return norm_error(real_currents, calc_currents)

##############################
##### Complete Problem 2 #####
##############################

# Read in the source voltages and measured currents for problem 2.
file_path = '/home/steven-wallace/Documents/asu/eee-419/projects/proj3/DiodeIV.txt'
source_voltages, real_currents = read_data(file_path)

# Define some constants used in problem 2.
area = 1e-8
temp = 375
TOL = 1e-9
MAX_ITER = 2_000

# Variables that change with each iteration for problem 2.
phi_val = 0.8
n_val = 1.5
res_val = 10e3
error = 100
iter_num = 0

# Begin a loop to optimize for resistance, ideality, and phi.
while (error > TOL) and (iter_num < MAX_ITER):

    # Optimize for the resistance, then ideality, and finally the value of phi.
    res_arr = leastsq(func=opt_r, x0=res_val, 
                      args=(n_val, phi_val, area, temp, source_voltages, real_currents))
    n_arr = leastsq(func=opt_n, x0=n_val, 
                      args=(res_val, phi_val, area, temp, source_voltages, real_currents))
    phi_arr = leastsq(func=opt_phi, x0=phi_val, 
                      args=(res_val, n_val, area, temp, source_voltages, real_currents))
    
    # Extract the new values for R, n, and phi from these arrays returned by the optimize function.
    res_val = res_arr[0][0]
    n_val = n_arr[0][0]
    phi_val = phi_arr[0][0]

    # Calculate the error for this iteration.
    res = opt_phi(phi_val, res_val, n_val, area, temp, source_voltages, real_currents) 
    error = np.sum(np.abs(res))/len(res)

    # Add one to the iteration counter, and print the current status to the terminal.
    iter_num += 1
    print(f'Iteration {iter_num} Values:\n\tresistor: {res_val}\n\tideality: {n_val}\n\tphi value: {phi_val}\n\terror: {error}')

# Calculate the diode currents matching these phi, R, and n values.
is_value = area * temp * temp * np.exp(-phi_val * Q / ( K * temp ) )
_, calc_currents = calc_diode_iv(source_voltages, res_val, n_val, temp, is_value)

# Create the graph for problem 2.
plt.figure()
plt.semilogy(source_voltages, real_currents, label='Measured Current (A)', color='black', marker='o')
plt.semilogy(source_voltages, calc_currents, label='Model Current (A)', color='red')
plt.xlabel('Source Voltage (V)')
plt.ylabel('Diode Current (A)')
plt.title('Diode Current vs Source Voltage (#2)')
plt.legend()
plt.grid(True)
plt.show()