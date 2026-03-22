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

def compute_diode_current(Vd: float, n: float, T: float, Is: float):
    """Calculate the diode current given a single diode voltage and several parameters."""

    Vt = n*K*T/Q
    return Is * (np.exp(Vd/Vt)-1)

def solve_diode(Vd: float, Vs: float, R: float, n: float, T: float, Is: float):
    """Find the difference between diode and resistor current for a specific combination of voltages and parameters."""

    i_resistor = (Vs - Vd)/R
    i_diode = compute_diode_current(Vd, n, T, Is)
    return i_diode - i_resistor

def calc_diode_iv(source_voltages: float, R: float, n: float, Is: float, T: float):
    """Find the diode voltages and currents given the source voltages and a few other parameters."""

    prev_sol = 0.60
    diode_voltages = list()
    diode_currents = list()

    for Vs in source_voltages:

        Vd = fsolve(
            func=solve_diode,
            x0=prev_sol,
            args=(Vs, R, n, T, Is)
        )

        Id = (Vs - Vd) / R

        diode_voltages.append(Vd)
        diode_currents.append(Id)

        prev_sol = Vd
    
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
# source_voltages = np.arange(0.1, 2.6, step=0.1)
# diode_voltages, diode_currents = calc_diode_iv(source_voltages, res_val, n_val, is_val, temp)

# Create the graph for problem 1.
# plt.semilogy(source_voltages, diode_currents, label='Idiode vs Source Voltage', color='red')
# plt.semilogy(diode_voltages, diode_currents, label='Idiode vs Diode Voltages', color='black')
# plt.xlabel('Voltage (V)')
# plt.ylabel('Current (A)')
# plt.title('First Problem IV Curve')
# plt.legend()
# plt.grid(True)
# plt.show()

#############################
##### Functions Block 2 #####
#############################

def read_data():
    """Read the data stored in the provided txt file and return two variables: the source voltages and the real currents."""

    col_names = ['Vsource', 'Idiode']
    df = pd.read_csv('/home/steven-wallace/Documents/asu/eee-419/projects/proj3/DiodeIV.txt', names=col_names, header=None, delimiter=' ')
    source_voltages = df['Vsource'].to_numpy()
    real_currents = df['Idiode'].to_numpy()

    return source_voltages, real_currents

def opt_r(r_value,
          ide_value,phi_value,area,temp,src_v,real_currents):
    """Optimize for the resistance value."""

    est_v = np.zeros_like(src_v)       # an array to hold the diode voltages
    calc_currents = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.6                # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(solve_diode,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    calc_currents = compute_diode_current(est_v,ide_value,temp,is_value)

    residual = real_currents - calc_currents
    
    return residual

def opt_n(ide_value,
          r_value,phi_value,area,temp,src_v,real_currents):
    """Optimize for the ideality value."""

    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    calc_currents = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.6                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(solve_diode,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    calc_currents = compute_diode_current(est_v,ide_value,temp,is_value)

    residual = (real_currents - calc_currents) / (real_currents + calc_currents + 1e-15)

    return residual

def opt_phi(phi_value,
          r_value,ide_value,area,temp,src_v,real_currents):
    """Optimize for the phi value."""

    est_v   = np.zeros_like(src_v)       # an array to hold the diode voltages
    calc_currents = np.zeros_like(src_v)       # an array to hold the diode currents
    prev_v = 0.6                 # an initial guess for the voltage

    # need to compute the reverse bias saturation current for this phi!
    is_value = area * temp * temp * np.exp(-phi_value * Q / ( K * temp ) )

    for index in range(len(src_v)):
        prev_v = fsolve(solve_diode,prev_v,
				(src_v[index],r_value,ide_value,temp,is_value),
                                xtol=1e-12)[0]
        est_v[index] = prev_v            # store for error analysis

    # compute the diode current
    calc_currents = compute_diode_current(est_v,ide_value,temp,is_value)
    
    residual = (real_currents - calc_currents) / (real_currents + calc_currents + 1e-15)

    return residual

##############################
##### Complete Problem 2 #####
##############################

# Read in the source voltages and measured currents for problem 2.
source_voltages, real_currents = read_data()

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

    iter_num += 1

    print(f'Iteration {iter_num} Values:\n\tresistor: {res_val}\n\tideality: {n_val}\n\tphi value: {phi_val}\n\terror: {error}')

# Calculate the diode currents matching these phi, R, and n values.
is_value = area * temp * temp * np.exp(-phi_val * Q / ( K * temp ) )
_, calc_currents = calc_diode_iv(source_voltages, res_val, n_val, is_value, temp)

# Create the graph for problem 2.
plt.figure()
plt.semilogy(source_voltages, real_currents, label='Real Current (A)', color='black', marker='o')
plt.semilogy(source_voltages, calc_currents, label='Calculated Current (A)', color='red')
plt.xlabel('Source Voltage (V)')
plt.ylabel('Diode Current (A)')
plt.title('Source Voltage vs Diode Current')
plt.legend()
plt.grid(True)
plt.show()