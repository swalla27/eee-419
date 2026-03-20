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

#####################
##### Problem 1 #####
#####################

k = 1.380648e-23
q = 1.6021766208e-19
Is = 1e-9
n = 1.7
R = 11e3
T = 350

def idiode_from_vdiode(Vd: float):
    Vt = n*k*T/q
    return Is * (np.exp(Vd/Vt)-1)

def solve_for_vdiode(Vd: float, Vsrc: float, n: float, R: float, Is: float):
    Vt = n*k*T/q
    return Is*(np.exp(Vd/Vt)-1) - (Vsrc-Vd)/R

diode_currents = list()
diode_voltages = list()
source_voltages = np.arange(0.1, 2.6, step=0.1)

for Vsrc in source_voltages:

    root = fsolve(
        func=solve_for_vdiode,
        x0=0.60,
        args=(Vsrc, n, R, Is)
    )

    diode_voltages.append(root)
    diode_currents.append(idiode_from_vdiode(root))


if False:
    plt.semilogy(source_voltages, diode_currents, label='Idiode vs Source Voltage', color='red')
    plt.semilogy(diode_voltages, diode_currents, label='Idiode vs Diode Voltages', color='black')
    plt.xlabel('Voltage (V)')
    plt.ylabel('Current (A)')
    plt.title('First Problem IV Curve')
    plt.legend()
    plt.grid(True)

    plt.show()

#####################
##### Problem 2 #####
#####################



col_names = ['Vsource', 'Idiode']
df = pd.read_csv('/home/steven-wallace/Documents/asu/eee-419/projects/proj3/DiodeIV.txt', names=col_names, header=None, delimiter=' ')
source_voltages = df['Vsource']
real_currents = df['Idiode']

A = 1e-8
T = 375

phi = 0.8
n = 1.5
R = 10e3

def find_error(phi: float, n: float, R: float):

    def find_Is():
        Is = A * T**2 * np.exp(-phi*q/(k*T))
        return Is
    Is = find_Is()

    sim_currents = list()
    for Vsrc in source_voltages:
        root = fsolve(
            func=solve_for_vdiode,
            x0=0.60,
            args=(Vsrc, n, R, Is)
        )
        sim_currents.append(idiode_from_vdiode(root))
    
    error = 0
    for idx, val in enumerate(real_currents):
        error += (val - sim_currents[idx])
    return error


def residual_phi(phi, n, R):
    return find_error(phi, n, R)

def residual_n(n, R, phi):
    return find_error(phi, n, R)

def residual_R(R, phi, n):
    return find_error(phi, n, R)


tolerance = 1e-6
max_iters = 20_000

error = 100
iter_num = 0

while (error > tolerance) and (iter_num < max_iters):

    phi_array = leastsq(func=residual_phi, x0=phi, args=(n, R))
    phi = phi_array[0][0]

    n_array = leastsq(func=residual_n, x0=n, args=(R, phi))
    n = n_array[0][0]

    R_array = leastsq(func=residual_R, x0=R, args=(phi, n))
    R = R_array[0][0]

    error = find_error(phi, n, R)
    iter_num += 1

    print(error)

    print(f'Iteration Num: {iter_num}\n\tphi = {phi:.3f}\n\tn = {n:.3f}\n\tR = {R:.3f}\n\tError = {error:.3f}')


sim_currents = list()
for Vsrc in source_voltages:
    root = fsolve(
        func=solve_for_vdiode,
        x0=0.60,
        args=(Vsrc, n, R, Is)
    )
    sim_currents.append(idiode_from_vdiode(root))

plt.figure()
plt.semilogy(source_voltages, real_currents, label='Real Current (A)', color='red')
plt.semilogy(source_voltages, sim_currents, label='Simulated Current (A)', color='black')
plt.xlabel('Source Voltage (V)')
plt.ylabel('Diode Current (A)')
plt.title('Source Voltage vs Diode Current')
plt.legend()
plt.grid(True)
plt.show()