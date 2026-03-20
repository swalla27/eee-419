# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 19 March 2026

# Project 3

# I did not use AI at all to complete this assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys
import os

# Boltzman's constant and the unit charge.
k = 1.380648e-23
q = 1.6021766208e-19
SAT_CURR = 1e-9
IDEALITY = 1.7
RES = 11e3
TEMP = 350

#####################
##### Problem 1 #####
#####################

def diode(vdiode: float):
    vtherm = IDEALITY*k*TEMP/q
    return SAT_CURR * (np.exp(vdiode/vtherm)-1)


def prob1_fx(vdiode: float, vsource: float):
    vtherm = IDEALITY*k*TEMP/q

    return SAT_CURR*(np.exp(vdiode/vtherm)-1) - (vsource-vdiode)/RES


diode_currents = list()
diode_voltages = list()
source_voltages = np.arange(0.1, 2.6, step=0.1)

for vsource in source_voltages:

    root = fsolve(
        func=prob1_fx,
        x0=0.60,
        args=vsource
    )

    diode_voltages.append(root)
    diode_currents.append(diode(root))

plt.semilogy(source_voltages, diode_currents, label='Idiode vs Source Voltage', color='red')
plt.semilogy(diode_voltages, diode_currents, label='Idiode vs Diode Voltages', color='black')
plt.xlabel('Voltage (V)')
plt.ylabel('Current (A)')
plt.title('First Problem IV Curve')
plt.grid(True)

plt.show()

