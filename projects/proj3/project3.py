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

def diode(vdiode: float, sat_curr: float, ideality: float, temp: float):
    vtherm = ideality*k*temp/q
    return sat_curr * (np.exp(vdiode/vtherm)-1)


#####################
##### Problem 1 #####
#####################

def fx_prob1(idiode: float, vsource: float):
    return diode(vdiode=(vsource-idiode*R), sat_curr=1e-9, ideality=1.7, temp=350) - idiode

R = 11e3
diode_currents = list()
source_voltages = np.arange(0.1, 2.6, step=0.1)

for vsource in source_voltages:
    curr_est = 1e-3

    root = fsolve(
        func=fx_prob1,
        x0=curr_est,
        args=vsource
    )

    diode_currents.append(root)


plt.scatter(source_voltages, diode_currents)
plt.show()
