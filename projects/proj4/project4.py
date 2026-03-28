# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 26 March 2026

# Project 4

# I did not use AI at all to complete this assignment

import numpy as np
import subprocess
import time
import sys
import os

#####################
##### Functions #####
#####################

def read_original_infile():
    """
    Read the input file and return the netlist until a certain point. We stop at the word 'fan' to cut off the last few lines.
    
    Parameters
    ----------
    None

    Returns 
    -------
    ORIGINAL_NETLIST : list
        The original netlist, stored as a list. I will append to this to create every other netlist in the program.
    """

    # Open the input file and store the netlist in a variable called "ORIGINAL_NETLIST".
    ORIGINAL_NETLIST = list()
    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:

            # Stop reading once we see the word "fan". This will cut off the last few lines.
            if "fan" in line:
                break

            ORIGINAL_NETLIST.append(line)

    return ORIGINAL_NETLIST

def make_new_netlist(n_value: int, fan_value: int):
    """
    Create a new netlist using the original one, an n-value, and a fan value.
    
    Parameters
    ----------
    n_value : int
        The number of inverters in this circuit. The program will loop over many values for this.
    fan_value : int
        The ratio between subsequent inverters, in that inverter B will be 'fan' times larger than A, if B follows A. 

    Returns
    -------
    None
    """

    # Create a copy of the original netlist. This is the starting point for our netlist in this iteration of the program.
    current_netlist = ORIGINAL_NETLIST.copy()

    # Add a line to the netlist specifying what the fan value should be.
    current_netlist.append(f'.param fan = {fan_value}\n')
    
    # The first inverter will have its output node set to z if there is only one inverter in the whole circuit, and b if not.
    if n_value == 1:
        current_netlist.append('Xinv1 a z inv M=1\n')
    else:
        current_netlist.append('Xinv1 a b inv M=1\n')

    # Initialize the previous and next node variables so that we can enter this loop.
    prev_node = 'b'
    next_node = 'b'

    # I will loop over inverter id, which is the integer identifying each inverter (after the first).
    # We want to stop at n+1 because there will be exactly n inverters in the circuit.
    for inverter_id in np.arange(start=2, stop=n_value+1, step=1):

        # If this inverter is the final one, then its output node should be z. Otherwise, we should increment the letter by one.
        if inverter_id == n_value:
            next_node = 'z'
        else:
            # This will increment the letter by one. For example, a becomes b and c becomes d.
            next_node = chr(ord(prev_node) + 1)

        # Add a line to the current netlist for this particular inverter id, complete with its nodes and strength.
        current_netlist.append(f'Xinv{inverter_id} {prev_node} {next_node} inv M=fan**{inverter_id-1}\n')

        # The next iteration should start with its input node equal to the output of this inverter.
        prev_node = next_node

    # Now that we are done adding lines to the netlist, we place the end statement.
    current_netlist.append('.end\n')

    # This will write the netlist we just created back to the original file. 
    # There is no need to return anything because the next function will access this file directly.
    with open(INPUT_FILE, 'w') as f:
        for line in current_netlist:
            f.write(line)

def get_tphl_from_hspice():
    """
    Run the hspice script and return the high to low propagation delay for this netlist.

    Parameters
    ----------
    None

    Returns
    -------
    tphl : float
        The high to low propagation delay hspice has calculated for this circuit.
    """

    # Create a process object which runs hspice on the input file. This file contains the netlist.
    proc = subprocess.Popen(["hspice", INPUT_FILE],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    
    # Extract the output and error messages with this process. These variables are not currently printed to the terminal.
    output, err = proc.communicate()
    # print(f'output: {output.decode()}')
    # print(f'err: {err.decode()}')

    # Extract the high to low propagation delay from the output file, then return it.
    # There was something here earlier, can't remember what, but it was deprecated and so I had to change it.
    data = np.loadtxt(OUTPUT_FILE, delimiter=',', skiprows=3, dtype=str)
    tphl = float(data[1][0])

    return tphl

# Define the working folder and the files we are referencing throughout this program. 
# You can easily change these constants to reference different files.
WORKING_FOLDER = os.getcwd()
INPUT_FILE = os.path.join(WORKING_FOLDER, 'InvChain.sp')
OUTPUT_FILE = os.path.join(WORKING_FOLDER, 'InvChain.mt0.csv')

# Extract the original netlist from the input file, and store it in the variable ORIGINAL_NETLIST.
ORIGINAL_NETLIST = read_original_infile()

# Define the values for N and fan that we intend to test. We will do a nested loop over all combinations.
# These are also easily changed to test different N and fan values.
n_values = np.arange(1, 15+1, step=2)
fan_values = np.arange(1, 11+1, step=1)

# Initialize the best delay variables so that I can print the best delay at the end of the program.
best_delay = 1000
best_n = 0
best_fan = 0

# Begin to loop over the N and fan values.
for n_value in n_values:
    for fan_value in fan_values:

        # Make a netlist based on this combination of N and fan.
        make_new_netlist(n_value, fan_value)

        # Get the high to low propagation delay from hspice.
        tphl = get_tphl_from_hspice()

        # If this is the best propagation delay we have seen so far, then make note of it.
        if tphl < best_delay:
            best_n = n_value
            best_fan = fan_value
            best_delay = tphl

        # Print the N, fan, and tphl values to the terminal just as requested in the prompt.
        print(f'N {n_value} fan {fan_value} tphl {tphl:.3e}')
        
# Print the best results to the terminal. During testing, this happened with 7 inverters and a fan of 3, which produced tphl = 0.72 ns.
print(f'Best values were:\n\tfan = {best_fan}\n\tnum_inverters = {best_n}\n\ttphl = {best_delay:.3e}')