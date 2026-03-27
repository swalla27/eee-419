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

    ORIGINAL_NETLIST = list()

    with open(INPUT_FILE, 'r') as f:
        lines = f.readlines()
        for line in lines:

            if "fan" in line:
                break

            ORIGINAL_NETLIST.append(line)

    return ORIGINAL_NETLIST

def make_new_netlist(n_value: float, fan_value: float):

    current_netlist = ORIGINAL_NETLIST.copy()

    current_netlist.append(f'.param fan = {fan_value}\n')
    
    if n_value == 1:
        current_netlist.append('Xinv1 a z inv M=1\n')
    else:
        current_netlist.append('Xinv1 a b inv M=1\n')
    prev_node = 'b'
    next_node = 'b'

    for inverter_id in np.arange(start=2, stop=n_value+1, step=1):

        if inverter_id == n_value:
            next_node = 'z'
        else:
            next_node = chr(ord(prev_node) + 1)

        current_netlist.append(f'Xinv{inverter_id} {prev_node} {next_node} inv M=fan**{inverter_id-1}\n')

        prev_node = next_node

    current_netlist.append('.end\n')

    with open(INPUT_FILE, 'w') as f:
        for line in current_netlist:
#            print(line)
            f.write(line)

def get_tphl_from_hspice():
    proc = subprocess.Popen(["hspice", INPUT_FILE],
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    output, err = proc.communicate()
#    print(f'output: {output.decode()}')
#    print(f'err: {err.decode()}')

    # extract tphl from the output file
    data = np.loadtxt(OUTPUT_FILE, delimiter=',', skiprows=3, dtype=str)
#    print(data)
    tphl = float(data[1][0])
#    print(tphl)

    return tphl

t0 = time.time()

WORKING_FOLDER = os.getcwd()
INPUT_FILE = os.path.join(WORKING_FOLDER, 'InvChain.sp')
OUTPUT_FILE = os.path.join(WORKING_FOLDER, 'InvChain.mt0.csv')

ORIGINAL_NETLIST = read_original_infile()

n_values = np.arange(1, 15+1, step=2)
fan_values = np.arange(1, 11+1, step=1)

results_list = list()
best_delay = 1000

for n_value in n_values:
    for fan_value in fan_values:
        make_new_netlist(n_value, fan_value)
        tphl = get_tphl_from_hspice()

        entry = [n_value, fan_value, tphl]
        results_list.append(entry)

        if tphl < best_delay:
            best_delay = tphl
            best_result = entry

        print(f'N {n_value} fan {fan_value} tphl {tphl:.2e}')
        
print(f'Best Delay:\n\tN = {best_result[0]}\n\tfan = {best_result[1]}\n\ttphl = {best_result[2]}')
