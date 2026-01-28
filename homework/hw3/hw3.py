# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 28 January 2026

# Homework 3
# I did not use AI at all to complete this assignment



import numpy as np 
from numpy.linalg import solve
from read_netlist import read_netlist
import comp_constants as COMP


def get_dimensions(netlist: list):
    """This function will accept the netlist as an input, outputting the number of nodes and the number of voltage sources.\n
       It does not care what the nodes are named, they can even be strings. I'm using a set to count the number of unique nodes, which
       should be pretty robust."""

    volt_cnt = 0
    set_of_nodes = set()

    for component in netlist: # Loop through each component in the netlist
        if component[COMP.TYPE] == COMP.VS: # Determine whether we have a voltage source here, and if so, then add to the count
            volt_cnt += 1
        
        if component[COMP.I] not in set_of_nodes:
            set_of_nodes.add(component[COMP.I])
        
        if component[COMP.J] not in set_of_nodes:
            set_of_nodes.add(component[COMP.J])

    node_cnt = len(set_of_nodes)

    print(f'Number of Nodes: {node_cnt}\nNumber of Voltage Sources: {volt_cnt}')
    return node_cnt, volt_cnt

def stamper(y_add: np.array, netlist: list, currents: np.array, voltages: np.array, node_cnt: int):

    resistors = list()
    curr_sources = list()
    volt_sources = list()

    for component in netlist:
        if component[COMP.TYPE] == COMP.R:
            resistors.append(component)
        elif component[COMP.TYPE] == COMP.IS:
            curr_sources.append(component)
        elif component[COMP.TYPE] == COMP.VS:
            volt_sources.append(component)

    for resistor in resistors:
        i = resistor[COMP.I]
        j = resistor[COMP.J]

        y_add[i,i] += 1.0/resistor[COMP.VAL]
        y_add[j,j] += 1.0/resistor[COMP.VAL]
        y_add[i,j] -= 1.0/resistor[COMP.VAL]
        y_add[j,i] -= 1.0/resistor[COMP.VAL]

    for curr_source in curr_sources:
        i = curr_source[COMP.I]
        j = curr_source[COMP.J]

        currents[i] -= curr_source[COMP.VAL]
        currents[j] += curr_source[COMP.VAL]

    volt_source_id = 0
    for volt_source in volt_sources:
        i = volt_source[COMP.I]
        j = volt_source[COMP.J]

        M = node_cnt + volt_source_id

        # Step 1: The admittance matrix
        y_add[M,:] = 0
        y_add[:,M] = 0
        y_add[M,i] = 1
        y_add[i,M] = 1
        y_add[M,j] = -1
        y_add[j,M] = -1

        # Step 2: The current matrix
        currents[M] = volt_source[COMP.VAL]

        # Step 3: The voltage matrix
        voltages[M] = 0

        # Step 4: Increment the voltage source id
        volt_source_id += 1

    # I am deleting the first row and column because they would make the matrix singular
    y_add = np.delete(y_add, (0), axis = 0)
    y_add = np.delete(y_add, (0), axis = 1)
    currents = np.delete(currents, (0), axis = 0)
    voltages = np.delete(voltages, (0), axis = 0)

    return y_add, currents, voltages

################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist!
netlist = read_netlist()

# Print the netlist so we can verify we've read it correctly
for index in range(len(netlist)):
    print(netlist[index])
print("\n")

node_cnt, volt_cnt = get_dimensions(netlist)

voltages = np.zeros([node_cnt + volt_cnt, 1])
currents = np.zeros([node_cnt + volt_cnt, 1])
y_add = np.zeros([node_cnt + volt_cnt, node_cnt + volt_cnt])

y_add, currents, voltages = stamper(y_add, netlist, currents, voltages, node_cnt)
print(y_add)
voltages = solve(y_add, currents)
print(voltages)