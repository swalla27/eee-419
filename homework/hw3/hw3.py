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

    # Initialize the voltage source count variable and the set containing every node name
    volt_cnt = 0
    set_of_nodes = set()

    # Loop through each component in the netlist, adding one to the voltage source count if we 
    # find a voltage source, and adding each node number to the set when we encounter a new one
    for component in netlist:
        if component[COMP.TYPE] == COMP.VS: # Voltage Source
            volt_cnt += 1
        
        if component[COMP.I] not in set_of_nodes: # Node i
            set_of_nodes.add(component[COMP.I])
        
        if component[COMP.J] not in set_of_nodes: # Node j
            set_of_nodes.add(component[COMP.J])

    # The number of nodes is the length of the node set. This will prevent the program from counting nodes twice
    node_cnt = len(set_of_nodes)

    # Now we are ready to return the number of nodes and voltage sources
    return node_cnt, volt_cnt


def stamper(y_add: np.array, netlist: list, currents: np.array, voltages: np.array, node_cnt: int):
    """The purpose of this function is to stamp the admittance matrix and the current/voltage vectors according to the netlist.\n
       I will handle this in several steps, firstly the resistors, then the current sources, and finally the voltage sources.\n
       However, I will remove the 0th row and the 0th column at the end of this procedure.\n
       The function will output the modified admittance matrix along with the current/voltage vectors."""

    ########################################################
    ##### Sort the components into lists based on type #####
    ########################################################

    # This block of code will create three lists, where the resistors list contains all the resistors in this netlist
    # Of course, the current source and voltage source lists do a similar thing

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

    ######################################
    ##### Handle the resistors first #####
    ######################################

    # I chose to handle the resistors first because I was afraid this procedure would interfere with the voltage source procedure
    # Doing it in this order ought to prevent the resistor algorithm from writing data to the extra rows that should be zero
    # Regardless, it passes both test cases. The code below will add 1/R to the admittance matrix entries i,i and j,j
    # Of course, it will also subtract 1/R to the admittance matrix entries i,j and j,i

    for resistor in resistors:
        i = resistor[COMP.I]
        j = resistor[COMP.J]

        y_add[i,i] += 1.0/resistor[COMP.VAL]
        y_add[j,j] += 1.0/resistor[COMP.VAL]
        y_add[i,j] -= 1.0/resistor[COMP.VAL]
        y_add[j,i] -= 1.0/resistor[COMP.VAL]
    
    #################################################
    ##### Now I will handle the current sources #####
    #################################################

    # The current source procedure is pretty simple
    # All we have to do is add the source's value to the current matrix at location j and subtract it from location i

    for curr_source in curr_sources:
        i = curr_source[COMP.I]
        j = curr_source[COMP.J]

        currents[i] -= curr_source[COMP.VAL]
        currents[j] += curr_source[COMP.VAL]

    #################################################
    ##### Now I will handle the voltage sources #####
    #################################################

    # There are four steps to the voltage source procedure, and each has its own header below
    # Step 1: The Admittance Matrix
        # Firstly, I will set the entire row M and column M to be 0
        # Then, I will set locations M,i and i,M to be 1
        # Finally, locations M,j and j,M are set to be -1
    # Step 2: The Current Vector
        # Location M in the current vector should be set to the value of this voltage source
    # Step 3: The Voltage Vector
        # Location M in the voltage vector should be set to 0
    # Step 4: Increment the voltage source id
        # Add one to the voltage source id so that, on the next iteration, we complete this procedure for the next row and column

    volt_source_id = 0
    for volt_source in volt_sources:
        i = volt_source[COMP.I]
        j = volt_source[COMP.J]

        M = node_cnt + volt_source_id # I decided to use the same syntax as the lectures, where M represents the row / column being modified currently

        # Step 1: The Admittance Matrix
        y_add[M,:] = 0
        y_add[:,M] = 0
        y_add[M,i] = 1
        y_add[i,M] = 1
        y_add[M,j] = -1
        y_add[j,M] = -1

        # Step 2: The Current Vector
        currents[M] = volt_source[COMP.VAL]

        # Step 3: The Voltage Vector
        voltages[M] = 0

        # Step 4: Increment the voltage source id
        volt_source_id += 1

    ################################################################
    ##### The final step is to delete the first row and column #####
    ################################################################

    y_add = np.delete(y_add, (0), axis = 0)
    y_add = np.delete(y_add, (0), axis = 1)
    currents = np.delete(currents, (0), axis = 0)
    voltages = np.delete(voltages, (0), axis = 0)

    # Now we are ready to return the modified matrix and vectors
    return y_add, currents, voltages

##################################
##### Start the main program #####
##################################

# Read the netlist
netlist = read_netlist()

# Print the netlist so we can verify we've read it correctly
for index in range(len(netlist)):
    print(netlist[index])
print("\n")

# This will find the number of nodes and voltage sources in the netlist
node_cnt, volt_cnt = get_dimensions(netlist)

# I am initializing empty vectors and a matrix for the voltages, currents, and admittances
voltages = np.zeros([node_cnt + volt_cnt, 1])
currents = np.zeros([node_cnt + volt_cnt, 1])
y_add = np.zeros([node_cnt + volt_cnt, node_cnt + volt_cnt])

# Call the stamper function, which will modify the matrix and vectors, then return them
y_add, currents, voltages = stamper(y_add, netlist, currents, voltages, node_cnt)

# This step uses numpy to solve for the voltages based on the admittance matrix and the current vector
voltages = solve(y_add, currents)

# Print the results to the terminal in the requested format
if volt_cnt < 1:
    print(f'Voltages vector is {voltages.T[0]}')
    print('There are no voltage sources in this circuit')
else:
    print(f'Voltages and currents vector is {voltages.T[0]}')
    volt_source_currents = voltages[-volt_cnt:]
    avg_volt_source_current = sum(volt_source_currents) / len(volt_source_currents)
    print(f'Voltage source average current is {avg_volt_source_current[0]:.2f} A')