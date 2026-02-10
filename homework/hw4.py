# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Homework 4

###############################
##### Notes about sources #####
###############################

# I did not use AI at all to complete this assignment

"""I knew how to do most of this on my own. My final solution used tkinter String Variables for each of the entry fields,
because that allowed me to access those things inside the calculate function. I used google to find how to set default values
for each field, I did not remember that off the top of my head. I knew how to use the grid method inside tkinter beforehand,
and I prefer that method to pack, which is what the lectures used. I also looked up how to trim the zeros off a numpy array
using np.trim_zeros().'b' as the second argument makes the function only remove the trailing zeros. I needed to reference the notes to 
figure out how to update the wealth at retirement label dynamically, but adapting that idea to this program was pretty simple. 
I messed around with the padding for the widgets a little, but that idea came straight from the tkinter documentation. Zero AI 
whatsoever, and the sources I consulted were package websites (numpy, tkinter, etc.)"""

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import sys

# Create the root, then define M (the number of simulations) and N (the number of years until the end)
root = tk.Tk()
root.title('Retirement Calculator')
M = 20
N = 60

#################################################
##### Create entries and labels for the GUI #####
#################################################

# This section will make the labels on the left side of the GUI
fields_list = ['Mean Return (%)', 
          'Std Dev Return (%)', 
          'Yearly Contribution ($)', 
          'No. Years of Contribution', 
          'No. Years to Retirement', 
          'Annual Retirement Spend']

for idx, field in enumerate(fields_list):
    tk.Label(root, text=field).grid(row=idx, column=0, padx=20)

# This section will make the entries on the right side of the GUI. Each one gets a tkinter String Variable
mean_return = tk.StringVar()
std_dev_return = tk.StringVar()
yearly_contribution = tk.StringVar()
years_of_contribution = tk.StringVar()
years_to_retirement = tk.StringVar()
annual_retirement_spend = tk.StringVar()

# Each entry also gets a default value. These values came from the homework prompt
mean_return.set('6')
std_dev_return.set('20')
yearly_contribution.set('10000')
years_of_contribution.set('30')
years_to_retirement.set('40')
annual_retirement_spend.set('80000')

# Each of the six variables becomes a tkinter entry widget
a = tk.Entry(root, textvariable=mean_return)
b = tk.Entry(root, textvariable=std_dev_return)
c = tk.Entry(root, textvariable=yearly_contribution)
d = tk.Entry(root, textvariable=years_of_contribution)
e = tk.Entry(root, textvariable=years_to_retirement)
f = tk.Entry(root, textvariable=annual_retirement_spend)

# Now we are finally putting the entry widgets into the root using a grid system
a.grid(row=0, column=1, padx=20)
b.grid(row=1, column=1, padx=20)
c.grid(row=2, column=1, padx=20)
d.grid(row=3, column=1, padx=20)
e.grid(row=4, column=1, padx=20)
f.grid(row=5, column=1, padx=20)

###########################################################
##### The function that runs when you press calculate #####
###########################################################

def calculate_fx(mean_return, std_dev_return, yearly_contribution, years_of_contribution, 
                 years_to_retirement, annual_retirement_spend, output_label):

    """This function will run every time the user presses the 'Calculate' button. Of course, it will be \
       slightly different each time because there is random noise involved.\n
       It takes all entries as inputs, and in addition it also takes the output label in the second to last row.\n
       It will create a graph with M number of simulations over N number of years, and then calculate the average wealth at retirement."""
    
    ###########################
    ##### Data validation #####
    ###########################

    # Attempt to extract the required integers from the entry fields and place them in six different variables
    try:
        r = int(mean_return.get()) # The mean return for the retirement account
        Y = int(yearly_contribution.get()) # The yearly contribution to the retirement account
        S = int(annual_retirement_spend.get()) # The amount of money spent in retirement per year
        sigma = int(std_dev_return.get()) # The standard deviation of the return, which is proportional to the noise 
        A = int(years_of_contribution.get()) # The number of years this person contributes to their retirement account
        B = int(years_to_retirement.get()) # The number of years until retirement
    except:
        print('Error! There was a problem with the values you entered. Please try again.')
        output_label.config(text='Invalid values were entered. Please try again.')
        return

    # The user entered incorrect values for A, B, or both. The program cannot continue in this situation
    if A>N or B>N:
        print('Error! Contributions must stop at or prior to retirement. Please try again.')
        output_label.config(text='Invalid values were entered. Please try again.')
        return

    # The user said they will retire before the end of their contributions. We will assume this cannot happen
    elif A>B:
        print('Error! The number of contribution years must be less than the number of years to retirement. Please try again.')
        output_label.config(text='Invalid values were entered. Please try again.')
        return

    ###############################
    ##### Run the simulations #####
    ###############################

    # Initialize the wealth at retirement list, a bankruptcy counter, and close all matplotlib figures
    wealth_at_retirement = list() # This will hold the wealth at retirement across all simulations
    bankruptcy_counter = 0 # This counts the number of simulations that lead to bankruptcy
    plt.close() # I don't like having more than one graph open at a time. This closes graphs from previous runs

    # Loop through the following section of code once for every simulation (M defines the number of simulations)
    # I don't need to capture this index, so I throw it out
    for _ in range(M):

        # Initialize the wealth array to be all zeros of length N (number of years), and noise should be the same length
        wealth = np.zeros(N)
        noise = (sigma/100)*np.random.randn(N)

        # Now I am looping over every year in this particular simulation. This time I do need the index "i", so I capture it
        for i in range(1, N):

            if i < A: # This section executes during contributions
                wealth[i] = wealth[i-1]*(1+(r/100)+noise[i-1]) + Y
                continue # No matter what happens here, we keep contributing. We will not check for zero wealth in this stage

            elif i <= B: # This section executes after contributions, but prior to retirement
                wealth[i] = wealth[i-1]*(1+(r/100)+noise[i-1])

            else: # This section executes after retirement
                wealth[i] = wealth[i-1]*(1+(r/100)+noise[i-1]) - S

            # This check will occur every year in which we are not contributing (here, that is i > A)
            if wealth[i] < 0: # I am checking to see if we have reached bankruptcy
                wealth[i] = 0
                bankruptcy_counter += 1 # Add one to the bankruptcy counter
                break # Break out of this simulation, we're done here

        # I am going to plot this simulation on the figure, but trim the trailing zeros off
        plt.plot(np.trim_zeros(wealth, 'b'), marker='x') 

        # I am appending the wealth at retirement to this list we initialized earlier
        wealth_at_retirement.append(wealth[B])

    #####################################
    ##### All simulations completed #####
    #####################################

    # Find the average wealth at retirement and update the GUI with this information
    # print(f'{(bankruptcy_counter/M)*100:.0f}% of simulations resulted in bankruptcy.')

    avg_wealth_ret = int(sum(wealth_at_retirement)/len(wealth_at_retirement))
    output_label.config(text=f'Wealth at Retirement: ${avg_wealth_ret:,d}')

    # Add axis labels and a title to the graph
    plt.xlabel('Years')
    plt.ylabel('Wealth ($)')
    plt.title(f'Wealth Over {N} Years')

    # I like to have vertical lines showing me when contributions ended and when retirement began. This is personal preference
    # plt.axvline(A, color='black', linewidth=1)
    # plt.axvline(B, color='black', linewidth=1)

    # Show the grid and then the plot
    plt.grid(True)
    plt.show()

#################################################################
##### All failure and quit operations lead to this function #####
#################################################################

def shut_down_action():
    """I wanted a function that I could call throughout the program whose function is to kill everything.\n
       This function will destroy the root, then close all matplotlib figures, and finally exit the program."""
    
    root.destroy()
    plt.close()
    sys.exit()

#####################################################
##### Final three widgets, then root.mainloop() #####
#####################################################

# This makes the default value for the "Wealth at retirement" field near the bottom of the GUI. It will be updated each time you press "Calculate"
output_label = tk.Label(root, text='Wealth at retirement: Unknown')
output_label.grid(row=6, column=0, columnspan=2)

# This section makes the quit and calculate buttons on the bottom row of the GUI. Their actions should be pretty obvious at this point
tk.Button(root, text='Quit', width=12, command=shut_down_action).grid(row=7, column=0, pady=10)
calculate_button = tk.Button(root, text='Calculate', width=12, 
          command=lambda: calculate_fx(mean_return, std_dev_return, yearly_contribution, years_of_contribution, 
                                       years_to_retirement, annual_retirement_spend, output_label))
calculate_button.grid(row=7, column=1, pady=10)

# The main loop for the root, meaning we keep the window alive until either the user presses "Quit" or the "X"
root.mainloop()