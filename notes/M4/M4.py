####################
####### Code 1: Signal Processing
# Sinusoidal, amplitude, phase and frequency estimation
# author: allee updated by sdm

import numpy as np                     # need math and arrays
from numpy.linalg import inv           # matrix inversion
import matplotlib.pyplot as plt        # and plotting

# First, let's create a noisy set of "samples" of a cosine wave...
A = 1                  # the Amplitude
ftrue = 0.1           # true frequency
phi = 0*np.pi/4          # phase shift
N = 200                 # number of samples
var = 5            # modify the noise
mu = 0                 # additional noise

n = np.arange(0,N,1)                        # sample points 0 <= n < N
s = np.cos((2 * np.pi * ftrue * n) + phi)   # an array of cos values
wgn = np.sqrt(var) * np.random.randn(N) + mu  # an array of noise
x = A*s + wgn                               # mult by amplitude and add noise
plt.plot(n,x,label='Noisy Signal')            # plot the noisy samples
plt.xlabel('sample')
plt.ylabel('value')
plt.title('actual and sampled values')
plt.plot(n,s,label="Noise-Free Signal")         # plot the clean samples

f0 = np.arange(0.01,0.5,0.01)               # array of frequencies

# Remember that matrix products are done right to left!
# And each entry in J is:
# J = Xt * H * inv( Ht * H ) * Ht * X
# where X is the sample array and Xt its transpose
# H is the array of sine and cos entries for a particular frequency
# inv() is the inverse of the contents
# and * is the dot product

J = []                                      # maximize J to find f 
h = np.zeros((N,2))                         # create the H matrix
for f in f0:                                # for every frequency to try...
    h[:,0] = np.cos(2 * np.pi * f * n)      # column 0 gets the cosines
    h[:,1] = np.sin(2 * np.pi * f * n)      # column 1 gets the sines
    a = np.dot(h.transpose(),x)             # Ht * X
    b = inv(np.dot(h.transpose(),h))        # inverse of the product Ht * H
    c = np.dot(b,a)                         # dot the above two terms
    d = np.dot(h,c)                         # and with H
    J.append(np.dot(x.transpose(),d))       # and with Xt and into the list

# print(J)
indexmax = np.argmax(J)                     # find the index of largest value
f_est = f0[indexmax]                        # then get the corresponding freq
print('freq est', f_est, '\tvs actual', ftrue)

h[:,0] = np.cos(2 * np.pi * f_est * n)      # set up H with that frequency
h[:,1] = np.sin(2 * np.pi * f_est * n)

# apply least squares estimator
# [alpha1 alpha2] = inv(Ht * H) * Ht * X

a = np.dot(h.transpose(),x)             # Ht * X
b = inv(np.dot(h.transpose(),h))        # inverse of Ht * H
c = np.dot(b,a)                         # product of the above

a_est = np.linalg.norm(c)               # amplitude = sqrt(alpha1^2 + alpha2^2)
print('a_est', a_est, '\tvs actual',A )

phase_est = np.arctan(-c[1]/c[0])       # compute the estimated phase
print('phase_est', phase_est, '\tvs actual', phi)

# now use all the above to plot the estimated curve
s_est =  a_est * np.cos((2 * np.pi * f_est * n) + phase_est)
# plt.plot(n, s_est, label="estimated signal")
plt.legend()
plt.show()

# ###########################
# ############ Code 2: importing everything in a package
# # from numpy import pi
# # from numpy import sqrt
# # from numpy import linspace
# # import numpy as np
# # print(np.pi)
# from numpy import *         # import everything
# print(pi)

# ##########################
# ######### Code 3: GUI Hello World
# from tkinter import *
# from tkinter.constants import *
# tk = Tk()
# frame = Frame(tk, relief=RIDGE, borderwidth=2)
# frame.pack(fill=BOTH,expand=1)
# label = Label(frame, text="Hello, World")
# label.pack(fill=X, expand=1)
# ## Add a button:
# button = Button(frame,text="Exit",command=tk.destroy)
# button.pack(side=BOTTOM)

# # ## Add a button that displays a number in a textbox
# # # First: Create a textbox (Entry widget)
# # entry = Entry(frame)
# # entry.pack(fill=X, expand=1)
# # # entry.delete(0, END)  # Clear the textbox
# # # entry.insert(0, "1")  # Insert the given number

# # ## Second: Create a button:
# # button = Button(frame,text="Click Me",command=lambda: (entry.delete(0, END), entry.insert(0, "1")))
# # button.pack(side=BOTTOM)

# # # ### Get text from textbox:
# # # def capture_text():
# # #     global captured_text
# # #     captured_text = entry.get()  # Store the text in captured_text
# # #     print(f"Inside the function: {captured_text}")  # Print it to the console (optional)

# # # # Create a button that captures the text when clicked
# # # button = Button(tk, text="Capture Text", command=capture_text, font=("Arial", 14))
# # # button.pack(pady=5)
# # # # print("Outside the function",captured_text)

# tk.mainloop()

# #####################################
# ################ Code 4: diference between calling my_fun and calling my_fun()
# def my_fun():
#     x=2*3
#     print("function executed")

# print(my_fun())
# # y=mutliply_fun*2


# #########################
# ############ Code 4: Create a Basic (+-*/) calculator:

# from tkinter import *

# # Global variable to store first number and operation
# first_number = None

# def update_textbox(num):
#     """ Append the clicked number to the textbox """
#     entry.insert(END, num)

# def add():
#     """ Store the first number and clear the textbox for second input """
#     global first_number
#     first_number = entry.get()  # Store the first number
#     entry.delete(0, END)  # Clear textbox

# def calculate():
#     """ Perform addition and display result """
#     global first_number
#     if first_number is not None:
#         second_number = entry.get()
#         try:
#             result = float(first_number) + float(second_number)
#             entry.delete(0, END)
#             entry.insert(0, str(result))  # Display result
#         except ValueError:
#             entry.delete(0, END)
#             entry.insert(0, "Error")  # Handle invalid input
#         first_number = None  # Reset for next calculation

# # Create the main window
# tk = Tk()
# tk.title("Simple Calculator")

# # Create a frame
# frame = Frame(tk, relief=RIDGE, borderwidth=2)
# frame.pack(fill=BOTH, expand=1)

# # Create a label
# label = Label(frame, text="Enter a number:")
# label.pack(fill=X, expand=1)

# # Create a textbox (Entry widget)
# entry = Entry(frame)
# entry.pack(fill=X, expand=1)

# # Create a frame for buttons
# button_frame = Frame(frame)
# button_frame.pack()

# # Create a button with "1" and bind it to update the textbox
# button = Button(frame, text="1", command=lambda: update_textbox("1"))
# button.pack(side=BOTTOM)

# # # Create number buttons (0-9)
# # for i in range(10):
# #     button = Button(button_frame, text=str(i), command=lambda i=i: update_textbox(str(i)), width=5, height=2)
# #     button.grid(row=i // 5, column=i % 5, padx=5, pady=5)  # Arrange buttons in a 2-row grid

# # # Add "+" button
# # plus_button = Button(button_frame, text="+", command=add, width=5, height=2)
# # plus_button.grid(row=2, column=0, padx=5, pady=5)

# # # Add "=" button to calculate result
# # equal_button = Button(button_frame, text="=", command=calculate, width=5, height=2)
# # equal_button.grid(row=2, column=1, padx=5, pady=5)

# # Run the main event loop
# tk.mainloop()




############################
########## Code 5: Loan Calculator:

# # Example payment calculator
# # author: samsb updated by sdm

# from tkinter import *                    # import everything from tkinter

# # define the field names and their indices
# FIELD_NAMES = ['Annual Rate', 'Number of Payments', 'Loan Principle',
#                'Monthly Payment', 'Remaining Loan']

# F_RATE     = 0    # index for annual rate
# F_NUMPAY   = 1    # index for number of payments
# F_PRINC    = 2    # index for loan principle
# F_MONTHPAY = 3    # index for monthly payment
# F_REMAINS  = 4    # index for remaining loan

# NUM_FIELDS = 5    # how many fields there are

# ################################################################################
# # Function to compute monthly payment to get to final amount in specified time #
# # Input:                                                                       #
# #    entries - the list of entry fields                                        #
# # Output:                                                                      #
# #    only output is to screen (for debug) and the GUI                          #
# ################################################################################

# def monthly_payment(entries):
#    # period (monthly) rate:
#    r = (float(entries[F_RATE].get()) / 100) / 12
#    #print("r", r)

#    # get the remaining values...
#    loan = float(entries[F_PRINC].get())
#    n =  float(entries[F_NUMPAY].get())
#    remaining_loan = float(entries[F_REMAINS].get())

#    # compute the compounding and the monthly payment
#    q = (1 + r)** n
#    monthly = r * ( (q * loan - remaining_loan) / ( q - 1 ))
#    monthly = ("%8.2f" % monthly).strip()

#    # put the values into the GUI and print to the screen
#    entries[F_MONTHPAY].delete(0,END)
#    entries[F_MONTHPAY].insert(0, monthly )
#    #print("Monthly Payment: %f" % float(monthly))

# ################################################################################
# # Function to compute final balance given specified time and payments          #
# # Input:                                                                       #
# #    entries - the list of entry fields                                        #
# # Output:                                                                      #
# #    only output is to screen (for debug) and the GUI                          #
# ################################################################################

# def final_balance(entries):
#    # period (monthly) rate:
#    r = (float(entries[F_RATE].get()) / 100) / 12
#    #print("r", r)

#    # get the remaining values
#    loan = float(entries[F_PRINC].get())
#    n =  float(entries[F_NUMPAY].get()) 
#    monthly = float(entries[F_MONTHPAY].get())

#    # compute the compounding and the remaining balance
#    q = (1 + r)** n
#    remaining = q * loan  - ( (q - 1) / r) * monthly
#    remaining = ("%8.2f" % remaining).strip()

#    # put the values into the GUI and print to the screen
#    entries[F_REMAINS].delete(0,END)
#    entries[F_REMAINS].insert(0, remaining )
#    #print("Remaining Loan: %f" % float(remaining))

# ################################################################################
# # Function to create the GUI                                                   #
# # Inputs:                                                                      #
# #    root - the handle for the GUI                                             #
# # Outputs:                                                                     #
# #    entries - the list of entry fields                                        #
# ################################################################################

# def makeform(root):
#    entries = []                             # create an empty list
#    for index in range(NUM_FIELDS):          # for each of the fields to create
#       row = Frame(root)                     # get the row and create the label
#       lab = Label(row, width=22, text=FIELD_NAMES[index]+": ", anchor='w')

#       ent = Entry(row)                      # create the entry and init to 0
#       ent.insert(0,"0")

#       # fill allows the widget to take extra space: X, Y, BOTH, default=NONE
#       # expand allows the widget to use up sapce in the parent widget
#       row.pack(side=TOP, fill=X, padx=5, pady=5)   # place it in the GUI
#       lab.pack(side=LEFT)
#       ent.pack(side=RIGHT, expand=YES, fill=X)

#       entries.append(ent)                   # add it to the list

#    return entries                           # and return the list

# # start the main program
# root = Tk()                                 # create a GUI
# ents = makeform(root)                       # make the fields

# b1 = Button(root, text='Calculate "Remaining Loan"',                # add balance button
#           command=(lambda e=ents: final_balance(e)))
# b1.pack(side=LEFT, padx=5, pady=5)

# b2 = Button(root, text='Calculate "Monthly Payments"',              # add payment button
#           command=(lambda e=ents: monthly_payment(e)))
# b2.pack(side=LEFT, padx=5, pady=5)

# b3 = Button(root, text='Quit', command=root.destroy)   # add quit button
# b3.pack(side=LEFT, padx=5, pady=5)

# root.mainloop()                              # start execution
