# Example payment calculator
# author: samsb updated by sdm

from tkinter import *                    # import everything from tkinter

# define the field names and their indices
FIELD_NAMES = ['Annual Rate', 'Number of Payments', 'Loan Principle',
               'Monthly Payment', 'Remaining Loan']

F_RATE     = 0    # index for annual rate
F_NUMPAY   = 1    # index for number of payments
F_PRINC    = 2    # index for loan principle
F_MONTHPAY = 3    # index for monthly payment
F_REMAINS  = 4    # index for remaining loan

NUM_FIELDS = 5    # how many fields there are

################################################################################
# Function to compute monthly payment to get to final amount in specified time #
# Input:                                                                       #
#    entries - the list of entry fields                                        #
# Output:                                                                      #
#    only output is to screen (for debug) and the GUI                          #
################################################################################

def monthly_payment(entries):
   # period (monthly) rate:
   r = (float(entries[F_RATE].get()) / 100) / 12
   #print("r", r)

   # get the remaining values...
   loan = float(entries[F_PRINC].get())
   n =  float(entries[F_NUMPAY].get())
   remaining_loan = float(entries[F_REMAINS].get())

   # compute the compounding and the monthly payment
   q = (1 + r)** n
   monthly = r * ( (q * loan - remaining_loan) / ( q - 1 ))
   monthly = ("%8.2f" % monthly).strip()

   # put the values into the GUI and print to the screen
   entries[F_MONTHPAY].delete(0,END)
   entries[F_MONTHPAY].insert(0, monthly )
   #print("Monthly Payment: %f" % float(monthly))

################################################################################
# Function to compute final balance given specified time and payments          #
# Input:                                                                       #
#    entries - the list of entry fields                                        #
# Output:                                                                      #
#    only output is to screen (for debug) and the GUI                          #
################################################################################

def final_balance(entries):
   # period (monthly) rate:
   r = (float(entries[F_RATE].get()) / 100) / 12
   #print("r", r)

   # get the remaining values
   loan = float(entries[F_PRINC].get())
   n =  float(entries[F_NUMPAY].get()) 
   monthly = float(entries[F_MONTHPAY].get())

   # compute the compounding and the remaining balance
   q = (1 + r)** n
   remaining = q * loan  - ( (q - 1) / r) * monthly
   remaining = ("%8.2f" % remaining).strip()

   # put the values into the GUI and print to the screen
   entries[F_REMAINS].delete(0,END)
   entries[F_REMAINS].insert(0, remaining )
   #print("Remaining Loan: %f" % float(remaining))

################################################################################
# Function to create the GUI                                                   #
# Inputs:                                                                      #
#    root - the handle for the GUI                                             #
# Outputs:                                                                     #
#    entries - the list of entry fields                                        #
################################################################################

def makeform(root):
   entries = []                             # create an empty list
   for index in range(NUM_FIELDS):          # for each of the fields to create
      row = Frame(root)                     # get the row and create the label
      lab = Label(row, width=22, text=FIELD_NAMES[index]+": ", anchor='w')

      ent = Entry(row)                      # create the entry and init to 0
      ent.insert(0,"0")

      # fill allows the widget to take extra space: X, Y, BOTH, default=NONE
      # expand allows the widget to use up sapce in the parent widget
      row.pack(side=TOP, fill=X, padx=5, pady=5)   # place it in the GUI
      lab.pack(side=LEFT)
      ent.pack(side=RIGHT, expand=YES, fill=X)

      entries.append(ent)                   # add it to the list

   return entries                           # and return the list

# start the main program
root = Tk()                                 # create a GUI
ents = makeform(root)                       # make the fields

b1 = Button(root, text='Final Balance',                # add balance button
          command=(lambda e=ents: final_balance(e)))
b1.pack(side=LEFT, padx=5, pady=5)

b2 = Button(root, text='Monthly Payment',              # add payment button
          command=(lambda e=ents: monthly_payment(e)))
b2.pack(side=LEFT, padx=5, pady=5)

b3 = Button(root, text='Quit', command=root.destroy)   # add quit button
b3.pack(side=LEFT, padx=5, pady=5)

root.mainloop()                              # start execution
