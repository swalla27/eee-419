# Example payment calculator
# author: samsb updated by sdm

import PySimpleGUI as sg          # get the higher-level GUI package

# define the field names and their indices
FN_RATE     = 'Annual Rate'
FN_NUMPAY   = 'Number of Payments'
FN_PRINC    = 'Loan Principle'
FN_MONTHPAY = 'Monthly Payment'
FN_REMAINS  = 'Remaining Loan'

FIELD_NAMES = [ FN_RATE, FN_NUMPAY, FN_PRINC, FN_MONTHPAY, FN_REMAINS ]

NUM_FIELDS = 5    # how many fields there are

B_BALANCE = 'Final Balance'    # need things in more than one place
B_PAYMENT = 'Monthly Payment'
B_QUIT    = 'Quit'
BK_PAYMNT = 'payment'          # needed to differentiate button from field

################################################################################
# Function to compute monthly payment to get to final amount in specified time #
# Inputs:                                                                      #
#    window  - the top-level widget                                            #
#    entries - the dictionary of entry fields                                  #
# Output:                                                                      #
#    only output is to screen (for debug) and the GUI                          #
################################################################################

def monthly_payment(window,entries):
   # period (monthly) rate:
   r = (float(entries[FN_RATE]) / 100) / 12
   #print("r", r)

   # get the remaining values...
   loan = float(entries[FN_PRINC])
   n =  float(entries[FN_NUMPAY])
   remaining_loan = float(entries[FN_REMAINS])

   # compute the compounding and the monthly payment
   q = (1 + r)** n
   monthly = r * ( (q * loan - remaining_loan) / ( q - 1 ))
   monthly = ("%8.2f" % monthly).strip()

   # put the values into the GUI and print to the screen
   window[FN_MONTHPAY].Update(monthly)
   #print("Monthly Payment: %f" % float(monthly))

################################################################################
# Function to compute final balance given specified time and payments          #
# Input:                                                                       #
#    window  - the top-level widget                                            #
#    entries - the dictionary of entry fields                                  #
# Output:                                                                      #
#    only output is to screen (for debug) and the GUI                          #
################################################################################

def final_balance(window,entries):
   # period (monthly) rate:
   r = (float(entries[FN_RATE]) / 100) / 12
   #print("r", r)

   # get the remaining values
   loan = float(entries[FN_PRINC])
   n =  float(entries[FN_NUMPAY]) 
   monthly = float(entries[FN_MONTHPAY])

   # compute the compounding and the remaining balance
   q = (1 + r)** n
   remaining = q * loan  - ( (q - 1) / r) * monthly
   remaining = ("%8.2f" % remaining).strip()

   # put the values into the GUI and print to the screen
   window[FN_REMAINS].Update(remaining)
   #print("Remaining Loan: %f" % float(remaining))

# set the font and size
sg.set_options(font=('Helvetica',20))

# The layout is a list of lists
# Each each entry in the top-level list is a row in the GUI
# Each entry in the next-level lists is a widget in that row
# Order is top to bottom, then left to right

layout = []                                 # start with the empty list
for index in range(NUM_FIELDS):             # for each of the fields to create
    layout.append([sg.Text(FIELD_NAMES[index]+': ',size=(20,1)), \
                   sg.InputText(key=FIELD_NAMES[index],size=(10,1))])

layout.append([sg.Button(B_BALANCE),                 \
               sg.Button(B_PAYMENT,key=BK_PAYMNT),   \
               sg.Button(B_QUIT)])

# start the window manager
window = sg.Window('Loan Calculator',layout)

# Run the event loop
while True:
    event, values = window.read()
    #print(event,values)
    if event == sg.WIN_CLOSED or event == B_QUIT:
        break
    if event == B_BALANCE:
        final_balance(window,values)
    elif event == BK_PAYMNT:
        monthly_payment(window,values)

window.close()

