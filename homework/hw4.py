# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Homework 4
# I did not use AI at all to complete this assignment

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import sys

root = tk.Tk()

M = 20
N = 60

fields_list = ['Mean Return (%)', 
          'Std Dev Return (%)', 
          'Yearly Contribution ($)', 
          'No. Years of Contribution', 
          'No. Years to Retirement', 
          'Annual Retirement Spend']

for idx, field in enumerate(fields_list):
    tk.Label(root, text=field).grid(row=idx, column=0)

mean_return = tk.StringVar()
std_dev_return = tk.StringVar()
yearly_contribution = tk.StringVar()
years_of_contribution = tk.StringVar()
years_to_retirement = tk.StringVar()
annual_retirement_spend = tk.StringVar()

mean_return.set('6')
std_dev_return.set('20')
yearly_contribution.set('10000')
years_of_contribution.set('30')
years_to_retirement.set('40')
annual_retirement_spend.set('80000')

a = tk.Entry(root, textvariable=mean_return)
b = tk.Entry(root, textvariable=std_dev_return)
c = tk.Entry(root, textvariable=yearly_contribution)
d = tk.Entry(root, textvariable=years_of_contribution)
e = tk.Entry(root, textvariable=years_to_retirement)
f = tk.Entry(root, textvariable=annual_retirement_spend)

a.grid(row=0, column=1)
b.grid(row=1, column=1)
c.grid(row=2, column=1)
d.grid(row=3, column=1)
e.grid(row=4, column=1)
f.grid(row=5, column=1)

def calculate_fx(mean_return, std_dev_return, yearly_contribution, years_of_contribution, 
                 years_to_retirement, annual_retirement_spend, output_label):

    try:
        r = int(mean_return.get())
        Y = int(yearly_contribution.get())
        S = int(annual_retirement_spend.get())
        sigma = int(std_dev_return.get())
        A = int(years_of_contribution.get())
        B = int(years_to_retirement.get())
    except:
        print('Error! There was a problem with the values you entered. Please try again.')
        root.destroy()
        plt.close()
        sys.exit()

    if A>=N or B>=N:
        print('Error! You have exceeded the maximum number of years. Please try again.')
        root.destroy()
        plt.close()
        sys.exit()

    elif A>B:
        print('Error! The number of contribution years must be less than the number of years to retirement. Please try again.')
        root.destroy()
        plt.close()
        sys.exit()

    wealth_at_retirement = list()
    bankruptcy_counter = 0
    plt.close('all')

    for _ in range(M):

        wealth = np.zeros(N)
        noise = (sigma/100)*np.random.randn(N)

        for i in range(1, N):

            if i < A:
                wealth[i] = wealth[i-1]*(1+(r/100)+noise[i-1]) + Y
            elif i < B:
                wealth[i] = wealth[i-1]*(1+(r/100)+noise[i-1])
            else:
                wealth[i] = wealth[i-1]*(1+(r/100)+noise[i-1]) - S

            if wealth[i] < 0:
                wealth[i:] = 0
                bankruptcy_counter += 1
                break
        
        plt.plot(wealth, marker='x')

        wealth_at_retirement.append(wealth[B])

    print(f'{(bankruptcy_counter/M)*100:.0f}% of simulations resulted in bankruptcy.')
    avg_wealth_ret = int(sum(wealth_at_retirement)/len(wealth_at_retirement))
    output_label.config(text=f'Wealth at Retirement: ${avg_wealth_ret:,d}')

    plt.xlabel('Years')
    plt.ylabel('Wealth ($)')
    plt.title(f'Wealth Over {N} Years')

    plt.axvline(A, color='black', linewidth=1)
    plt.axvline(B, color='black', linewidth=1)

    plt.grid(True)
    plt.show()

def quit_button_action():
    root.destroy()
    plt.close()
    sys.exit()

output_label = tk.Label(root, text='Wealth at retirement: Unknown')
output_label.grid(row=6, column=0, columnspan=2)

tk.Button(root, text='Quit', width=12, command=quit_button_action).grid(row=7, column=0)
calculate_button = tk.Button(root, text='Calculate', width=12, 
          command=lambda: calculate_fx(mean_return, std_dev_return, yearly_contribution, years_of_contribution, 
                                       years_to_retirement, annual_retirement_spend, output_label))
calculate_button.grid(row=7, column=1)

root.mainloop()