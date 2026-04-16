# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 16 April 2026

# Homework M13

# I did not use AI at all to complete this assignment.

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import sys
import os

#############################
##### Class Definitions #####
#############################

class Calculator():
    def __init__(self):
        pass

    @staticmethod
    def add(a: float, b: float):
        return a + b
    
    @staticmethod
    def subtr(a: float, b: float):
        return a - b
    
    @staticmethod
    def mult(a: float, b: float):
        return a * b
    
    @staticmethod
    def div(a: float, b: float):
        return a / b
    
class Scientific(Calculator):

    # Class Variable for the logarithm and exponent methods.
    x = 2

    def log(self, a: float):
        """
        Return the base-2 logarithm of a using the change of base formula.
        """

        return np.log(a) / np.log(self.x)
    
    def exp(self, a: float):
        """
        Return the value of 2 raised to the x.
        
        """

        return self.x ** a
    
    @staticmethod
    def sin(a: float):
        return np.sin(a)
    
    @staticmethod
    def cos(a: float):
        return np.cos(a)
    
class Graphical(Scientific):

    @staticmethod
    def graph(y: np.array):
        num_elements = len(y)
        x = np.linspace(0, 100, num_elements)

        plt.plot(x, y)
        plt.xlabel('X-Axis')
        plt.ylabel('Y-Axis')
        plt.title('Homework M13 Graphical Calculator')
        plt.grid(True)
        plt.show()

###################################
##### Testing the calculators #####
###################################

# Create instances of all three classes.
c = Calculator()
s = Scientific()
g = Graphical()

# Demonstrate the functionality of the basic calculator.
a = 10
b = 5
result = c.add(a, b)
print(f'Adding {a} and {b} gives {result}.')

# Demonstrate the functionality of the scientific calculator.
a = 20
result = s.log(20)
print(f'The base-{s.x} logarithm of {a} is {result:.4f}.')

# Demonstrate the functionality of the graphical calculator.
y = np.arange(0, 50, 1)
g.graph(np.arange(0, 50, 1))


#######################
##### Tkinter GUI #####
#######################

def update_result(result_label, newresult: str):
    newtext = f'Result = {newresult:.2f}'
    result_label.config(text=newtext)


root = tk.Tk()
root.title('HW13 Calculator')

a = tk.StringVar()
b = tk.StringVar()

a.set('15')
b.set('20')

entry_a = tk.Entry(root, textvariable=a).grid(row=0, column=0)
entry_b = tk.Entry(root, textvariable=b).grid(row=1, column=0)

result_label = tk.Label(root, text='Result = ')
result_label.grid(row=2, column=0, rowspan=2)

add_button = tk.Button(root, text='+', width=10, height=4,
                       command=lambda: update_result(result_label, c.add(int(a.get()), int(b.get())))
                       ).grid(row=0, column=1)

sub_button = tk.Button(root, text='-', width=10, height=4,
                       command=lambda: update_result(result_label, c.subtr(int(a.get()), int(b.get())))
                       ).grid(row=1, column=1)

mult_button = tk.Button(root, text='x', width=10, height=4,
                        command=lambda: update_result(result_label, c.mult(int(a.get()), int(b.get())))
                        ).grid(row=2, column=1)

div_button = tk.Button(root, text='/', width=10, height=4,
                       command=lambda: update_result(result_label, c.div(int(a.get()), int(b.get())))
                       ).grid(row=3, column=1)


root.mainloop()