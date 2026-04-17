# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 16 April 2026

# Homework M13

# I did not use AI at all to complete this assignment.

import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from typing import Callable
import sys
import os

#############################
##### Class Definitions #####
#############################

class Calculator():
    """
    A calculator class containing methods for addition, subtraction, multiplication, and division. 
    It is the parent of the scientific and graphical calculators.
    """

    @staticmethod
    def add(a: float, b: float):
        """
        Return the sum of two numbers, called a and b. 

        Paremeters
        ----------
        a: float
            The first number to be added.
        b: float
            The second number to be added.

        Returns
        -------
        result: float
            The sum of the numbers a and b.
        """

        return a + b
    
    @staticmethod
    def subtr(a: float, b: float):
        """
        Return the difference of two numbers, called a and b.

        Paremeters
        ----------
        a: float
            The first number in the equation.
        b: float
            The value subtracted from the first.

        Returns
        -------
        result: float
            The difference of the numbers a and b.
        """

        return a - b
    
    @staticmethod
    def mult(a: float, b: float):
        """
        Return the product of two numbers, called a and b.

        Paremeters
        ----------
        a: float
            The first number in the equation
        b: float
            The value that a is multiplied by.

        Returns
        -------
        result: float
            The product of the numbers a and b.
        """

        return a * b
    
    @staticmethod
    def div(a: float, b: float):
        """
        Return the quotient of two numbers, called a and b.

        Paremeters
        ----------
        a: float
            The first number in the equation
        b: float
            The value that a is divided by.

        Returns
        -------
        result: float
            The quotient of the numbers a and b.
        """

        return a / b
    
class Scientific(Calculator):
    """
    A class which inherits arithmetic methods from Calculator, but also has logarithmic, exponential, and trigonometric functions.
    """

    # Define a class variable for the logarithm and exponent methods.
    x = 2

    def log(self, a: float):
        """
        Return the base-2 logarithm of "a" using the change of base formula.

        Parameters
        ----------
        a: float
            The argument of the logarithm function.
        
        Returns
        -------
        result: float
            The base-2 logarithm of "a." Of course, changing x would also change the base in this equation.
        """

        return np.log(a) / np.log(self.x)
    
    def exp(self, a: float):
        """
        Return the value of 2 raised to the a.

        Parameters
        ----------
        a: float
            The argument of the exponential function.
        
        Returns
        -------
        result: float
            The value of x raised to the a. Changing the class variable x would also affect this method.
        
        """

        return self.x ** a
    
    @staticmethod
    def sin(a: float):
        """
        Return the sine of an input number a.

        Parameters
        ----------
        a: float
            The input to the sine function.

        Returns
        -------
        result: float
            The sine of the value a.
        """

        return np.sin(a)
    
    @staticmethod
    def cos(a: float):
        """
        Return the cosine of an input number a.

        Parameters
        ----------
        a: float
            The input to the cosine function.

        Returns
        -------
        result: float
            The cosine of the value a.
        """

        return np.cos(a)
    
class Graphical(Scientific):
    """
    A class which inherits arithmetic methods from Calculator, but also has graphing capabilities.
    """

    @staticmethod
    def graph(y: np.array):
        """
        Create a matplotlib graph of an input array y with another array x containing values from 0 to 100.

        Parameters
        ----------
        y: np.array
            The input array to be graphed on the vertical axis.

        Returns
        -------
        None
        """

        # Create an array for the horizontal axis based on the length of the array called y.
        num_elements = len(y)
        x = np.linspace(0, 100, num_elements)

        # Plot the two arrays, label the axes, create a titlte, and display the graph.
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

def update_result(result_label: tk.Label, a: tk.StringVar, b: tk.StringVar, fx: Callable):
    """
    This function updates the label to the tkinter GUI using the two string variables and the function fx.

    Parameters
    ----------
    result_label: tk.Label
        A label in the tkinter GUI used to display the result. The present function will update that label.
    a: tk.StringVar
        A tkinter string variable. I will extract an integer and use that as one input to the function "fx."
    b: tk.StringVar
        A tkinter string variable. I will extract an integer for use as the other input to the function "fx."
    fx: Callable
        The function we are executing for this label update. For example, we might add or subtract "a" and "b".

    Returns
    -------
    None
    """
    
    in1 = int(a.get())
    in2 = int(b.get())

    result = fx(in1, in2)
    newtext = f'Result = {result}'

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
                       command=lambda: update_result(result_label, a, b, c.add)
                       ).grid(row=0, column=1)

sub_button = tk.Button(root, text='-', width=10, height=4,
                       command=lambda: update_result(result_label, a, b, c.subtr)
                       ).grid(row=1, column=1)

mult_button = tk.Button(root, text='x', width=10, height=4,
                        command=lambda: update_result(result_label, a, b, c.mult)
                        ).grid(row=2, column=1)

div_button = tk.Button(root, text='/', width=10, height=4,
                       command=lambda: update_result(result_label, a, b, c.div)
                       ).grid(row=3, column=1)

root.mainloop()