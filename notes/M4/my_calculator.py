# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Task 4b
# I did not use AI at all to complete this assignment

import tkinter as tk
import numpy as np
import re
import sys

def append_to_screen(text):
    screen.insert('end', text)

def surround_with_function(text):
    screen.insert(0, text + "(")
    screen.insert('end', ')')

def clear_screen():
    screen.delete(0, 'end')

def sign_operation():
    current_contents = screen.get()
    if current_contents[0] == '-':
        screen.delete(0, 1)
    else:
        screen.insert(0, '-')

def calculate():
    # Get the contents of the screen
    screen_contents = screen.get()

    number_periods = 0
    for char in screen_contents:
        if char == '.':
            number_periods += 1
        if number_periods > 1:
            print('Error, too many decimal points!')
            root.destroy()
            sys.exit()

    if ('log' in screen_contents):
        step1 = screen_contents.split('(')[1]
        step2 = step1.split(')')[0]
        inside = eval(step2)
        result = np.log10(inside)
    elif ('sin' in screen_contents):
        step1 = screen_contents.split('(')[1]
        step2 = step1.split(')')[0]
        inside = eval(step2)
        result = np.sin(inside)
    elif ('cos' in screen_contents):
        step1 = screen_contents.split('(')[1]
        step2 = step1.split(')')[0]
        inside = eval(step2)
        result = np.cos(inside)
    else:
        result = eval(screen_contents)

    clear_screen()
    screen.insert(0, f'{result:.2f}')

root = tk.Tk()
root.title('Calculator')

screen = tk.Entry(root, text='0', width=35, justify='right')

tk.Button(root, text='**', width=5, height=2, command= lambda i='**': append_to_screen(i)).grid(row=1, column=0)
tk.Button(root, text='log', width=5, height=2, command= lambda i='log': surround_with_function(i)).grid(row=1, column=1)
tk.Button(root, text='sin', width=5, height=2, command= lambda i='sin': surround_with_function(i)).grid(row=1, column=2)
tk.Button(root, text='cos', width=5, height=2, command= lambda i='cos': surround_with_function(i)).grid(row=1, column=3)

tk.Button(root, text='C', width=5, height=2, command=clear_screen).grid(row=2, column=0)
tk.Button(root, text='+/-', width=5, height=2, command=sign_operation).grid(row=2, column=1)
tk.Button(root, text='%', width=5, height=2).grid(row=2, column=2)
tk.Button(root, text='/', width=5, height=2, command= lambda i='/': append_to_screen(i)).grid(row=2, column=3)
tk.Button(root, text='7', width=5, height=2, command= lambda i=7: append_to_screen(i)).grid(row=3, column=0)
tk.Button(root, text='8', width=5, height=2, command= lambda i=8: append_to_screen(i)).grid(row=3, column=1)
tk.Button(root, text='9', width=5, height=2, command= lambda i=9: append_to_screen(i)).grid(row=3, column=2)
tk.Button(root, text='*', width=5, height=2, command= lambda i='*': append_to_screen(i)).grid(row=3, column=3)
tk.Button(root, text='4', width=5, height=2, command= lambda i=4: append_to_screen(i)).grid(row=4, column=0)
tk.Button(root, text='5', width=5, height=2, command= lambda i=5: append_to_screen(i)).grid(row=4, column=1)
tk.Button(root, text='6', width=5, height=2, command= lambda i=6: append_to_screen(i)).grid(row=4, column=2)
tk.Button(root, text='-', width=5, height=2, command= lambda i='-': append_to_screen(i)).grid(row=4, column=3)
tk.Button(root, text='1', width=5, height=2, command= lambda i=1: append_to_screen(i)).grid(row=5, column=0)
tk.Button(root, text='2', width=5, height=2, command= lambda i=2: append_to_screen(i)).grid(row=5, column=1)
tk.Button(root, text='3', width=5, height=2, command= lambda i=3: append_to_screen(i)).grid(row=5, column=2)
tk.Button(root, text='+', width=5, height=2, command= lambda i='+': append_to_screen(i)).grid(row=5, column=3)
tk.Button(root, text='0', width=14, height=2, command= lambda i=0: append_to_screen(i)).grid(row=6, column=0, columnspan=2)
tk.Button(root, text='.', width=5, height=2, command= lambda i='.': append_to_screen(i)).grid(row=6, column=2)
tk.Button(root, text='=', width=5, height=2, command=calculate).grid(row=6, column=3)

screen.grid(row=0, column=0, columnspan=4)

root.mainloop()