# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 21 January 2026

# Homework 4
# I did not use AI at all to complete this assignment

import tkinter as tk
import re
import sys

def append_to_screen(text):
    screen.insert('end', text)

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

    result = eval(screen_contents)
    clear_screen()
    screen.insert(0, result)

    # # Search for any numbers in the entry field
    # number_pattern = r'\d*\.?\d+'
    # numbers_as_strings = re.findall(number_pattern, screen_contents)
    # numbers = [float(x) if '.' in x else int(x) for x in numbers_as_strings]

    # # Search for the operators in the entry field
    # allowed_operators = ['+', '-', 'x', '\u00F7']
    # operators = list()
    # for char in screen_contents:
    #     if char in allowed_operators:
    #         operators.append(char)

    # print(numbers)
    # print(operators)

    # # Execute the multiplication or division operations
    # idx = 0
    # while operators:
    #     operator = operators[idx]
    #     if operator in ['x', '\u00F7']:
    #         if operator == 'x':
    #             result = numbers[idx] * numbers[idx+1]
    #         else:
    #             result = numbers[idx] / numbers[idx+1]
    #     elif operator in ['+', '-']:
    #         if operator == '+':
    #             result = numbers[idx] + numbers[idx+1]
    #         else:
    #             result = numbers[idx] - numbers[idx+1]
        
    #     numbers[idx] = result
    #     numbers.pop(idx+1)
    #     operators.pop(idx)
    #     idx += 1
    
    # clear_screen()
    # if isinstance(numbers[0], float):
    #     screen.insert(0, f'{numbers[0]:.3f}')
    # else:
    #     screen.insert(0, f'{numbers[0]}')

root = tk.Tk()
root.title('Calculator')

screen = tk.Entry(root, text='0', width=35, justify='right')
clear = tk.Button(root, text='C', width=5, height=2, command=clear_screen)
sign = tk.Button(root, text='+/-', width=5, height=2, command=sign_operation)
perc = tk.Button(root, text='%', width=5, height=2)
divide = tk.Button(root, text='/', width=5, height=2, command= lambda i='/': append_to_screen(i))
seven = tk.Button(root, text='7', width=5, height=2, command= lambda i=7: append_to_screen(i))
eight = tk.Button(root, text='8', width=5, height=2, command= lambda i=8: append_to_screen(i))
nine = tk.Button(root, text='9', width=5, height=2, command= lambda i=9: append_to_screen(i))
times = tk.Button(root, text='*', width=5, height=2, command= lambda i='*': append_to_screen(i))
four = tk.Button(root, text='4', width=5, height=2, command= lambda i=4: append_to_screen(i))
five = tk.Button(root, text='5', width=5, height=2, command= lambda i=5: append_to_screen(i))
six = tk.Button(root, text='6', width=5, height=2, command= lambda i=6: append_to_screen(i))
minus = tk.Button(root, text='-', width=5, height=2, command= lambda i='-': append_to_screen(i))
one = tk.Button(root, text='1', width=5, height=2, command= lambda i=1: append_to_screen(i))
two = tk.Button(root, text='2', width=5, height=2, command= lambda i=2: append_to_screen(i))
three = tk.Button(root, text='3', width=5, height=2, command= lambda i=3: append_to_screen(i))
plus = tk.Button(root, text='+', width=5, height=2, command= lambda i='+': append_to_screen(i))
zero = tk.Button(root, text='0', width=14, height=2, command= lambda i=0: append_to_screen(i))
decimal_point = tk.Button(root, text='.', width=5, height=2, command= lambda i='.': append_to_screen(i))
equals = tk.Button(root, text='=', width=5, height=2, command=calculate)

screen.grid(row=0, column=0, columnspan=4)
clear.grid(row=1, column=0)
sign.grid(row=1, column=1)
perc.grid(row=1, column=2)
divide.grid(row=1, column=3)
seven.grid(row=2, column=0)
eight.grid(row=2, column=1)
nine.grid(row=2, column=2)
times.grid(row=2, column=3)
four.grid(row=3, column=0)
five.grid(row=3, column=1)
six.grid(row=3, column=2)
minus.grid(row=3, column=3)
one.grid(row=4, column=0)
two.grid(row=4, column=1)
three.grid(row=4, column=2)
plus.grid(row=4, column=3)
zero.grid(row=5, column=0, columnspan=2)
decimal_point.grid(row=5, column=2)
equals.grid(row=5, column=3)

root.mainloop()