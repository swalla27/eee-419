############ This code creates a Basic (+) calculator:

from tkinter import *

# Global variable to store first number and operation
first_number = None

def update_textbox(num):
    """ Append the clicked number to the textbox """
    entry.insert(END, num)

def add():
    """ Store the first number and clear the textbox for second input """
    global first_number
    first_number = entry.get()  # Store the first number
    entry.delete(0, END)  # Clear textbox

def calculate():
    """ Perform addition and display result """
    global first_number
    if first_number is not None:
        second_number = entry.get()
        try:
            result = float(first_number) + float(second_number)
            entry.delete(0, END)
            entry.insert(0, str(result))  # Display result
        except ValueError:
            entry.delete(0, END)
            entry.insert(0, "Error")  # Handle invalid input
        first_number = None  # Reset for next calculation

# Create the main window
tk = Tk()
tk.title("Simple Calculator")

# Create a frame
frame = Frame(tk, relief=RIDGE, borderwidth=2)
frame.pack(fill=BOTH, expand=1)

# Create a label
label = Label(frame, text="Enter a number:")
label.pack(fill=X, expand=1)

# Create a textbox (Entry widget)
entry = Entry(frame)
entry.pack(fill=X, expand=1)

# Create a frame for buttons
button_frame = Frame(frame)
button_frame.pack()

# Create number buttons (0-9)
for i in range(10):
    button = Button(button_frame, text=str(i), command=lambda i=i: update_textbox(str(i)), width=5, height=2)
    button.grid(row=i // 5, column=i % 5, padx=5, pady=5)  # Arrange buttons in a 2-row grid

# Add "+" button
plus_button = Button(button_frame, text="+", command=add, width=5, height=2)
plus_button.grid(row=2, column=0, padx=5, pady=5)

# Add "=" button to calculate result
equal_button = Button(button_frame, text="=", command=calculate, width=5, height=2)
equal_button.grid(row=2, column=1, padx=5, pady=5)

# Run the main event loop
tk.mainloop()