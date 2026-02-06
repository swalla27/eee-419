##### This calculator adds, subtracts, multiplies and divides.
##### It also has a button to clear the screen
from tkinter import *

# Create the main window
tk = Tk()
tk.title("Calculator")
tk.geometry("300x400")

# Entry widget to display numbers and results
entry = Entry(tk, width=20, font=("Arial", 18), borderwidth=5, justify="right")
entry.grid(row=0, column=0, columnspan=4, pady=10)

# Function to handle button clicks
def on_button_click(value):
    current_text = entry.get()
    if value == "=":
        try:
            result = eval(current_text)  # Evaluate the expression
            entry.delete(0, END)
            entry.insert(0, str(result))  # Display result
        except Exception:
            entry.delete(0, END)
            entry.insert(0, "Error")
    elif value == "C":
        entry.delete(0, END)  # Clear entry field
    else:
        entry.insert(END, value)  # Append number/operator

# Button layout
buttons = [
    ("7", "8", "9", "/"),
    ("4", "5", "6", "*"),
    ("1", "2", "3", "-"),
    ("C", "0", "=", "+"),
]

# Create and place buttons
for r, row in enumerate(buttons, start=1):
    for c, char in enumerate(row):
        Button(tk, text=char, width=5, height=2, font=("Arial", 14),
               command=lambda ch=char: on_button_click(ch)).grid(row=r, column=c, padx=5, pady=5)

# Run the main event loop
tk.mainloop()