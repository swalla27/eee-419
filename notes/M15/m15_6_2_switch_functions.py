# example of calling functions with a Python "switch"

# create the functions...
def blue():
    print('Your color is blue')

def red():
    print('Your color is red')

def green():
    print('Your color is green')

def teal():
    print('Your color is teal')

def pink():
    print('Your color is pink')

# Create the dictionary where the values are the functions!
switch = { 0:blue, 1:red, 2:green, 3:teal, 4:pink }

# Get the value
pick = int(input("pick a number between 0 and 4, inclusive: "))

# make sure that the value is in the dictionary
# this isn't necessary if the value is previously constrained
if pick in switch:
    color = switch[pick]                 # get the value!
    color()
else:
    print('Please follow directions!')   # print a error message

# can also use get to implement the default!
def default():
    print('Invalid entry!')

pick = int(input("pick another number between 0 and 4, inclusive: "))
color = switch.get(pick,default)
color()

