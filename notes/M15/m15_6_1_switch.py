# example of using a dictionary as a switch statement

# Create the dictionary
switch = { 0:'blue', 1:'red', 2:'green', 3:'teal', 4:'pink' }

# Get the value
pick = int(input("pick a number between 0 and 4, inclusive: "))

# make sure that the value is in the dictionary
# this isn't necessary if the value is previously constrained
if pick in switch:
    color = switch[pick]                 # get the value!
    print('Your color is',color)
else:
    print('Please follow directions!')   # print a error message

