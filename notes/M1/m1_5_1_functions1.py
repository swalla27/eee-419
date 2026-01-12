# script to illustrate functions
# how to create a function
# always provide a comment - high-level description of what this function does!

################################################################################
# Function crazy_mult multiplies two numbers                                   #
# inputs:                                                                      #
#   left_num:  a first number to multiply                                      #
#   right_num: a second number to multiply                                     #
# outputs:                                                                     #
#   answer:    the product of the two numbers                                  #
#   left_num:  the original first number                                       #
#   right_num: the original second number                                      #
################################################################################

def crazy_mult(left_num, right_num):   # NOTE the colon
    answer = left_num * right_num      # indent!
    return answer, left_num, right_num # return all these values

# Function ends with outdent
ans, was_left, was_right = crazy_mult(3,4)  # multiple outputs!
print("multiplied",was_left,"by",was_right,"and got",ans)

input()      # pause here

# Notice how the function takes care of the data types!

ans, was_left, was_right = crazy_mult(3.5,2)      # a floating point input
print("multiplied",was_left,"by",was_right,"and got",ans)

ans, was_left, was_right = crazy_mult(1-1j,1+1j)  # imaginary numbers
print("multiplied",was_left,"by",was_right,"and got",ans)

input()      # pause here

# Discuss scoping...

################################################################################
# Function: loc_function uses, but can't change, a higher-level variable       #
# inputs:                                                                      #
#   in_val: a number                                                           #
# outputs:                                                                     #
#   tmp: a computed value                                                      #
################################################################################

def loc_function(in_val):    # now a function that will use it DANGER!!!!
    my_int = -5              # we think we are changing it!
    tmp = in_val + my_int    # we'll use the change locally...
    return tmp

my_int = 23                                # create a higher-level variable
print("function output:",loc_function(3))  # yup, used the local change
print("my_int:",my_int)                    # but outside it didn't change

input()      # pause here

################################################################################
# Function: global_function: uses, and changes, a global variable              #
# inputs:                                                                      #
#   val2: a number                                                             #
# outputs:                                                                     #
#   tmp: a computed value                                                      #
################################################################################

def global_function(val2):    # now a function that will change it !!!!DANGER!!!!
    global my_int
    my_int = -5               # this changes the external value!
    tmp = val2 + my_int
    return tmp

my_int = 23                                # create a higher-level variable
print("global output:",global_function(3)) # yup, used the new value
print("my_int",my_int)                     # hey - it changed!

