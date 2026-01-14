# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 13 January 2026

# Homework 1
# I did not use AI at all to complete this assignment

##############################################
##### Problem 1: Greatest Common Divisor #####
##############################################

# The gcd input list, which contains integers and other data that I want to ignore
gcd_input_list = [40, 90, 30, 120] 

# This section will make a list of nothing but integers, regardless of what was in the original list
purely_integers_list = list()
for entry in gcd_input_list:
    if int(entry) != entry: # If the integer cast is not equal to the original entry, then it cannot be an integer
        print(f"The entry {entry} was not an integer, and so it will be ignored!")
    else:
        purely_integers_list.append(int(entry))
purely_integers_list = sorted(purely_integers_list) # Sort the list of integers

# I wrote this function from memory, although I did look some of this up on Stack Overflow yesterday
# The function that I referenced almost certainly did not use sets, that was something unique to my approach
# I am looking through every number between 1 and x, testing to see whether it is evenly divisible
# It creates and returns a set containing all of the factors
def factor(x: int):
    factors = set()
    for i in range(1, x+1):
        if x%i == 0:
            factors.add(i)
    return factors

# This section will create a set called "common_factors", whose purpose is to contain each factor common to every integer in the list
common_factors = set(factor(purely_integers_list[0])) # Initalize using the factors for the 0th entry
for integer in purely_integers_list:
    new_factors = factor(integer) # I am placing the factors for this integer into a variable called "new_factors"
    common_factors = common_factors.intersection(new_factors) # This iteration will remove any factors absent from the newly tested integer
common_factors = sorted(common_factors) # Once we are done, we can sort the set (this turns it into a list)

print(f'The GCD of the integers in this list is: {common_factors[-1]}') # Of course, the largest entry is the GCD

###########################################
##### Problem 2: Prime Number Checker #####
###########################################

# The prime input list, which contains integers and other data that I want to ignore
prime_input_list = [31.0, 17, 81.4, 28]

for entry in prime_input_list:
    if int(entry) != entry: # This tests whether the entry is an integer
        print(f'{entry} is NOT an integer. Skipping...')
        continue # This will skip to the next iteration of the for loop
    else:
        entry = int(entry) # I need to cast this as an integer, because otherwise my function wouldn't work

    factors = factor(entry)
    if len(factors) == 2: # If the number only has two factors, then it is prime
        print(f'{entry} is a prime')
    else:
        print(f'{entry} is not a prime')

############################################
##### Problem 3: Reimann-Zeta Function #####
############################################

number_terms = 10000 # This is the number of terms I am using to approximate the Reimann-Zeta function
single_integer = int(input('Input an integer: ')) # Accepts an integer input from the user

running_total = 0 # Initialize the running total to be 0
for n in range(1, number_terms+1): # Iterate from 1 until the number of terms desired
    running_total += 1/(n**single_integer) # Each iteration should add a small amount to the running total

print(f'Zeta({single_integer}) = {running_total:.4f} based on {number_terms} terms') # This outputs the approximation