input_list = [40, 90, 30, 120] # The input list, which contains integers and other data that I want to ignore

purely_integers_list = list() # This list will only contain integers
for entry in input_list:
    if int(entry) != entry: # If the integer cast is not equal to the original entry, then it cannot be an integer
        print(f"The entry {entry} was not an integer, and so it will be ignored!")
    else:
        purely_integers_list.append(int(entry))
purely_integers_list = sorted(purely_integers_list) # Sort the list of nothing but integers

####################################################################################
# I found these functions online, forums like Stack Overflow and not AI
def find_factors(num):
    factors = list()
    for i in range(1, num + 1):
        if num % i == 0:
            factors.append(i)
    return sorted(factors, reverse=True)

def find_gcd(a, b):
    while b:
        a, b = b, a%b
    return a
####################################################################################

# Alright, so I'm initializing this loop by setting the gcd_thus_far variable to be the 0th entry in the purely_integers_list variable
# That is because I want the first iteration to compare that 0th entry with itself, and return the 0th entry for the next loop iteration
# The second loop is when real decisions start happening. We find the GCD of that 0th entry with the 1st entry, for example that might be 5
# On the next iteration, we are comparing that result (5) with the next entry in the list
# We keep doing this, comparing the most recent GCD with the next component in the list, until we run out of elements in the list
# At that point, we have found the GCD of the entire list
gcd_thus_far = purely_integers_list[0] # Initialize the loop
for entry in purely_integers_list:
    gcd_thus_far = find_gcd(gcd_thus_far, entry)
final_gcd = gcd_thus_far
print(f'The GCD of the integers in the list is: {final_gcd}')