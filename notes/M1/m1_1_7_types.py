# Program to demonstrate data types and conversions

an_int = 5
a_float = 4.5
a_complex = 2 + 3j
a_big_float = 4e25
a_small_float = 5e-10

print(an_int)
print(a_float)
print(a_complex)
print(a_big_float)
print(a_small_float)

input()    # pause here

# now, let's convert
was_int = float(an_int)
print("this was an integer",was_int)

was_int2 = complex(an_int)
print("this was also an integer",was_int2)

input()    # pause here

# do some math for fun; note what happens to the 2
new_val = a_complex - 2
print("what now:",new_val)

# on the fly conversions...
want_a_float = float("4")
print("what happened to the string:",want_a_float)

input()    # pause here

# what am I?
print("an_int is a",type(an_int))
print("a_float is a",type(a_float))
print("a_complex is a",type(a_complex))

# declaring variables without assignment
my_int = int()      # it's an integer - default value 0
print("my_int is",type(my_int),"with value",my_int)

input()    # pause here

# what about strings? NOTE: both quote types work, but be careful!!!
msg1 = "I'm a double-quoted string"
msg2 = 'I am a single-quoted string'
print(msg1)
print(msg2)
print("msg1 is a", type(msg1))
print("msg2 is a", type(msg2))

input()    # pause here

# convert between numbers and characters
a_val = ord('a')    # ord converts characters to ASCII values
print("ASCII value of 'a' is:",a_val)

b = chr(98)         # chr converts ASCII values to characters
print("Character corresponding to ASCII 98 is:",b)

input()    # pause here

# cool thing about assignments!
# can assign multiple things in one go
# especially useful when we get to functions!
thing1, thing2 = 23, 47
print("thing1", thing1, "thing2", thing2)

# and can swap them!
thing1, thing2 = thing2, thing1
print("thing1", thing1, "thing2", thing2)

input()    # pause here

# input() returns string!
answer = input("Enter a number: ")           # add a space
print(type(answer))                          # it's a string!
number = int(input("Enter a new number: "))  # cast to an integer
print(type(number))                          # Success!
