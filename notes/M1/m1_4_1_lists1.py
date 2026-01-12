# Program to illustrate lists

# create an empty list
my_list = []
print(my_list)

input()    # pause here...

# create a list with lots of different things in it!
my_list = [1, 4.5, "book", 2-1j]
print(my_list)

input()    # pause here

# access things in the list - starts at 0!
for index in range(len(my_list)):              # note new function len()
    print("Entry",index,"is",my_list[index])

input()    # pause here

# add something to the list
my_list.append("new thing")
print(my_list)

input()    # pause here

# use extend to combine lists...
next_list = [0, 8, 11.2]
my_list.extend(next_list)
print(my_list)

input()    # pause here

# using append doesn't work correctly to add a list to a list!
my_list.append(next_list)
print(my_list)

input()    # pause here

# Now look at list slicing...
# Remember: indices start at 0!!!

alist = [1,2,3,4,5,6,7]
print(alist)

middle_few = alist[3:5]      # start:end, but end isn't included!
print(middle_few)

input()    # pause here

first_few = alist[:2]        # leave off the start - it starts at the beginning
print(first_few)

input()    # pause here

last_few = alist[5:]         # leave of the end - it ends at the end
print(last_few)
print(alist)                 # note that it hasn't changed!

input()    # pause here

# now add lists to each other - like extend but get a new list
list1 = [1,2,3]
list2 = [4,5,6]
list3 = list1 + list2
print(list3)

# another way to extend a list
list1 += list2
print(list1)

# insert puts in an element - lists are put in as lists
list2.insert(1,8)
print(list2)

input()    # pause here

# take the last item off the list and return it
last = my_list.pop()
print("before popping:",my_list)
print(last)
print("After popping:",my_list)

input()    # pause here

# take any item off the list and return it
third = my_list.pop(3)
print(third)
print(my_list)

input()    # pause here

# take any item off the list and don't return it
del(my_list[1:3])          # NOTE del is a function, not a method!
print(my_list)

input()

# remove an item based on its value - finds FIRST match

my_list.remove('new thing')
print(my_list)

input()

# negative index makes no sense, so have it count from the back!
print("mylist[-1] is", my_list[-1])
print("mylist[-2] is", my_list[-2])
