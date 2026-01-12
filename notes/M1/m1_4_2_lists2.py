# More list examples

# my_list = []               # create an empty list
# for index in range(10):
#     my_list.append(index)  # NOTE: you can't append to a list that doesn't exist

# print(my_list)

# input()    # pause here...

# putting values into a list...
x = 0
y = 1
z = 2
num_list = [x,y,z]
print(x,y,z,num_list)
x = 4
print(x,y,z,num_list)     # num_list didn't change!

# input()    # pause here...

# list1 = list2 does NOT create a new list, it just copies the pointer!
new_list = num_list
print(new_list)
new_list[1] = 8
print(new_list,hex(id(new_list)))
print(num_list,hex(id(num_list)))


# input()    # pause here...

# here's how to copy a list...
cp_list = num_list.copy()
cp_list[1] = 37
print(cp_list,hex(id(cp_list))) #different memory address than num_list
print(num_list,hex(id(num_list)))

# input()    # pause here...

# # mapping onto a list
# map_float_list = map(float,num_list)
# float_list = list(map(float,num_list))
# print(map_float_list,hex(id(map_float_list)))
# print(float_list,hex(id(float_list)))

# input()    # pause here...

# # other functions on lists
# print(num_list)
# sum = sum(num_list)
# len = len(num_list)
# easy_min = min(new_list)    # NOTE: avoid name collisions!
# easy_max = max(new_list)

# print("sum is", sum)
# print("len is", len)
# print("min is", easy_min)
# print("max is", easy_max)

# input()    # pause here...

# # for loops with lists...
# mylist = [ 3, "apple", 2-1j, 5.76 ]

# for entry in mylist:       # walk through the list
#     print(entry)

# input()    # pause here...

# # Careful - this is how you replicate a list
# print("original list:",new_list)
# print("double in size:",2*new_list)
# newer_list = 4*new_list                 # quadruple it!
# print("original list:",new_list)
# print("quad list:",newer_list)

# input()    # pause here

# # so, this is a different way to make a copy:
# alist = [1,2,3,4,5,6,7]
# new_list = alist[:]          # but this technique can't be used in all places...
# item = new_list.pop()        # including when the list is of objects
# print(alist)                 # for example, lists of lists, or of arrays, etc.
# print(new_list)              # it should work for numbers, characters, etc.

# input()    # pause here

# empty a list

my_list = [1,2,3]
print(my_list,hex(id(my_list)))
my_list.clear()
print(my_list,hex(id(my_list)))

input()    # pause here

# Unpacking entries in a list
# Sometimes needed when calling functions!
my_list1 = [1,2,3]
my_list2 = [4,5,6]
list3 = [ my_list1, 99, my_list2 ]
list4 = [ *my_list1, 99, *my_list2 ]
print("sublists:",list3)
print("all unpacked:",list4)
print(hex(id(my_list1)),*my_list1)
print(hex(id(my_list2)),*my_list2)

input()    # pause here

# sorting examples

alph = ['dog', 'Cat', 'bee']    # Note uppercase C
alph.sort()                     # do initial sort
print(alph)

alph.sort(reverse=True)         # reverse the order
print(alph)

alph.sort(key=str.lower)        # modifies case for sort
print(alph)                     # but leaves values unchanged
