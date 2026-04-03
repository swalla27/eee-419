# set creation and modification examples
# author sdm

# create a set
my_set = { 1, 2, 3 }
print(my_set)

# create an empty set
my_empty = set()
print(my_empty)

# create a set from a list
my_set2 = set( [2, 3, 4] )
print(my_set2)

# copy a set
my_set3 = my_set.copy()
print(my_set3)

# add a single value to a set
my_set3.add(0)
print(my_set3)

# add sets
my_set3.update(my_set2)
print(my_set3)

# add a list to a set
my_set3.update([5,6,7])
print(my_set3)

# removing values from a set
# discard does nothing if element is not found
my_set3.discard(5)
print(my_set3)

my_set3.discard(11)
print(my_set3)

# remove will error if the element is not found
my_set3.remove(6)
print(my_set3)

# pop will remove an item from the set
# no sure way to predict which item!
item = my_set3.pop()
print(item,my_set3)

# empty the set - notice how it prints! Not a dictionary...
my_set3.clear()
print(my_set3)

# the length of the set
print('this set has',len(my_set),'entries')

# the sum of the values in the set
print('this set totals',sum(my_set))

# other functions include:
# all: True if all items are true
# any: True if any items are true
# min, max, sorted
