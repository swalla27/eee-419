# set operation examples
# author sdm

set_1 = { 1, 2, 3, 4, 5 }     # create a pair of sets
set_2 = { 3, 4, 5, 6, 7 }
print('the sets we will use')
print(set_1,set_2)

# union of two sets
set_union = set_1 | set_2
print('\nthe union')
print(set_union)

# union of two sets
set_union2 = set_1.union(set_2)
print('\nthe union method')
print(set_union2)

# intersection of two sets
set_int = set_1 & set_2
print('\nthe intersection')
print(set_int)

# intersection via the method
set_int2 = set_1.intersection(set_2)
print('\nthe intersection method - set_1 is not changed')
print(set_int2)
print(set_1)

# use intersection update method
set_3 = set_2.copy()
set_3.intersection_update(set_1)
print('\ndo intersection and update')
print(set_3)

# set difference removes second set items
set_diff = set_1 - set_2
print('\ndifference via subtraction set_1 - set_2')
print(set_diff)

# set difference removes second set items
set_diff2 = set_2 - set_1
print('\ndifference via subtraction set_2 - set_1')
print(set_diff2)

# set difference removes second set items
set_diff = set_1.difference(set_2)
print('\ndifference via method set_1 - set_2')
print(set_diff)

# set difference removes second set items
set_diff2 = set_2.difference(set_1)
print('\ndifference via method set_2 - set_1')
print(set_diff2)

# symmetric difference - items not in both sets
set_sym = set_1.symmetric_difference(set_2)
print('\nsymmetric difference via method')
print(set_sym)

# symmetric difference - items not in both sets
set_sym2 = set_1 ^ set_2
print('\nsymmetric difference via ^')
print(set_sym2)

# update a set by removing elements from the other set
set_2.difference_update(set_sym2)
print('\nremove and update elements from the set')
print(set_2)

# properties of sets
val = set_diff.isdisjoint(set_diff2)
print('\nare these sets disjoint?')
print(val)
val = set_1.isdisjoint(set_2)
print('\nwhat about these?')
print(val)

# subset
val = set_diff.issubset(set_1)
print('\nIs set_diff a subset of set_1?')
print(val)

# superset
val = set_1.issuperset(set_diff)
print('\nIs set_1 a superset of set_diff?')
print(val)

# membership test
if 3 in set_1:
    print('\nfound 3 in set_1!')

# iterating
print('\nprint the members of the set')
for thing in set_1:
    print(thing)

# enumerate
print('\nget tuples (not THE index)')
for pairs in enumerate(set_1):
    print(pairs)
