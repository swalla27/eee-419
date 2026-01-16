# dictionaries provide a connection between "keys" and "data"

# create an empty dictionary
my_dict = {}
other_dict = dict()
print(my_dict,other_dict)

input()     # pause here

# add some things to the dictionary
my_dict["hat"] = "bowler"     # if key is not there, new entry
my_dict["shirt"] = "polo"
print(my_dict)
my_dict["hat"] = "fedora"     # if key is there, replace the value
print(my_dict)

input()     # pause here

# get stuff from the dictionary
print(my_dict["hat"])
shirt_type = my_dict["shirt"]
print(shirt_type)

print("\nIterate through the keys...")
for key in my_dict:
    print("  got key:",key)
    print("  its data is:",my_dict[key])

print("\nOr, even better... iterate through both!")
for key,value in my_dict.items():
    print("  got key:", key, "; got value:", value)

input()     # pause here

# how to handle a key that isn't there? Error if it isn't and you try directly!
# get takes a key and a default to return if the key isn't present
shoes = my_dict.get("shoes","none entered")
hat = my_dict.get("hat","none entered")
print("shoes:",shoes,"; hat:",hat)

# otherwise, you have to do this before trying to extract data - checks KEYS!
if ( "hat" in my_dict ):
    print("hat is a KEY in the dictionary!")
else:
    print("hat is not a KEY in the dictionary.")

if ( "fedora" not in my_dict ):
    print("fedora is not a KEY in the dictionary!")
else:
    print("fedora is a KEY in the dictionary!")

input()     # pause here

# how to get all the values into a list?
print("a list of the values:",list(my_dict.values()))

input()     # pause here

# how many items in the dictionary?
print("how many entries:",len(my_dict))

