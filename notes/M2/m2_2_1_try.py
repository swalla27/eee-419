# some random examples

# what are the keywords in Python
import keyword
print(keyword.kwlist)

input()     # pause here

# what's in a package?
print(dir(keyword))
print("is 'and' a keyword:",keyword.iskeyword("and"))
print("is 'AND' a keyword:",keyword.iskeyword("AND"))

input()     # pause here

# getting help for a function or type

help(str)                # everything associated with strings

input()     # pause here

# exception catching
try:
    bad_div = 12/0          # can be a full block of code...
except:                     # can have except lines for indidual exceptions
    print("don't do that!")

input()     # pause here

# specific exception catching
try:
    bad_div = 12/0
except ZeroDivisionError:          # only this one exception
    print("still don't do that!")
except:
    print("some other exception")  # just in case...

input()     # pause here

# default exception catching
try:
    bad_div = 12/0
except TypeError:                  # only this one exception
    print("really don't do that!")
except:
    print("some other exception")  # just in case...

