# Example creation of class
# author: sdm

class person:                          # create the class
    def __init__(self,first,last):     # called when object is created
        self.first = first             # instance variables
        self.last  = last              # values may be unique to each instance

    def print(self):                   # a method
        print(self.first,self.last)

prof = person("Steve", "Millman")      # create an instance of the class
prof.print()                           # print the instance

gravity = person("Isaac", "Newton")    # create another instance of the class
gravity.print()                        # print the instance

