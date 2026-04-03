# Example of class variables
# author: sdm

class planet:                          # create the class
    type = "astronomical"              # a class variable
    def __init__(self,name):           # called when object is created
        self.name = name               # instance variables

    def print(self):                   # a method
        print(self.name,self.type)

p1 = planet('Earth')                   # create some planets
p2 = planet('Mars')
p3 = planet('Venus')

p1.print()                             # print them
p2.print()
p3.print()

p1.type = 'oops'                       # not good practice!

p1.print()                             # print them
p2.print()
p3.print()

planet.type = 'fixed'                  # can change all in a class!

p1.print()                             # print them: Earth was not changed!
p2.print()
p3.print()
