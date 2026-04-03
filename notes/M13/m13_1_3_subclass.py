# Example creation of subclass
# author: sdm

PI = 3.14                              # define for convenience

class shape:                           # create the class
    def __init__(self,name):           # called when object is created
        self.name = name               # instance variables
        self.area = 0.0
        self.perimeter = 0.0

    def print(self):                   # print the values
        print(self.name,"has perimeter",self.perimeter,
              'and area',self.area)

class square(shape):                   # parent class in parentheses
    def __init__(self,side):
        shape.__init__(self,'square')
        self.side = side
        self._calcp_()
        self._area_()

    def _calcp_(self):                 # method to calculate perimeter
        self.perimeter = 4 * self.side

    def _area_(self):                 # method to calculate area
        self.area = self.side * self.side

    def update(self,side):             # update the side length
        self.side = side
        self._calcp_()
        self._area_()

class circle(shape):                   # parent class in parentheses
    def __init__(self,radius):
        shape.__init__(self,'circle')
        self.radius = radius
        self._calcp_()
        self._area_()

    def _calcp_(self):                 # method to calculate perimeter
        self.perimeter = 2 * PI * self.radius

    def _area_(self):                 # method to calculate area
        self.area = PI * self.radius * self.radius

    def update(self,radius):          # update the radius length
        self.radius = radius
        self._calcp_()
        self._area_()

sq2 = square(2)
sq4 = square(4)

sq2.print()
sq4.print()

sq2.update(5)
sq2.print()

cir2 = circle(2)
cir4 = circle(4)

cir2.print()
cir4.print()

cir2.update(8)
cir2.print()
