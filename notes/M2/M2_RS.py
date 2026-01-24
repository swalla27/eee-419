import numpy as np
import sys

def linear(x: float):
    return x

def quadratic(x: float):
    return x**2

def exponential(x: float):
    a = 10
    b = 5
    return a*np.exp(b*x)

########### Code 2-1
######### AND/OR with integers not boolean:
array_bin_0 = np.full(4,0)
# print(type(array_bin_0[0]))
print(array_bin_0)
# print(sys.getsizeof(array_bin_0),np.size(array_bin_0))
# array_bin_0[2] = 1
print(array_bin_0,hex(id(array_bin_0)))
array_bin_1 = np.full(4,0)
array_bin_0[2] = 1
array_bin_0[3] = 1
array_bin_1[0] = 1
array_bin_1[2] = 4
# array_bool = True
# print("Bool = ",array_bool, type(array_bool))
print("array_bin_0=",array_bin_0,"\narray_bin_1=",array_bin_1)            # note empty quotes to create a space!
array_bin_2 = array_bin_0 & array_bin_1
# array_bin_1 = np.full(4,True)
print(array_bin_1)
array_bin_3 = array_bin_0 | array_bin_1
print("and:\n",array_bin_2,"\nor:\n",array_bin_3)
# print(type(array_bin_2[0]))
# # # 00000000000000000000000000000001
# # # 00000000000000000000000000000100

############ Code 2-2:
############ plotting examples
import numpy as np                     # get the array functions
import matplotlib                      # get plotting functions
matplotlib.use('TkAgg')                # get the Tk interface
import matplotlib.pyplot as plt        # get plotting functions

# create some empty arrays to plot and the corresponding x values
a = -5 
b = 5
n = 100
# func_0 = np.zeros(n,float)
# func_1 = np.zeros(n,float)
x_values  = np.linspace(a, b, n)

# # for index, x_value in enumerate(x_values):
# #     func_0[index] = x_value           # create y=x
# #     func_1[index] = x_value**2        # create y=x**2

# print("x_val =",x_values)
# print("func_0 =",func_0)
# print("func_1 =",func_1)
# plot func_0
# plt.plot(x_values,linear(x_values),label='f0=x')  # data to plot
# plot func_1 to the same plot
plt.plot(x_values,exponential(x_values),label='f1=x**2')  # data to plot
# # Fill in the Y only and let Python infer the X:
# plt.plot(func_1,label='f1=x^2')
plt.xlabel("X")           # set axis labels
plt.ylabel("Function")
plt.title("A Plot")
plt.legend
plt.grid(True)
plt.show() 

""" Never forget about adding the plt.show!!! """

print("plotting finished")


############## Code 2-3:
#### 3D Plotting:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Create a figure
fig = plt.figure()

# Add a 3D subplot
ax = fig.add_subplot(projection='3d')

# Generate data
x = np.linspace(-5, 5, n)
y = np.linspace(10, 20, n)
x_mesh, y_mesh = np.meshgrid(x, y) 

""" The meshgrid function is very powerful, make sure you know how to use this """

z_mesh = 5*(x_mesh)**2 + 10*y_mesh
print("x =\n",x)
print("y =\n",y)
print("x_mesh =\n",x_mesh)
print("y_mesh =\n",y_mesh)
print("z_mesh =\n",z_mesh)


# Plot the surface
ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='plasma')

# Set labels
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()


""" Functions are pass by value when you pass a variable (int, float..)
    Functions are pass by reference when you pass a list """

""" He said that he will give us an exam question about this """

## Try passing each
def fun_adds(x):
    x[0] = 'rat'
    print("I'm inside the function", x)
    return x
    # If you leave the "return" statement off, then the function will return None by default

x = [2, 3]
print(x)
y = fun_adds(x)
print(y)

# y = fun_adds 
""" Missing parenthesis means that you don't want to call the function yet. You want to pass it to someone else so they can open it later
    That variable can be executed just like a function afterwards, kind of weird """
# print(y(1))

# print(x)
# print(type(fun_adds))
print("After executing x =", x)


############# Code 2-4:
########### Dictionaries:
my_dict_0 = {"Name":"Ahmed" , "age":10 , "Year":2025, "Classes":["419" , "591"]}#, "Name":"John"}
print(my_dict_0)
Name_0 = my_dict_0.get("age")
print(Name_0)
Name_0_1 = my_dict_0.get("AGE","This key does not exist in the dictionary")
print(Name_0_1)


# Method 1 to create a Dict:
my_dict_1 = {"Name":"Peter" , "Age":27 , "Year":"Sophomore" , "Nickname":"Peter", "Nickname":"Pet"}

""" Putting duplicate keys in the dictionary definition line will overwrite the first.
    You cannot have identical keys no matter what, they must be unique """

print("my_dict_1=", my_dict_1)

class Student():
    def __init__(self, name: str, age: int, year: str, nickname: str, classes: list):

        """ The way you wrote this method means that every one of those arguments is required.
            If you said something like *args or *kwargs instead, that wouldn't necessarily be true.
            Of course, writing x = 10 in the arguments line would assign a default value (used unless overwritten) """

        self.name = name
        self.age = age
        self.year = year
        self.nickname = nickname
        self.classes = classes

student_list = [
    ['Peter', 27, 'Sophomore', 'Pete', ['491', '591']],
    ['Robert', 105, 'Super Senior', 'pissing me off', ['101']]
]

Students = list()
for student in student_list:
    Students.append(Student(*student))

print(Students[1].nickname)

# Method 2 to create a Dict:
my_dict_2 = {}
my_dict_2["Name"] = "John"     # if key is not there, new entry
my_dict_2["Age"] = 50
my_dict_2["Year"] = "Senior"     # if key is there, replace the value
my_dict_2["Year"] = "Freshman"     # if key is there, replace the value
print("my_dict_2=", my_dict_2)


# Method 3 to create a Dict:
keys = ["Name" , "Age" , "Year"]
values_3 = ["Henry" , 22 , "Freshman"]
my_dict_3 = dict(zip(keys,values_3))
print(my_dict_3)
# print(list(my_dict_3.keys()))

# # # Put them in a list:
list_of_dicts = [my_dict_1 , my_dict_2 , my_dict_3]
print("Complete Dict =", list_of_dicts)

# # Retrieve values:
age_3 = list_of_dicts[2].get("Age")
print(age_3)

# # # Retrieve all values corresponding to some key:
key = "Nickname"
all_ages = [d.get(key) for d in list_of_dicts if key in d] #

""" You can write a for loop in a single line, which is called list comprehension """

print("all_ages = ", all_ages)


## Storing data in a dataframe:
import pandas as pd

""" One way to build a df is to feed a list of dictionaries into pd.DataFrame() """

data_df = pd.DataFrame(list_of_dicts)
print(data_df)                  # print entire dataframe
print(data_df.loc[1,:])         # print a specific row
print(data_df.loc[:,"Age"])     # print a specific col
print(data_df.iloc[:,1])        # print a specific col by index

""" It sounds like .iloc is looking for indices, whereas .loc looks at the actual names """

import notes.M2.scatchpad as scratch

print(scratch.a)

""" How to reference a module from a different folder!!!
    https://www.reddit.com/r/learnpython/comments/xzfdd2/importing_a_module_from_a_different_folder/ """