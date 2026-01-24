# plotting examples
# @author: olhartin@asu.edu updated by sdm

import numpy as np                  # needed for creating data to plot
import matplotlib.pyplot as plt     # needed to create the plots

x = np.linspace(0,10,11,float)      # generate x values
y1 = x**2                           # a first function
y2 = 2*x**2 + 1.0                   # a second function

# subplot(x,y,z) says create a subplot where there are
# x rows of subplots
# y columns of subplots
# and this one goes at index z, where index 1 is top left

plt.subplot(3,3,1)                  # 3x3 grid, position 1
plt.plot(x,y1,color='blue')         # make it blue
plt.xlabel('x')                     # label the axes
plt.ylabel('y1')
plt.grid()                          # add a grid
plt.title('plot 1: subplot(3,3,1)') # and a title

plt.subplot(3,3,3)                  # now add one at position 3
plt.plot(x,y2,color='red')
plt.xlabel('x')
plt.ylabel('y2')
plt.grid()
plt.title('plot 2: subplot(3,3,3)')


plt.subplot(3,3,(7,8))              # now at positions 7 and 8 (stretch!)
plt.plot(x,y1,color='green')
plt.plot(x,y2,color='black')
plt.xlabel('x')
plt.ylabel('y1 and y2')
plt.grid()
plt.title('plot 3: subplot(3,3,(7,8))')

plt.show()                          # show the plot

# alternate linestyles
# '-'   regular line
# '--'  dashed line
# '-.'  dots and dashes
# ':'   dots
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html

# the comma is important as plot returns a list
# in this case, a list of 1 item

plt.plot(x,y1, label='line1', linestyle=':')
plt.plot(x,y2, label='line2', linestyle='-.')

plt.xlabel('x axis')          # add labels, title, grid and show it
plt.ylabel('y axis')
plt.title('plot 5: y1 and y2 vs x ')
plt.legend()
plt.grid()
plt.show()

# color bar example

data = [[0,1,4], [8,9,12], [100, 200, 300]]          # grid data to use

fig, ax = plt.subplots()            # Note 2 return values!

# cmap is the color map to use
# interpolation is how to map the values to the colors
# vmin and vmax are how much range to show on the color bar

im = ax.imshow(data, cmap=plt.get_cmap('cool'), interpolation='nearest',
               vmin=0, vmax=15)
# https://matplotlib.org/stable/users/explain/colors/colormaps.html

fig.colorbar(im)   # put the data and color bar into the figure
plt.title("my heat diagram")
plt.show()         # show it

plt.plot(x,y1, label='line1')
plt.xlabel('x axis')                  # add labels, title, grid and show it
plt.ylabel('y axis')
plt.arrow(2,60,10,0,                   # from x=2 to x=6, y=60
          shape='full',               # a complete arrow head
          head_width=5,               # how wide the arrow head is
          head_length=.5,             # how long the arrow head is
          length_includes_head=True)  # control inclusion
plt.title('an arrow')
plt.legend()
plt.show()

