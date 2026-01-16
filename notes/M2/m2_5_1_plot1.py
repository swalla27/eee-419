# plotting examples
import numpy as np                     # get the array functions
import matplotlib                      # get plotting functions
matplotlib.use('TkAgg')                # get the Tk interface
import matplotlib.pyplot as plt        # get plotting functions

# create some empty arrays to plot and the corresponding x values
func_0 = np.zeros(10,int)
func_1 = np.zeros(10,int)
x_val  = np.zeros(10,int)

for index in range(10):
    x_val[index]  = index          # set the x values
    func_0[index] = index          # create y=x
    func_1[index] = 2*index        # create y=2x

# do a single function plot...
mng = plt.get_current_fig_manager()       # make it full screen
mng.resize(*mng.window.maxsize())
plt.plot(x_val,func_0,label='happiness')  # data to plot
plt.xlabel("distance from GWC")           # set axis labels
plt.ylabel("happiness factor")
plt.title("A First Plot")
plt.legend()
plt.show()                                # create the plot

# NOTE: program automatically stalls until the plot is closed!
# Except in spyder and some other environments...

# create background and foreground colors... NOT recommended
fig = plt.figure()                   # set the external background color
fig.set_facecolor('pink')
ax = plt.axes()                      # set the internal background color
ax.set_facecolor('grey')
plt.plot(x_val,func_0,label='happiness')  # data to plot
plt.xlabel("distance from GWC")           # set axis labels
plt.ylabel("happiness factor")
plt.title("A Second Plot")
plt.legend(facecolor='pink',edgecolor='green',title='Legend')
plt.show()                                # create the plot

# now do two functions...
plt.xlabel("distance from GWC")      # set axis labels
plt.ylabel("happiness factor")
plt.plot(x_val,func_0,label="y=x",marker="x",color="red")     # data to plot
plt.plot(x_val,func_1,label="y=2x",marker="+",color="blue")   # data to plot
plt.title("Two Functions")
plt.legend()
plt.show()

# one function with inferred x-axis
plt.plot(func_0**2,label='happiness')  # can do math, not too much
plt.xlabel("sample number")
plt.ylabel("happiness factor")
plt.title("Infer the X Values")
plt.legend()
plt.show()

# two functions with separate y-axes
ax1 = plt.subplot()
ax2 = ax1.twinx()
ax1.plot(x_val,func_0,'r-',label='y=x')
ax2.plot(x_val,func_1**2,'b-',label='y=(2x)^2')
plt.grid()
ax1.set_ylabel('happiness with func_0')
ax1.yaxis.label.set_color('red')
ax1.legend(loc=0)                      # "best" location automatically picked
ax2.set_ylabel('happiness with func_1')
ax2.yaxis.label.set_color('blue')
ax2.legend(loc=4)                      # 4 is lower right corner
ax1.set_xlabel('distance from GWC')
plt.title("Separate Y Axes")
plt.show()

# rotate Y label and control font size
plt.plot(func_0**2,label='happiness')  # can do math, not too much
plt.xlabel("sample number")
plt.ylabel("Y",rotation=0,fontsize=16)
plt.title("Changing Fonts",fontsize=24)
plt.legend()
plt.show()

