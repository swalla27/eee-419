# pair plotting example
# author: sdm

import pandas as pd                              # needed to read the data
import matplotlib.pyplot as plt                  # used for plotting
import seaborn as sns                            # data visualization

iris = pd.read_csv('m5_2_7_iris.csv')                # load the data
sns.set(style='whitegrid', context='notebook')   # set the apearance
sns.pairplot(iris,height=1.5)                    # create the pair plots
plt.show()                                       # and show them