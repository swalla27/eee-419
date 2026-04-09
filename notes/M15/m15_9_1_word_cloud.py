# example word clouds
# author: sdm

from wordcloud import WordCloud, STOPWORDS    # get packages we'll need
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import os
cwd = os.getcwd()
txtpath = os.path.join(cwd, 'notes/M15/gettysburg.txt')

getty = open(txtpath,'r').read()     # read our text
stops = set(STOPWORDS)                        # words to drop

# create the word cloud object using the stop words
getty_wc = WordCloud(stopwords=stops,background_color='white')

getty_wc.generate(getty)                      # generate the cloud
plt.imshow(getty_wc,interpolation='bilinear') # add it to a plot
plt.axis('off')                               # no need for axes
plt.show()                                    # show it!

stops.add('great')            # let's add a stopword and redraw...
getty_wc.generate(getty)      # regenerate and add to a plot
plt.imshow(getty_wc,interpolation='bilinear')
plt.axis('off')
plt.show()

# now let's add a background picture
photopath = os.path.join(cwd, 'notes/M15/Lincoln.png')
lincoln = np.array(Image.open(photopath)) # read the picture
plt.imshow(lincoln)                           # add it to a plot

# create the word cloud object with:
# mask - this sets the shape of the word cloud
# background - this is a transparent background!
getty_wc = WordCloud(stopwords=stops, mask=lincoln,
                     background_color="rgba(255,255,255,0)",mode="RGBA")

getty_wc.generate(getty)                      # generate and show
plt.imshow(getty_wc,interpolation='bilinear')
plt.axis('off')
plt.show()


