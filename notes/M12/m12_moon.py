# Image processing of moon landing photo.

# author: allee updated by sdm

import numpy as np                     # import packages
import matplotlib.pyplot as plt
import matplotlib.cm as cm             # a colormap
import matplotlib.mlab as mlab         # matlab compatible calls

im = plt.imread('/home/steven-wallace/Documents/asu/eee-419/notes/M12/m12_moon.png')        # load the image

im_f = np.fft.fft2(im)                 # Fourier transform the image
power = abs(im_f)                      # Power spectrum of the image

# Copy im_f so we can retain the original if needed
# The low frequencies are in the corners, so we eliminate everything else
im_f_clean = im_f.copy()
im_f_clean[:,50:-50] = 0               # set all rows, column 50 thru -51 to 0
im_f_clean[50:-50,:] = 0               # set all columns, row 50 thru -51 to 0
powernew = abs(im_f_clean)             # power spectrum of cleaned image

# Reconstruct the image from the clipped FT data - inverse fft
im_new = np.fft.ifft2(im_f_clean).real

# plotting the images one by one; this is the original
plt.imshow(im,cmap=cm.gray)
plt.title('original photo')
plt.show()

# cutting the power enables contrast to show...
# flattening the array allows prctile to compute power_cut, or 95th percentile
# and then set_clim sets the min and max values to display

img=plt.imshow(power,cmap=cm.Blues)      # plot the original power
power_cut = 95.0
clipped_power = np.percentile(power.flatten(), power_cut) # make 1D and cut
img.set_clim(0, clipped_power)                            # clip it
plt.title('all power with clipping')
plt.show()

img2=plt.imshow(powernew,cmap=cm.Blues)  # do the same with the new power
power_cut = 95.0
clipped_power = np.percentile(power.flatten(), power_cut)
img2.set_clim(0, clipped_power)
plt.title('cleaned power with clipping')
plt.show()

plt.imshow(im_new,cmap=cm.gray)          # and show the filtered image
plt.title('cleaned image')
plt.show()

