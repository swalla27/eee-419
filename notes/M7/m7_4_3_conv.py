# "Cat eyes" by andres_cadena84 is licensed under CC PDM 1.0 
# Zebra photo: This Photo by Unknown Author is licensed under CC BY-SA

import cv2                         # use to read the photo
import numpy as np                 # needed for arrays

CONV_DIM = 5                       # size of convolution
CONV_LOW = CONV_DIM//2             # middle of array
CONV_REM = (CONV_DIM-CONV_LOW)//2  # needed for Sobel array creation
CONV_HIG = 1+CONV_DIM//2           # distance from end when to stop

FILE_CAT = 'notes/M7/images/cat.jpg'
FILE_ZEB = 'notes/M7/images/zebra.jpg'
NAME_CAT = 'notes/M7/images/cat_'
NAME_ZEB = 'notes/M7/images/zeb_'
NUM_PHOTOS = 2
files = [FILE_CAT,FILE_ZEB]
names = [NAME_CAT,NAME_ZEB]

################################################################################
# Function convolve performs convolution on an image                           #
# inputs:                                                                      #
#   orig: an array holding the original image                                  #
#   dim_x,dim_y: the dimensions, in pixels, of the image                       #
#   conv: the array to use for the convolution                                 #
#   factor: value to divide the result by, default value is 1                  #
# outputs:                                                                     #
#   new_img - converted to an array                                            #
################################################################################

def convolve(orig,dim_x,dim_y,conv,factor=1):
    new_img = []                                           # list of lists
    for x_index in range(CONV_LOW,dim_x-CONV_HIG):         # for every pixel
        new_row = []                                       # list for each row
        for y_index in range(CONV_LOW,dim_y-CONV_HIG):     # mult by conv
            convolution = conv * img[x_index-CONV_LOW:x_index+CONV_HIG,
                                     y_index-CONV_LOW:y_index+CONV_HIG]
            new_row.append(convolution.sum()/factor)       # need to compensate?
        new_img.append(new_row)                            # add row to image
    return np.array(new_img)                               # return an array

# Create a variety of convolution arrays
conv_copy = np.zeros([CONV_DIM,CONV_DIM],int)    # this one makes a copy
conv_copy[CONV_LOW,CONV_LOW] = 1
print(conv_copy)

conv_blur = np.ones([CONV_DIM,CONV_DIM],int)     # this one blurs the image
print(conv_blur)

conv_soby = np.zeros([CONV_DIM,CONV_DIM],int)    # this one accentuates
conv_soby[0:CONV_REM] = -1                       # horizontal edges
conv_soby[0:CONV_REM,CONV_REM:CONV_DIM-CONV_REM] -= 1
conv_soby[-CONV_REM:] = -conv_soby[0:CONV_REM]
print(conv_soby)

conv_sobx = conv_soby.T                          # this one accentuates
print(conv_sobx)                                 # vertical edges

conv_lapl = np.zeros([CONV_DIM,CONV_DIM],int)    # this one accentuates
conv_lapl[CONV_LOW,:] = 1                        # both horizontal and
conv_lapl[:,CONV_LOW] = 1                        # vertical edges
conv_lapl[CONV_LOW,CONV_LOW] = 2 + (-CONV_DIM * 2 )
print(conv_lapl)

for index in range(NUM_PHOTOS):
    img = cv2.imread(files[index],0)             # 0 converts to grayscale
    root = names[index]

    x,y = img.shape
    print(x,y)
    cv2.imwrite(root+'out.jpg',img)              # verify it looks right

    img_copy = convolve(img,x,y,conv_copy)       # reproduce the original
    cv2.imwrite(root+'copy.jpg',img_copy)

    img_blur = convolve(img,x,y,conv_blur,CONV_DIM*CONV_DIM) # blur the image
    cv2.imwrite(root+'blur.jpg',img_blur)

    img_sharp = convolve(img,x,y,conv_copy)      # to sharpen features
    img_sharp = (3*img_sharp) - (2*img_blur)
    cv2.imwrite(root+'sharp.jpg',img_sharp)

    img_soby = convolve(img,x,y,conv_soby)       # look for horizontal features
    cv2.imwrite(root+'soby.jpg',img_soby)

    img_sobx = convolve(img,x,y,conv_sobx)       # look for vertical features
    cv2.imwrite(root+'sobx.jpg',img_sobx)

    img_lapl = convolve(img,x,y,conv_lapl)       # look for both
    cv2.imwrite(root+'lapl.jpg',img_lapl)
