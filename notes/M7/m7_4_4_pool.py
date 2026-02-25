# "Cat eyes" by andres_cadena84 is licensed under CC PDM 1.0 
# Zebra photo: This Photo by Unknown Author is licensed under CC BY-SA

import cv2                         # use to read the photo
import numpy as np                 # needed for arrays

POOL_DIM = 2                       # size of stride

FILE_CAT = 'notes/M7/cat.jpg'
FILE_ZEB = 'notes/M7/zebra.jpg'
NAME_CAT = 'notes/M7/images/cat_'
NAME_ZEB = 'notes/M7/images/zeb_'
NUM_PHOTOS = 2
NUM_POOLS = 2
files = [FILE_CAT,FILE_ZEB]
names = [NAME_CAT,NAME_ZEB]
ptype = ['mean.jpg','max.jpg']

################################################################################
# Function pool performs pooling on an image                                   #
# inputs:                                                                      #
#   orig: an array holding the original image                                  #
#   dim_x,dim_y: the dimensions, in pixels, of the image                       #
#   max: 1 do max pooling; 0 do mean pooling                                   #
# outputs:                                                                     #
#   new_img - converted to an array                                            #
################################################################################

def pool(orig,dim_x,dim_y,max=1):
    new_img = []                                    # list of lists
    for x_index in range(0,dim_x,POOL_DIM):         # for each pixel, striding!
        new_row = []                                # list for each row
        for y_index in range(0,dim_y,POOL_DIM):     # for each group...
            if max:
                pool_val = img[x_index:x_index+POOL_DIM,
                               y_index:y_index+POOL_DIM].max()
            else:
                pool_val = img[x_index:x_index+POOL_DIM,
                               y_index:y_index+POOL_DIM].mean()
            new_row.append(pool_val)                # add to list
        new_img.append(new_row)                     # add to image
    return np.array(new_img)                        # return an array

for index in range(NUM_PHOTOS):
    img = cv2.imread(files[index],0)             # 0 converts to grayscale
    root = names[index]
    x,y = img.shape
    print(x,y)

    for pool_type in range(NUM_POOLS):
        img_pool = pool(img,x,y,pool_type)
        cv2.imwrite(root+ptype[pool_type],img_pool)
