# Trains a simple convolutional neural network (CNN) on the MNIST dataset.
# Gets to 99.25% test accuracy after 12 epochs.
# (there is still a lot of margin for parameter tuning).
# 16 seconds per epoch on a GRID K520 GPU.

from __future__ import print_function              # make print compatible with 3.7
import keras                                       # import the CNN
from keras.datasets import mnist                   # get the written number data set
from keras.models import Sequential                # get the Sequential model
from keras.layers import Dense, Dropout, Flatten   # get various layers
from keras.layers import Conv2D, MaxPooling2D      # and more layers
from keras import backend as K                     # and the backend

batch_size = 128    # number of samples per gradient update
num_classes = 10    # how many classes to classify (10 digits, 0-9)
epochs = 12         # how many epochs to run trying to improve

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# take care of difference in data order
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

# convert integers in range 0-255 to floats and then divide by 255 so range is 0-1
# The book implies that transforming to -1 to 1 works better... so
# we could subtract 0.5 and then mult by 2 to achieve that...
# This is left to the interested reader.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Print details on how the data looks...
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# in this case, one column per class with a 1 in the column corresponding to the class
# for each row.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()                                 # create a linear stack of layers
model.add(Conv2D(32, kernel_size=(3, 3),             # 32 channels from 3x3
                 activation='relu',                  # rectified linear unit
                 input_shape=input_shape))           # from above
model.add(Conv2D(64, (3, 3), activation='relu'))     # 64 channels, from 3x3, RELU again
model.add(MaxPooling2D(pool_size=(2, 2)))            # max pool by 2, so now 14x14 size
model.add(Dropout(0.25))                             # randomly set 25% to 0!
model.add(Flatten())                                 # go to 1-D
model.add(Dense(128, activation='relu'))             # 128-unit RELU
model.add(Dropout(0.5))                              # set 50% to 0!
model.add(Dense(num_classes, activation='softmax'))  # how to determine the classes

# compile the model we've defined...
# categorical_crossentropy is a measure between the input and the output.
# so the loss function is determining how well we are learning;
# the better, the lower the loss function.
# Adadelta using a moving window of gradients rather than keeping all past history.
# And we'll use accuracy as our metric

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# now train on the data and validate it
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# evaluate our model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
