from __future__ import division
import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, BatchNormalization, Convolution2D, AveragePooling2D, ELU, ReLU
from keras.layers import MaxPooling2D, Dropout, GRU
from keras.utils import print_summary
from keras.utils import plot_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import pickle
import tensorflow as tf
import cv2
import os
import numpy as np
import keras

def mod_elu():
    ELU(alpha=10.0)
    pass 

input_shape = (640,480,3)
def experimental(image_x, image_y):
    model = Sequential()
    
    model.add(Lambda(lambda x: x/127.5 - 1.,
                input_shape=(image_x, image_y, 3)))
    model.add(BatchNormalization())
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.1))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(3))
        
    model.save("learningmodel.h5")
    filepath = "experimental.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    return model, callbacks_list


