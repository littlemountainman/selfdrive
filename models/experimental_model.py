from __future__ import division
import numpy as np
from keras.layers import Dense, Activation, Flatten, Conv2D, Lambda, BatchNormalization
from keras.layers import MaxPooling2D, Dropout, GRU
from keras.utils import print_summary
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import pickle
import tensorflow as tf
import cv2
import os
import numpy as np
import keras

def experimental(image_x, image_y):
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=(image_x, image_y, 1)))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(Activation('tanh'))
    model.add(MaxPooling2D((2, 2), padding='valid'))

    model.add(Flatten())
    
    model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    model.add(Dense(1024))
    model.add(Dense(256))
    model.add(Dense(64))
    model.add(Dense(1))
    
    model.save("modelv2.h5")
    filepath = "experimental.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    return model, callbacks_list


