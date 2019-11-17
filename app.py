import numpy as np
import cv2
from keras.models import load_model
from keras.models import Sequential
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
import sys
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1.,
                input_shape=(640, 480, 3)))
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

model.load_weights(sys.argv[2])
def det_coeff(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return K.ones_like(v) - (u / v)
def keras_predict(model, image):
    processed = keras_process_image(image)
    angle = model.predict(processed, batch_size=16)
    steering_angle = angle[0][0] 
    gas = angle[0][1]
    brake = angle[0][2]
    print(steering_angle,gas,brake)
    
    return steering_angle


def keras_process_image(img):
    image_x = 640
    image_y = 480
    img = cv2.resize(img, (image_x, image_y))
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 3))
    return img


steer = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = steer.shape
smoothed_angle = 0

cap = cv2.VideoCapture(sys.argv[1])
while (cv2.waitKey(20) != ord('q')):
    ret, frame = cap.read()
    gray = cv2.resize(frame, (640, 480))
    steering_angle = keras_predict(model, gray)

    #print(steering_angle)
    cv2.imshow('frame', cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA))
    smoothed_angle += 0.2 * pow(abs((steering_angle - smoothed_angle)), 2.0 / 3.0) * (
        steering_angle - smoothed_angle) / abs(
        steering_angle - smoothed_angle)
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(steer, M, (cols, rows))
    cv2.imshow("steering wheel", dst)
    
    #img_all = np.concatenate((frame,dst,edges),axis=0)
    #cv2.imshow("Pilot viewer",img_all)
cap.release()
cv2.destroyAllWindows()
