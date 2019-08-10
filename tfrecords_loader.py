from __future__ import division
import cv2
import os
import numpy as np
import scipy
import h5py
import matplotlib.pyplot as plt
from itertools import islice

LIMIT = 10000
DATA_FOLDER = 'driving_dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (400, 400))
    return resized

def return_data():
    X = []
    y = []
    features = []

    with open(TRAIN_FILE) as fp:
        # Processing the whole dataset
        for line in islice(fp, LIMIT):
            print("Processed line:" + str(line))
            path, angle = line.strip().split()
            full_path = os.path.join(DATA_FOLDER, path)
            X.append(full_path)
            # using angles from -pi to pi to avoid rescaling the atan in the network
            y.append(float(angle) * scipy.pi / 180)
        print("------------- Saving to numpy binary -------------")

    for i in range(len(X)):
        img = plt.imread(X[i])
        print("Processed image "+ str(i) + " of " +str(len(X)))
        features.append(preprocess(img))
    # Using numpy boosts the performance up to 2x on NVME storage
    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    #np.save("labels.npy", labels)
    with h5py.File('features.h5', 'w') as hf:
        hf.create_dataset("features", data=features,compression="gzip")
    with h5py.File('labels.h5', 'w') as hf:
        hf.create_dataset("labels", data=labels)

return_data()