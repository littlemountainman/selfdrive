from __future__ import division
from keras.utils import print_summary
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from models.experimental_model import experimental
from models.cnn_model import cnn_stock_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam
def load_numpy():
    features = np.load("features.npy")
    labels = np.load("labels.npy")
    return features, labels

# TODO: Function to read tfrecord
def read_tf_record():
    pass

def model_deploy():
    features, labels = load_numpy()
    features, labels = shuffle(features, labels)
    print(labels.shape)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                            test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 100, 100, 1)
    test_x = test_x.reshape(test_x.shape[0], 100, 100, 1)
    model, callbacks_list = experimental(100, 100)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(optimizer=Adam(lr=0.001), loss="mse", metrics=["accuracy"])
    parallel_model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=100, batch_size=100,
               callbacks=callbacks_list)
    #model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=50, batch_size=16,
     #           callbacks=callbacks_list)
    print_summary(model)
    
    model.save('Autopilot_10.h5')


if __name__ == '__main__':
    model_deploy()




