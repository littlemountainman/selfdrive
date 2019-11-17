from __future__ import division
from keras.utils import print_summary
import numpy as np
import keras 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from models.experimental_model import experimental
from models.cnn_model import cnn_stock_model
from keras.utils import multi_gpu_model
from keras.optimizers import Adam, Nadam, SGD
from keras.losses import binary_crossentropy, categorical_hinge
import keras.backend as K
import sys

def load_numpy(f,l):
    features = np.load(f).astype('uint8')
    labels = np.load(l)
    print(labels.shape)
    print(labels[::100])
    labels = np.load(l)[::6]
    print(labels.shape)
    print(features.shape)
    labels = labels[:950]
    features = features[:950]
    
    return features, labels


def exp_deploy(features,labels):
    
    #features, labels = shuffle(features, labels)
    print(labels.shape)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, random_state=0,
                                                            test_size=0.1)
    train_x = train_x.reshape(train_x.shape[0], 640, 480,3)
    test_x = test_x.reshape(test_x.shape[0], 640, 480,3)
    tbCallBack = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1, batch_size=8,
               callbacks=[tbCallBack], verbose=1)
    #model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=, batch_size=8,
    #           callbacks=callbacks_list)
    print_summary(model)
    # Model gets saved after each dataset. It still improves. GG

    model.save("final.h5")

def det_coeff(y_true, y_pred):
    u = K.sum(K.square(y_true - y_pred))
    v = K.sum(K.square(y_true - K.mean(y_true)))
    return K.ones_like(v) - (u / v)
def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

if __name__ == '__main__':
    ## Training numpy files 
    filepath = sys.argv[1]
    opt = Adam(lr=1e-4)
    model, callbacks_list = experimental(640, 480)
    model.compile(optimizer=opt, loss="mse", metrics=["accuracy", det_coeff])
    i = 1
    j = 0
    features_list = []
    labels_list = []
    f_list = []
    l_list = []
    ## SET LIMIT HERE = number of files in folder
    LIMIT = None 
while j < 200:
        if LIMIT is not None:
            if i == LIMIT:
                i= 1
        label_optimizer = np.load(filepath+"/labels"+str(i)+".npy")
        
        if len(label_optimizer) >  250 :
            if label_optimizer[0][0] != label_optimizer[1000][0]:
                print(str(i))
                features_list.append(filepath+"/camera"+str(i)+".npy")
                labels_list.append(filepath+"/labels"+str(i)+".npy")
                print(len(features_list))
                ## Increase this number if you want to. Currently it's for 8GB of RAM. I had 64GB while testing I used 10.
                if len(features_list) == 1:
                    print("Jumped here")
                    for a in features_list:
                        f_list.append(np.load(a)[:950])
                    for b in labels_list:
                        l_list.append(np.load(b)[::6][:950])
                    features = np.concatenate(f_list)
                    labels = np.concatenate(l_list)
                    exp_deploy(features, labels)
                    features_list = []
                    labels_list = []
                    f_list = []
                    l_list = []
                    i = i +1
                    print("Done loading data")
                    ## Remove break if you want bigger data
                    break
                i = i+1
            else:
                i = i+1
                print("Data sucks")
                continue
        else:
            print("Data sucks")
            i = i +1 
            continue
        
            
