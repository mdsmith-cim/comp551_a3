'''
This code creates, compiles and executes a calculated approximation of VGG Convolutional neural
network on the supplied data.
'''

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
tf.python.control_flow_ops = tf

import numpy
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split


img_width = 60  
img_height = 60
no_of_classes = 19

#load already extracted features. x is (n x 60 x 60) image. y is one-hot encoded labels (n x 19)
root_folder = 'features_raw'
x_train = pickle.load(open(root_folder + '/extracted_data/x_train.pkl', "rb"))
y_train_cat = pickle.load(open(root_folder + '/extracted_data/y_train_cat.pkl', "rb")) 
x_valid = pickle.load(open(root_folder + '/extracted_data/x_valid.pkl', "rb")) 
y_valid_cat = pickle.load(open(root_folder + '/extracted_data/y_valid_cat.pkl', "rb"))
                  
#build convolutional neural network                                   
model = Sequential()
model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(img_width, img_height,1)))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(64, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(no_of_classes))
model.add(Activation('softmax'))

#compile CNN
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

#fit CNN to data
checkpoint_all = ModelCheckpoint(filepath=root_folder + "/models/model.{epoch:02d}-{val_acc:.2f}.hdf5", verbose=1, save_best_only=False, save_weights_only=False, monitor='val_acc', mode='max')        
callbacks_list = [checkpoint_all]

hist=model.fit(x_train, y_train_cat, nb_epoch=100, batch_size=100, verbose=1, validation_data=(x_valid,
    y_valid_cat), callbacks=callbacks_list)

#save history of training and validation results
pickle.dump(hist.history, open(root_folder + '/results/history.pkl', "wb"))


