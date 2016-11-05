import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

img_width= 128
img_height = 144
no_of_classes = 19

x_train = np.fromfile('../../data/train_x_sift.bin', dtype='uint8')                              
x_train = x_train.reshape(100000, img_width, img_height, 1)

y_train = pd.read_csv('../../data/train_y.csv')
y_train = y_train['Prediction'].as_matrix()

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.25,
                random_state=42)

y_train_cat = to_categorical(y_train, no_of_classes)
y_valid_cat = to_categorical(y_valid, no_of_classes)

#save training and validation data.
pickle.dump(x_train, open('features_sift/extracted_data/x_train.pkl', "wb"))
pickle.dump(y_train_cat, open('features_sift/extracted_data/y_train_cat.pkl', "wb"))
pickle.dump(x_valid, open('features_sift/extracted_data/x_valid.pkl', "wb"))
pickle.dump(y_valid_cat, open('features_sift/extracted_data/y_valid_cat.pkl', "wb"))

