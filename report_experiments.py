from preprocess import preprocess
from data_utils import load_data
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from joblib import dump
from pickle import HIGHEST_PROTOCOL

np.random.seed(984930)  # for reproducibility

trainX, trainY, testX = load_data()

X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.3, random_state=984930)

batch_size = 1000
nb_classes = np.unique(y_train).size
nb_epoch = 20

# input image dimensions
img_rows = trainX[0].shape[0]
img_cols = trainX[0].shape[1]

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_valid = np_utils.to_categorical(y_valid, nb_classes)

# Use MLP
model = Sequential()
model.add(Dense(1024, input_shape=(img_rows*img_cols,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(), metrics=['accuracy'])


print('CLEAN')

print('Processing baseline')

pp = preprocess('clean', flatten=True, normalize=True)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

# Clean data - baseline

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

# params = [dict(preprocess__threshold=[242, 244, 246, 248, 250, 252, 254],
#                preprocess__step_size=[3, 5, 7, 9], preprocess__center=[False, True], preprocess__closing=[True],
#                preprocess__center_pad=[2, 4, 6, 8]),
#           dict(preprocess__threshold=[242, 244, 246, 248, 250, 252, 254], preprocess__step_size=[3, 5, 7, 9],
#                preprocess__center=[False, True], preprocess__closing=[False], preprocess__center_pad=[2, 4, 6, 8],
#                preprocess__morph_size=[(2, 2), (3, 3), (5, 5), (7, 7)])]

dump(history.history, 'hist_mlp_baseline_clean.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing clean threshold=250')

pp = preprocess('clean', flatten=True, normalize=True, threshold=250)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_clean_thres=250.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing clean not centered')

pp = preprocess('clean', flatten=True, normalize=True, center=False)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_clean_center=False.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing clean closing True')

pp = preprocess('clean', flatten=True, normalize=True, closing=True)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_clean_closing=True.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing using SIFT: baseline')

model = Sequential()
model.add(Dense(1024, input_shape=(144*128,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(), metrics=['accuracy'])

pp = preprocess('sift', flatten=True, normalize=True)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_baseline_SIFT.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing SIFT threshold=250')

pp = preprocess('sift', flatten=True, normalize=True, threshold=250)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_SIFT_thres=250.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing SIFT not centered')

pp = preprocess('sift', flatten=True, normalize=True, center=False)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_SIFT_center=False.pkl', protocol=HIGHEST_PROTOCOL)

print('Processing SIFT closing True')

pp = preprocess('sift', flatten=True, normalize=True, closing=True)

X_t = pp.transform(X_train)
X_v = pp.transform(X_valid)

history = model.fit(X_t, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                    validation_data=(X_v, Y_valid))

dump(history.history, 'hist_mlp_sift_closing=True.pkl', protocol=HIGHEST_PROTOCOL)

