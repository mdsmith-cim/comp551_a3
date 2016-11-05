from keras.models import load_model
import pandas as pd
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf

model = load_model('features_raw/models/model.62-0.98.hdf5')

img_dim = 60
no_of_classes = 19

x_test = np.fromfile('../../data/test_x.bin', dtype='uint8')
x_test = x_test.reshape(20000, img_dim, img_dim, 1)

y_test_pred_cat = model.predict(x_test)
y_test_pred = y_test_pred_cat.argmax(1)

mydict = {'Id': np.arange(0,len(y_test_pred)), 'Prediction': y_test_pred}
y_test_pred_df = pd.DataFrame(mydict, columns=['Id', 'Prediction'])
y_test_pred_df.to_csv('features_raw/results/vgg_predictions.csv', index=False)
