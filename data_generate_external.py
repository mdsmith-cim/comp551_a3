from preprocess import preprocess
from data_utils import load_data
import numpy as np


np.random.seed(984930)  # for reproducibility

trainX, trainY, testX = load_data()


print('CLEAN')

print('Processing baseline')

pp = preprocess('clean', flatten=True, normalize=True)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_clean_baseline.bin')
X_test.tofile('test_clean_baseline.bin')

print('Processing clean threshold=250')

pp = preprocess('clean', flatten=True, normalize=True, threshold=250)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_clean_thres=250.bin')
X_test.tofile('test_clean_thres=250.bin')

print('Processing clean not centered')

pp = preprocess('clean', flatten=True, normalize=True, center=False)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_clean_not_centered.bin')
X_test.tofile('test_clean_not_centered.bin')

print('Processing clean closing True')

pp = preprocess('clean', flatten=True, normalize=True, closing=True)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_clean_closing_true.bin')
X_test.tofile('test_clean_closing_true.bin')

print('Processing using SIFT: baseline')

pp = preprocess('sift', flatten=True, normalize=True)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_sift_baseline.bin')
X_test.tofile('test_sift_baseline.bin')

print('Processing SIFT threshold=250')

pp = preprocess('sift', flatten=True, normalize=True, threshold=250)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_sift_thres=250.bin')
X_test.tofile('test_sift_thres=250.bin')

print('Processing SIFT not centered')

pp = preprocess('sift', flatten=True, normalize=True, center=False)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_sift_not_centered.bin')
X_test.tofile('test_sift_not_centered.bin')

print('Processing SIFT closing True')

pp = preprocess('sift', flatten=True, normalize=True, closing=True)

X_train = pp.transform(trainX)
X_test = pp.transform(testX)

X_train.tofile('train_sift_closing_true.bin')
X_test.tofile('test_sift_closing_true.bin')
