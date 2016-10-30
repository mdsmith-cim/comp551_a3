from preprocess import preprocess
from data_utils import load_data

trainX, trainY, testX = load_data()

# Initialize preprocess class: cleaning only
pp = preprocess('clean')

trainX_clean = pp.transform(trainX)

# Initialize for SIFT
pp = preprocess('sift')

trainX_sift = pp.transform(trainX)