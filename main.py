from preprocess import preprocess

# Initialize preprocess class
pp = preprocess()

# Load from disk, threshold
trainX, trainY, testX = pp.get_clean_data()

# SIFT Example

trainSIFTX, trainY, testSIFTX = pp.get_sift_features()
