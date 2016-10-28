from preprocess import preprocess

# Initialize preprocess class
pp = preprocess()

# Load from disk, threshold
trainX, trainY, testX = pp.get_clean_data()


