import numpy as np
import cv2
import pandas as pd

class preprocess:

    def __init__(self, train_X_file='data/train_x.bin', train_y_file='data/train_y.csv', test_X_file='data/test_x.bin', threshold=254):

        self.train_X_file = train_X_file
        self.test_X_file = test_X_file
        self.train_y_file = train_y_file
        self.threshold = threshold

    def get_clean_data(self):

        trainData = np.fromfile(self.train_X_file, dtype='uint8')
        trainData = trainData.reshape((100000, 60, 60))

        print("Training data loaded from file {0}".format(self.train_X_file))

        for i in range(trainData.shape[0]):

            img = trainData[i]
            retval, thres = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)

            trainData[i] = thres

        print("Training data processed")

        testData = np.fromfile(self.test_X_file, dtype='uint8')
        testData = testData.reshape((20000, 60, 60))

        print("Test data loaded from file {0}".format(self.test_X_file))

        for i in range(testData.shape[0]):

            img = testData[i]
            retval, thres = cv2.threshold(img, self.threshold, 255, cv2.THRESH_BINARY)

            testData[i] = thres

        print("Testing data processed")

        trainLabels = pd.read_csv(self.train_y_file)['Prediction'].values

        return trainData, trainLabels, testData

    def writeToDisk(self, data, filename='data.bin'):

        data.tofile(filename)

    def readFromDisk(self, filename, shape):

        return np.fromfile(filename, dtype='uint8').reshape(shape)
