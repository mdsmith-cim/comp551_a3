import numpy as np
import cv2
import pandas as pd
from cv2 import xfeatures2d
from cv2 import KeyPoint
from tqdm import tqdm

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

    def get_sift_features(self, step_size=5):

        trainX, trainY, testX = self.get_clean_data()

        sft = xfeatures2d.SIFT_create()

        imgSize = trainX.shape[-2:]

        # Create keypoint list
        kp = [KeyPoint(x, y, step_size) for y in range(0, imgSize[0], step_size)
              for x in range(0, imgSize[1], step_size)]

        print("Computing SIFT features [training]")

        trainXSIFT = np.zeros((trainX.shape[0], len(kp), sft.descriptorSize()), dtype='uint8')
        testXSIFT = np.zeros((testX.shape[0], len(kp), sft.descriptorSize()), dtype='uint8')

        for i in tqdm(range(trainX.shape[0])):
            img = trainX[i]

            dsift = sft.compute(img, kp)[1]
            trainXSIFT[i] = dsift

        print("Computing SIFT features [testing]")

        for i in tqdm(range(testX.shape[0])):
            img = testX[i]

            dsift = sft.compute(img, kp)[1]
            testXSIFT[i] = dsift

        print("Done!")

        return trainXSIFT, trainY, testXSIFT


    def writeToDisk(self, data, filename='data.bin'):

        data.tofile(filename)

    def readFromDisk(self, filename, shape):

        return np.fromfile(filename, dtype='uint8').reshape(shape)
