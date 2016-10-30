import numpy as np
import cv2
import pandas as pd
from cv2 import xfeatures2d
from cv2 import KeyPoint
from tqdm import tqdm

class preprocess:

    valid_processes = ['clean', 'sift']

    def __init__(self, process, threshold=254, step_size=5):
        """
        Initializes the class.
        :param process: string
            Processing type to execute.  Possible options: 'clean' for clean data, 'sift' for SIFT features
        :param threshold: integer
            Threshold to apply when cleaning data.  Applies to all process types.
        :param step_size: integer
            Step size to use only when calculating SIFT features.  Smaller step sizes mean more features.
        """
        self.threshold = threshold
        self.step_size = step_size

        if process not in self.valid_processes:
            raise Exception('process {0} is a valid process. Valid methods are: {1}'.format(process,
                                                                                            self.valid_processes))
        self.process = process
        self.X_transform = None

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        if self.process == 'clean':
            return self._get_clean_data(X)

        elif self.process == 'sift':
            X_clean = self._get_clean_data(X)
            return self._get_sift_features(X_clean)
        else:
            raise Exception('Invalid process {0}'.format(self.process))

    def fit_transform(self, X, y=None):

        return self.transform(X)

    def _get_clean_data(self, X):

        X_clean = np.zeros(X.shape, dtype='uint8')

        max_uint8 = np.iinfo('uint8').max

        print("Cleaning data...")
        for i in range(X.shape[0]):
            retval, thres = cv2.threshold(X[i], self.threshold, max_uint8, cv2.THRESH_BINARY)
            X_clean[i] = thres

        print("Data cleaned")
        return X_clean

    def _get_sift_features(self, X):

        sft = xfeatures2d.SIFT_create()

        imgSize = X.shape[-2:]
        numImg = X.shape[0]

        # Create keypoint list
        kp = [KeyPoint(x, y, self.step_size) for y in range(0, imgSize[0], self.step_size)
              for x in range(0, imgSize[1], self.step_size)]

        print("Computing SIFT features...")

        XSIFT = np.zeros((numImg, len(kp), sft.descriptorSize()))

        for i in tqdm(range(numImg)):
            img = X[i]

            XSIFT[i] = sft.compute(img, kp)[1]

        return XSIFT

    def writeToDisk(self, data, filename='data.bin'):

        data.tofile(filename)

    def readFromDisk(self, filename, shape):

        return np.fromfile(filename, dtype='uint8').reshape(shape)

    # Crude imitation of sklearn's string representation functionality
    def __repr__(self):
        return 'preprocess' + str({'process': self.process, 'threshold': self.threshold, 'step_size': self.step_size})

