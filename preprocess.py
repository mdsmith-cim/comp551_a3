import numpy as np
from cv2 import xfeatures2d, KeyPoint, THRESH_BINARY
from cv2 import threshold as cvthreshold, resize, INTER_CUBIC
from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
from cv2 import morphologyEx, MORPH_CLOSE
from tqdm import tqdm
from joblib import Parallel, delayed, Memory

class preprocess:

    valid_processes = ['clean', 'sift']

    def __init__(self, process, threshold=254, step_size=5, flatten=False, center=True, closing=True,
                 center_pad=2, morph_size=(2, 2), n_jobs=-1, cache_dir="cache"):
        """
        Initializes the class.
        :param process: string
            Processing type to execute.  Possible options: 'clean' for clean data, 'sift' for SIFT features.
            Note that SIFT automatically cleans the data first, so options for cleaning also apply.
        :param threshold: integer
            Threshold to apply when cleaning data.  Applies to all process types.
        :param flatten boolean
            Whether to flatten the output matrix so that it is 2D (num examples x features)
        :param center boolean
            If true, images are cropped and resized when cleaned so that they contain only the numbers.
            In essence, this centers the numbers in the image.
        :param step_size: integer
            Step size to use only when calculating SIFT features.  Smaller step sizes mean more features.
        :param center_pad integer
            How many pixels to pad when resizing to center the numbers.  Only used if center is True.
        :param closing boolean
            Whether to apply the morphological operation closing when cleaning data.
        :param morph_size tuple of ints
            The size of the morphological operator to use when closing is True
        :param cache_dir: string
            Directory to use as a cache for features to avoid recalculation
        """
        self.threshold = threshold
        self.step_size = step_size
        self.flatten = flatten
        self.center = center
        self.center_pad = center_pad
        self.closing = closing
        self.morph_size = morph_size

        if process not in self.valid_processes:
            raise Exception('process {0} is a valid process. Valid methods are: {1}'.format(process,
                                                                                            self.valid_processes))
        self.process = process
        self.n_jobs = n_jobs

        self.memory = Memory(cachedir=cache_dir, verbose=1, compress=True)

        self._get_clean_data = self.memory.cache(self._get_clean_data)
        self._get_sift_features = self.memory.cache(self._get_sift_features, ignore=['self'])

    def fit(self, X, y=None):
        """
        Implemented for compatibility with sklearn.  Does nothing
        :param X: numpy ndarray
            X input matrix
        :param y: Ignored; for compatibility with sklearn.
        :return:  This preprocessor
        """
        return self

    def transform(self, X, y=None):
        """
        Transforms the input as specified by the process parameter in the constructor.
        :param X: numpy ndarray
            Input data matrix.
        :param y: Ignored; for compatibility with sklearn.
        :return: Transformed data matrix.
        """

        if self.process == 'clean':
            return self._get_clean_data(X, self.flatten, self.center, self.center_pad, self.closing,
                                        self.morph_size, self.threshold)

        elif self.process == 'sift':
            X_clean = self._get_clean_data(X, False, self.center, self.center_pad, self.closing,
                                           self.morph_size, self.threshold)
            return self._get_sift_features(X_clean, self.flatten, self.n_jobs, self.step_size)
        else:
            raise Exception('Invalid process {0}'.format(self.process))

    def fit_transform(self, X, y=None):
        """
        Transforms the data; a wrapper for transform to be compatible with sklearn as there is nothing to fit.
        :param X: numpy ndarray
            Input data matrix.
        :param y: Ignored; for compatibility with sklearn.
        :return: Transformed data matrix.
        """

        return self.transform(X)

    # Takes input data and thresholds to remove background noise.
    @staticmethod
    def _get_clean_data(X, flatten, center, center_pad, closing, morph_size, threshold):

        X_clean = np.zeros(X.shape, dtype='uint8')

        imgSize = X.shape[-2:]

        morph_kernel = np.ones(morph_size, dtype='uint8')

        max_uint8 = np.iinfo('uint8').max

        print("Cleaning data...")
        for i in tqdm(range(X.shape[0])):
            thres = cvthreshold(X[i], threshold, max_uint8, THRESH_BINARY)[1]

            if closing:
                thres = morphologyEx(thres, MORPH_CLOSE, morph_kernel)

            if center:
                contours = np.array(findContours(thres.copy(), RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)[1])
                contours = np.concatenate(contours).reshape((-1, 2))
                # Note: we add step_size pixels here as a buffer
                minX = np.max((np.min(contours[:, 0], axis=0) - center_pad, 0))
                maxX = np.min((np.max(contours[:, 0], axis=0) + center_pad, imgSize[1]))
                minY = np.max((np.min(contours[:, 1], axis=0) - center_pad, 0))
                maxY = np.min((np.max(contours[:, 1], axis=0) + center_pad, imgSize[0]))
                # Crop such that result is square -> not distorted
                diffX = maxX - minX
                diffY = maxY - minY
                if diffX >= diffY:
                    meanY = np.mean((minY, maxY)).astype('int64')
                    centerX = (diffX/2).astype('int64')
                    cropped = thres[np.max((meanY-centerX, 0)):np.min((meanY+centerX, imgSize[0])), minX:maxX]
                else:
                    meanX = np.mean((minX, maxX)).astype('int64')
                    centerY = (diffY/2).astype('int64')
                    cropped = thres[minY:maxY, np.max((meanX-centerY, 0)):np.min((meanX+centerY, imgSize[1]))]

                # if cropped.shape[0] == 0:
                #     print("Cropped shape: {0}".format(cropped.shape))
                #     print("minX,maxX,minY,maxY: {0},{1},{2},{3}".format(minX,maxX, minY, maxY))
                #     print("contours: {0}".format(contours))
                #     import matplotlib.pyplot as plt
                #     plt.imshow(thres, cmap='gray'); plt.show()

                thres = resize(cropped, imgSize, interpolation=INTER_CUBIC)


            X_clean[i] = thres

        print("Data cleaned")
        if flatten:
            X_clean = X_clean.reshape((X.shape[0], -1))

        return X_clean


    # Calculates SIFT features.  Uses threads to speed up computation.
    def _get_sift_features(self, X, flatten, n_jobs, step_size):

        numImg = X.shape[0]

        print("Computing SIFT features...")

        result = np.array(Parallel(n_jobs=n_jobs)(delayed(self.sift_compute)(X[i], step_size)
                                                  for i in tqdm(range(numImg))))
        if flatten:
            result = result.reshape((X.shape[0], -1))

        return result

    # Inefficient wrapper for SIFT calculation to get around pickle limitations on OpenCV objects
    @staticmethod
    def sift_compute(img, step_size):
        kps = [KeyPoint(x, y, step_size) for y in range(0, img.shape[0], step_size)
               for x in range(0, img.shape[1], step_size)]
        return xfeatures2d.SIFT_create().compute(img, kps)[1].astype('uint8')


    def writeToDisk(self, data, filename='data.bin'):
        """
        Writes numpy array to disk.
        :param data: Array to be written.
        :param filename: String specifying filename.
        :return: Nothing.
        """
        data.tofile(filename)

    def readFromDisk(self, filename, shape):
        """
        Reads numpy array from disk.
        :param filename: String specifying filename.
        :param shape: Shape data is stored in.
        :return: Data from disk.
        """
        return np.fromfile(filename, dtype='uint8').reshape(shape)

    # Crude imitation of sklearn's string representation functionality
    def __repr__(self):
        return 'preprocess' + str({'process': self.process, 'threshold': self.threshold, 'step_size': self.step_size,
                                   'n_jobs': self.n_jobs, 'caching': self.memory, 'center': self.center,
                                   'flatten': self.flatten, 'center_pad': self.center_pad})