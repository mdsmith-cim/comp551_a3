import numpy as np
from cv2 import xfeatures2d, KeyPoint, THRESH_BINARY
from cv2 import threshold as cvthreshold
from tqdm import tqdm
from joblib import Parallel, delayed, Memory

class preprocess:

    valid_processes = ['clean', 'sift']

    def __init__(self, process, threshold=254, step_size=5, n_jobs=-1, cache_dir="cache"):
        """
        Initializes the class.
        :param process: string
            Processing type to execute.  Possible options: 'clean' for clean data, 'sift' for SIFT features
        :param threshold: integer
            Threshold to apply when cleaning data.  Applies to all process types.
        :param step_size: integer
            Step size to use only when calculating SIFT features.  Smaller step sizes mean more features.
        :param cache_dir: string
            Directory to use as a cache for features to avoid recalculation
        """
        self.threshold = threshold
        self.step_size = step_size

        if process not in self.valid_processes:
            raise Exception('process {0} is a valid process. Valid methods are: {1}'.format(process,
                                                                                            self.valid_processes))
        self.process = process
        self.n_jobs = n_jobs

        self.memory = Memory(cachedir=cache_dir, verbose=1, compress=True)

        self.transform = self.memory.cache(self.transform)

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
            return self._get_clean_data(X)

        elif self.process == 'sift':
            X_clean = self._get_clean_data(X)
            return self._get_sift_features(X_clean)
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
    def _get_clean_data(self, X):

        X_clean = np.zeros(X.shape, dtype='uint8')

        max_uint8 = np.iinfo('uint8').max

        print("Cleaning data...")
        for i in range(X.shape[0]):
            retval, thres = cvthreshold(X[i], self.threshold, max_uint8, THRESH_BINARY)
            X_clean[i] = thres

        print("Data cleaned")
        return X_clean

    # Calculates SIFT features.  Uses threads to speed up computation.
    def _get_sift_features(self, X):

        numImg = X.shape[0]

        print("Computing SIFT features...")

        result = np.array(Parallel(n_jobs=self.n_jobs)(delayed(self.sift_compute)(X[i], self.step_size)
                                                       for i in tqdm(range(numImg))))

        return result

    # Inefficient wrapper for SIFT calculation to get around pickle limitations on OpenCV objects
    def sift_compute(self, img, step_size):
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
                                   'n_jobs': self.n_jobs, 'caching': self.memory})

