import numpy as np
import pandas as pd


def load_data(train_X_file='data/train_x.bin', train_y_file='data/train_y.csv', test_X_file='data/test_x.bin'):
    """
    Loads data from disk.
    :param train_X_file: string
        Path to training data matrix X, in binary format (x86-64 compatible)
    :param train_y_file: string
        Path to training data vector y, in csv format.
    :param test_X_file: string
        Path to testing data matrix X, in binary format (x86-64 compatible)
    :return: X (train), y (train), X (test)
    """
    trainData = np.fromfile(train_X_file, dtype='uint8')
    trainData = trainData.reshape((100000, 60, 60))

    print("Training data loaded from file {0}".format(train_X_file))

    testData = np.fromfile(test_X_file, dtype='uint8')
    testData = testData.reshape((20000, 60, 60))

    print("Test data loaded from file {0}".format(test_X_file))

    trainLabels = pd.read_csv(train_y_file)['Prediction'].values

    return trainData, trainLabels, testData