from preprocess import preprocess
from data_utils import load_data


if __name__ == '__main__':

    trainX, trainY, testX = load_data()

    # Initialize preprocess class: cleaning only
    # Relevant arguments: threshold, flatten
    pp = preprocess('sift', flatten=False, center=False, threshold=254, step_size=5, closing=False, morph_size=(2, 2),
                    center_pad=2)
    trainX_clean = pp.transform(trainX)

    testX_clean = pp.transform(testX)

    pp.writeToDisk(trainX_clean, 'trainX_sift_center=False_thres=254_step=5_close=False_center_pad=2.bin')
    pp.writeToDisk(testX_clean, 'testX_sift_center=False_thres=254_step=5_close=False_center_pad=2.bin')

    # Initialize preprocess class for SIFT
    # Relevant arguments:
    # - n_jobs: specify number of threads to run.  On a weaker computer use fewer threads to avoid running out of RAM
    # - step_size: Number of pixels between keypoints.  Smaller number = more keypoints and a larger feature vector
    # - flatten: flattens SIFT vector to 1-D for use with algorithms that ignore spatial positioning like logistic
    # regression
    #pp = preprocess('sift', flatten=False, center=True, threshold=254, closing=False)

    # Transform data
    # IMPORTANT NOTE: automatic caching is in use:
    # On first run, the features are calculated
    # On subsequent runs, *if arguments are identical*, data is loaded from disk saving computation time
    # This persists even if the Python is restarted, dependant only on the cache directory not being deleted
    # The cache directory is specified as an argument to the constructor
    #trainX_SIFT = pp.transform(trainX)