from preprocess import preprocess
from data_utils import load_data


if __name__ == '__main__':

    trainX, trainY, testX = load_data()

    # Initialize preprocess class: cleaning only
    # Relevant arguments: threshold, flatten
    #pp = preprocess('clean')
    #trainX_clean = pp.transform(trainX)

    # Initialize preprocess class for SIFT
    # Relevant arguments:
    # - n_jobs: specify number of threads to run.  On a weaker computer use fewer threads to avoid running out of RAM
    # - step_size: Number of pixels between keypoints.  Smaller number = more keypoints and a larger feature vector
    # - flatten: flattens SIFT vector to 1-D for use with algorithms that ignore spatial positioning like logistic
    # regression
    pp = preprocess('sift', flatten=True)

    # Transform data
    # IMPORTANT NOTE: automatic caching is in use:
    # On first run, the features are calculated
    # On subsequent runs, *if arguments are identical*, data is loaded from disk saving computation time
    # This persists even if the Python is restarted, dependant only on the cache directory not being deleted
    # The cache directory is specified as an argument to the constructor
    trainX_SIFT = pp.transform(trainX)