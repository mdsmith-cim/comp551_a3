from preprocess import preprocess
from data_utils import load_data
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    trainX, trainY, testX = load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.3, random_state=984930)

    pipe = make_pipeline(preprocess('clean', flatten=True), PCA(n_components=80),
                         SVC(kernel='poly', degree=2, cache_size=20000, verbose=True, decision_function_shape='ovr'))

    pipe.fit(X_train, y_train)

    print("Validation score: {0}".format(pipe.score(X_valid, y_valid)))

