from preprocess import preprocess
from data_utils import load_data
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':

    trainX, trainY, testX = load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(trainX, trainY, test_size=0.3)

    pipe = make_pipeline(preprocess('sift', flatten=True), LogisticRegression(solver='sag', n_jobs=-1, verbose=60))

    pipe.fit(X_train, y_train)

    print(pipe.score(X_valid, y_valid))

