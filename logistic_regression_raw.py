from data_utils import load_data
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

trainX, trainY, testX = load_data()

trainX = trainX.reshape((-1, trainX.shape[1]*trainX.shape[2]))

clf = LogisticRegression(solver='sag', n_jobs=-1, verbose=60)

params = dict(C=[0.01, 0.1, 1, 10, 100])

grid = GridSearchCV(clf, n_jobs=-1, verbose=50, param_grid=params)

grid.fit(trainX, trainY)

print("Results with raw data:\n{0}".format(grid.cv_results_))

joblib.dump(grid, "log_reg_CV_raw_data.pk")
