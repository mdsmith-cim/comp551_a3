from preprocess import preprocess
from data_utils import load_data
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import joblib

trainX, trainY, testX = load_data()

# --------- Clean data
pipe = make_pipeline(preprocess('clean', flatten=True, center=True, closing=False, normalize=True),
                     LogisticRegression(solver='sag', n_jobs=-1, verbose=60))

params = dict(logisticregression__C=[0.01, 0.1, 1, 10, 100])

grid = GridSearchCV(pipe, n_jobs=-1, verbose=50, param_grid=params)

grid.fit(trainX, trainY)

print("Results with clean data:\n{0}".format(grid.cv_results_))

joblib.dump(grid, "log_reg_CV_clean.pk")

# -------- SIFT

# Reload data just in case through a bug original data is modified
trainX, trainY, testX = load_data()

pipe = make_pipeline(preprocess('sift', flatten=True, center=True, closing=False, normalize=True),
                     LogisticRegression(solver='sag', n_jobs=-1, verbose=60))

params = dict(logisticregression__C=[0.01, 0.1, 1, 10, 100])

grid = GridSearchCV(pipe, n_jobs=-1, verbose=50, param_grid=params)

grid.fit(trainX, trainY)

print("Results with SIFT data:\n{0}".format(grid.cv_results_))

joblib.dump(grid, "log_reg_CV_sift.pk")
