from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score    # for stratified sampling and cross-validation
from sklearn import svm

import numpy as np
import pandas as pd


df = pd.read_json("../Week5/vehicle_data.json")

X = df.loc[:, df.columns != 'Col18'] 
y = df['Col18']

# generating a train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# defining a cross-validation model on linear SVM
# kernel chosen is rbf as the same kernel was chosen in the knime exercise
clf = svm.SVC(kernel = 'rbf', C = 1, random_state = 42, gamma = 0.8)
scores = cross_val_score(clf, X_train, y_train, cv = 10)

print("%0.2f accuracy with standard deviation of %0.2f" % (scores.mean(), scores.std()))