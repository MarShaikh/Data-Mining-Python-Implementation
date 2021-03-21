from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    # for stratified sampling
from sklearn.datasets import load_iris                  # to load data instead of getting local data

import numpy as np
import pandas as pd


X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# making the linear regression model
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)


# Metrics and scoring, for the linear regression model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

print(r2_score(y_test, y_pred, multioutput="uniform_average"))
print(mean_squared_error(y_test, y_pred))
