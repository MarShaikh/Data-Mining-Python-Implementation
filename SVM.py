from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    # for stratified sampling

import numpy as np
import pandas as pd


df = pd.read_json("../Week5/vehicle_data.json")

X = df.loc[:, df.columns != 'Col18'] 
y = df['Col18']


# K-fold cross-validation to train and test the model
