from sklearn import tree
from sklearn.model_selection import train_test_split    # for stratified sampling
from sklearn.metrics import confusion_matrix

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

calls_df = pd.read_csv("../Week4/Calls.csv")
contract_df = pd.read_csv("../Week4/ContractData.csv")

# joining the two dataframes

df = calls_df.join(contract_df.set_index(['Area Code', 'Phone']), on = ['Area Code', 'Phone'])


# plotting a correlation matrix to remove highly corr. features

# plt.figure(figsize=(19, 15))
# sns.heatmap(df.corr(), annot=True)
# plt.show()

# features: Day Charge, Eve Charge, Night Charge, Intl Charge can be removed
# removing other features like State, Area Code, and Phone as categorical variables are not supported by 
# sci-kit learn's tree implementation (as per the documentation)

X = df.drop(['Churn', 'Day Charge', 'Night Charge', 'Intl Charge', 'State', 'Area Code', 'Phone'], axis = 1)
y = df[['Churn']]

# creating a train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# creating the tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# score of the tree predictor
print(clf.score(X_test, y_test))

# creating confusion matrix
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)
