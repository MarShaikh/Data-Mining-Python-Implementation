from sklearn import tree
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

