import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #important for plotting in 3D

df = pd.read_csv("../Week3/data/01-maybridge_bionet_dspl_zinc_chi_k_descriptors.csv")

# seperating out the features
X = df.iloc[:, 3:].values

# target value
y = df.loc[:,['Library']].values

# normalising the features
X = StandardScaler().fit_transform(X)


# applying PCA to the dataframe
pca = PCA(n_components = 3)
principalComponents = pca.fit_transform(X)


# creating a principal components df
pc_df = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3'])

# combining the PC dataframe and the target variable dataframe
df = pd.concat([pc_df, df[['Library']]], axis = 1)


# Visualising the data
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 component PCA', fontsize = 20)
targets = ['Maybridge', 'ZINC Lead-like subset', 'Bionet', 'DSPL']
colors = ['#32a852', '#a83232', '#5032a8', '#fffc45']
for target, color in zip(targets,colors):
    indicesToKeep = df['Library'] == target
    ax.scatter(df.loc[indicesToKeep, 'PC1']
               , df.loc[indicesToKeep, 'PC2']
               , df.loc[indicesToKeep, 'PC3']
               , c = color
               , s = 20
               , alpha = 0.7)
ax.legend(targets)
ax.grid()