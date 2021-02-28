from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt


df = pd.read_csv("../Week3/data/Online Retail Germany.csv")

df.head(5)

# converting the Invoice date column to datetime format
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
 
# creating a new column `Recency` to calculate how recent a customer purchased
# from our shop

df['Recency'] = pd.Timestamp.now().normalize() - df['InvoiceDate']

#converting to number to normalize using Z-Score
df['Recency'].astype('int')

# only extracting number of days as an integer
df['Recency'] = df['Recency'].dt.days

# grouping data by customerId and aggregating over the total purchase, quantity, and how recent 
# they made a purchase from the store
groupedData = df.groupby('CustomerID').agg({'UnitPrice' : 'sum',
                            'Quantity': 'sum',
                            'Recency' : 'min'})


# normalizing the data
groupedData[['UnitPrice', 'Quantity', 'Recency']] = StandardScaler().fit_transform(groupedData[['UnitPrice', 'Quantity', 'Recency']]) 



# initialising the KMeans clustering class
kmeans = KMeans(init = "random",
                n_clusters = 3,
                n_init = 10,
                max_iter = 99,
                random_state = 42)

kmeans.fit(groupedData)

groupedData['cl'] = kmeans.labels_
groupedData.plot.scatter('Quantity', 'Recency', c = 'cl')
