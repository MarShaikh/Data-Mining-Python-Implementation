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

