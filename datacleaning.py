import numpy as np
import pandas as pd
df = pd.read_excel('../data/products.xlsx')
df = df.drop(['brand','files','url','image','file_urls','shipping'],axis = 1)
df = df.dropna()
df.to_csv('../data/products_cleaned.csv', index=False)