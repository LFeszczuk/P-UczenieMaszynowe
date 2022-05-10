import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('sensory_data.txt', sep='\t', skipinitialspace=True, engine='python')
print(df.head(5))