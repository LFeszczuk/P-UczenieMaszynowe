import pandas as pd
from sklearn import preprocessing


df = pd.read_csv('peace.txt', sep='  |   ', skipinitialspace=True, engine='python')
print(df.head(5))
df = df.drop(columns=df.columns[-1])
print(df.head(5))
df.columns = ["D", "SN", "WD", "KD", "KG", "KN", "SG", "MN~", "MG~", "SD", "ŚD", "MD", "WN", "WG", "ŚG", "ŚN"]

df = df.drop(columns=['MG~',"MN~"])
for headers in df.columns:
    df[headers] = df[headers].str.replace(" ", "")
df = df.astype("int")

# for headers in df.columns[12:]:
#     scaler = preprocessing.MinMaxScaler(feature_range=(0, 1023))
#     d = scaler.fit_transform(headers)
#     scaled_df = pd.DataFrame(d, headers)
#     df[headers]=scaled_df[headers]
#
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
names = df.columns
print(names)
print(df[names])
d = scaler.fit_transform(df)
scaled_df = pd.DataFrame(d, columns=names)

print(scaled_df.head(5))
print(df.head(5))
print(df.columns)
print(type(df.iloc[0, 0]))
