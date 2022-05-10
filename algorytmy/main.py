import pandas as pd
from sklearn import preprocessing


# Zebrać dane w zbiory, potem zebrac je w jeden zbiór z oopisanymi gestami 1,2,3,4,5
# podzielić na zbiory uczęce i testow fit itd.
def data_mod(file_name, gesture_id):
    df = pd.read_csv('{}.txt'.format(file_name), sep='  |   ', skipinitialspace=True, engine='python')
    print(df.head(5))
    df = df.drop(columns=df.columns[-1])
    print(df.head(5))
    df.columns = ["D", "SN", "WD", "KD", "KG", "KN", "SG", "MN~", "MG~", "SD", "ŚD", "MD", "WN", "WG", "ŚG", "ŚN"]
    df = df.drop(columns=['MG~', "MN~"])
    for headers in df.columns:
        df[headers] = df[headers].str.replace(" ", "")
    df = df.astype("int")
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    names = df.columns
    print(names)
    print(df[names])
    d = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(d, columns=names)
    scaled_df["gesture_id"] = gesture_id
    print(scaled_df.head(5))
    print(df.head(5))
    scaled_df.to_csv("{}_mod.txt".format(file_name), index=False, sep='\t')
    # print(df.columns)
    # print(type(df.iloc[0, 0]))

file_names=['ok','peace','flat','bottle','pointing']
for index,name in enumerate(file_names):
    data_mod(name,index)