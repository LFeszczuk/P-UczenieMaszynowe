from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import numpy as np
from matplotlib import pyplot as plt

iris_dataset = load_iris()
# print("Klucze z iris dataset \n{}".format(iris_dataset.keys()))

# print(iris_dataset['DESCR'] + "...\n")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
print(type(X_train))
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                                 alpha=0.8, cmap=mglearn.cm3)
plt.show();
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X_train, y_train)

X_new = np.array([[5, 2.9, 1, 0.2]])
print("X_new shape: {}".format(X_new.shape))
prediction = knn.predict(X_new)
print("prognoza:{}".format(prediction))
y_pred = knn.predict(X_test)
print("Prognoza zestawu: {}".format(y_pred))
print("Wynik predykcji: {:.2f}".format(np.mean(y_pred == y_test)))
print("Wynik dla zestawu{:.2f}".format(knn.score(X_test, y_test)))
