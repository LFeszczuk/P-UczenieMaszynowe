from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import mglearn
import numpy as np
from matplotlib import pyplot as plt

X, y = mglearn.datasets.make_forge()
# print(X.shape,y)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# mglearn.discrete_scatter(X[:,0],X[:,1],y)
# mglearn.plots.plot_knn_classification(n_neighbors=3)
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
prediction = clf.predict(X_test)
print(prediction)
print("dokładnosc dla test: {:.2f}".format(clf.score(X_test, y_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# print(fig,axes)
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} sasiadow".format(n_neighbors))
    ax.set_xlabel("cecha 0")
    ax.set_ylabel("cecha 1")
axes[0].legend(loc=3)

plt.show()
