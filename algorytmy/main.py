from http.client import OK
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def k_neighbors(X_test, X_train, y_test, y_train, neighbor):
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    # mglearn.plots.plot_2d_separator(knn, X_train, fill=True, eps=0.5, alpha=.4)
    # mglearn.discrete_scatter(X_train[:, 3], X_train[:, 5], y_train)
    # plt.show()
    # print(len(X_test))
    # print(len(y_pred))
    print("Prognoza zestawu: {}".format(y_pred))
    print("Wynik predykcji: {:.2f}".format(np.mean(y_pred == y_test)))
    print("Wynik dla zestawu{:.2f}".format(knn.score(X_test, y_test)))


def mlp_func(X_test, X_train, y_test, y_train):
    mlp = MLPClassifier(max_iter=5000, alpha=1, random_state=42, hidden_layer_sizes=[10])
    mlp.fit(X_train, y_train)
    print("Wynik w zestawie uczącym: {:.2f}".format(mlp.score(X_train, y_train)))
    print("Wynik dla zestawu testowego{:.2f}".format(mlp.score(X_test, y_test)))
    # print("Kształt funkcjki decyzyjnej: {}".format())
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


def svm_func(X_test, X_train, y_test, y_train):
    linear_svm = LinearSVC(C=0.1, max_iter=1000)
    linear_svm.fit(X_train, y_train)
    print("kształt współczynnnika: ", linear_svm.coef_.shape)
    # print("kształt przecięcia: ", linear_svm.intercept_.shape)
    # mglearn.discrete_scatter(X_train[:, 3], X_train[:, 6], y_train)
    # line = np.linspace(-15, 15)
    # for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
    #     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    # plt.show()



    

    # print("Kształt funkcjki decyzyjnej: {}".format())
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

def hist_drawing(data):
    fig, axes = plt.subplots(7, 2, figsize=(10, 3))
    ax=axes.ravel()
    # Gesture names 'ok', 'peace', 'flat', 'bottle', 'pointing'
    ok=data.loc[data["gesture_id"]==0].to_numpy()
    peace=data.loc[data["gesture_id"]==1].to_numpy()
    flat=data.loc[data["gesture_id"]==2].to_numpy()
    bottle=data.loc[data["gesture_id"]==3].to_numpy()
    pointing=data.loc[data["gesture_id"]==4].to_numpy()
    data_n=data.to_numpy()
    print(data.columns)
    # drawing the histograms    
    for i in range(14):
        _, bins = np.histogram(data_n[:,i],bins=50)
        ax[i].hist(ok[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
        ax[i].hist(peace[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
        ax[i].hist(bottle[:,i],bins=bins,color = "skyblue", lw=0,alpha=.5)
        ax[i].hist(flat[:,i],bins=bins,color = "saddlebrown", lw=0,alpha=.5)
        ax[i].hist(pointing[:,i],bins=bins,color = "red", lw=0,alpha=.5)
        ax[i].set_title(data.columns[i])
        ax[i].set_yticks(())
    # fig.tight_layout()

    # PCA algorithm for extracting features
    pca = PCA(n_components=2)
    x = data.drop(columns=["gesture_id"]).to_numpy()
    y = data["gesture_id"].to_numpy()
    pca.fit(x)
    X_pca=pca.transform(x)
    print("x shape: {}".format(str(X_pca.shape)))

    # drawing only two features
    plt.figure(figsize=(8,8))
    mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],y)
    plt.gca().set_aspect("equal")

    plt.show()

def main():
    df = pd.read_csv('sensory_data.txt', sep='\t', skipinitialspace=True, engine='python')
    print(df.head(5))
    hist_drawing(df)
    y = df["gesture_id"]
    x = df.drop(columns=["gesture_id"])
    # x = x.to_numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    # print(X_train.head())
    # print(X_test.head())
    # grr = pd.plotting.scatter_matrix(X_train[:20], c=y_train[:20], figsize=(15, 15), marker='o',
    #                                  hist_kwds={'bins': 20},
    #                                  s=60, alpha=0.8, cmap=mglearn.cm3)
    # plt.show();

    # k_neighbors(X_test, X_train, y_test, y_train,2)
    # mlp_func(X_test, X_train, y_test, y_train)
    # svm_func(X_test, X_train, y_test, y_train)


if __name__ == "__main__":
    main()
