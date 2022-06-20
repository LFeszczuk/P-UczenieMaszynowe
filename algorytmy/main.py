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
from sklearn.model_selection import ShuffleSplit

from learn import plot_learning_curve
# plt.style.use('seaborn')
global X_pca

def k_neighbors(x, y, neighbor):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_acc=[]
    test_acc=[]
    neighbors=range(1,11)
    plt.figure(figsize=(8,8))
    # fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for i in neighbors:
        clf = KNeighborsClassifier(n_neighbors=i).fit(X_train, y_train)
        train_acc.append(clf.score(X_train,y_train))
        test_acc.append(clf.score(X_test,y_test))
    plt.plot(neighbors,train_acc,label="dokładność w danych uczacych")
    plt.plot(neighbors,test_acc,label="dokładność w danych testowych")
    plt.xlabel("n_neighbors")
    plt.ylabel("Dokładność")
    plt.legend()
    # knn = KNeighborsClassifier(n_neighbors=neighbor)
    # knn.fit(X_train, y_train)
    # y_pred = knn.predict(X_test)
    # mglearn.plots.plot_2d_separator(knn, X_train, fill=True, eps=0.5, alpha=.4)
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    # print("Prognoza zestawu: {}".format(y_pred))
    # print("Wynik predykcji: {:.2f}".format(np.mean(y_pred == y_test)))
    # print("Wynik dla zestawu uczacego knn: {:.2f} i testowego {:.2f}".format(knn.score(X_train, y_train),knn.score(X_test, y_test)))
    # print("Wynik dla zestawu testowego knn: {:.2f}".format(knn.score(X_test, y_test)))


def mlp_func(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    mlp = MLPClassifier(max_iter=5000, alpha=1, random_state=42, hidden_layer_sizes=[10])
    mlp.fit(X_train, y_train)
    print("Wynik w zestawie uczącym mlp: {:.2f}".format(mlp.score(X_train, y_train)))
    print("Wynik dla zestawu testowego mlp: {:.2f}".format(mlp.score(X_test, y_test)))
    # print("Kształt funkcjki decyzyjnej: {}".format())
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


def svm_func(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    linear_svm = LinearSVC(C=10, max_iter=1000)
    linear_svm.fit(X_train, y_train)
    print("kształt współczynnnika: ", linear_svm.coef_.shape)
    print("kształt przecięcia: ", linear_svm.intercept_.shape)
    # mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    line = np.linspace(-2, 3)
    plt.figure(figsize=(8,8))
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ['b', 'r', 'g']):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)

    print("Wynik w zestawie uczącym svm: {:.2f}".format(linear_svm.score(X_train, y_train)))
    print("Wynik dla zestawu testowego svm: {:.2f}".format(linear_svm.score(X_test, y_test)))

def hist_drawing(data):
    global X_pca

    # Gesture names 'ok', 'peace', 'flat', 'bottle', 'pointing'
    ok=data.loc[data["gesture_id"]==0].to_numpy()
    peace=data.loc[data["gesture_id"]==1].to_numpy()
    flat=data.loc[data["gesture_id"]==2].to_numpy()
    bottle=data.loc[data["gesture_id"]==3].to_numpy()
    pointing=data.loc[data["gesture_id"]==4].to_numpy()
    data_n=data.to_numpy()

    # drawing the histograms    
    # fig, axes = plt.subplots(7, 2, figsize=(10, 3))
    # ax=axes.ravel()
    # for i in range(14):
    #     _, bins = np.histogram(data_n[:,i],bins=50)
    #     ax[i].hist(ok[:,i],bins=bins,color=mglearn.cm3(0),alpha=.5)
    #     ax[i].hist(peace[:,i],bins=bins,color=mglearn.cm3(2),alpha=.5)
    #     ax[i].hist(bottle[:,i],bins=bins,color = "skyblue", lw=0,alpha=.5)
    #     ax[i].hist(flat[:,i],bins=bins,color = "saddlebrown", lw=0,alpha=.5)
    #     ax[i].hist(pointing[:,i],bins=bins,color = "red", lw=0,alpha=.5)
    #     ax[i].set_title(data.columns[i])
    #     ax[i].set_yticks(())
    
    # ax[0].set_xlabel("Znaczenie cechy")
    # ax[0].set_ylabel("Częstotliwość")
    # ax[0].legend(["ok","peace","flat","bottle","pointing"],loc="upper right")
    # fig.tight_layout()
    
    # PCA algorithm for extracting features
    pca = PCA(n_components=6)
    x = data.drop(columns=["gesture_id"])
    names=np.array(x.columns)
    class_names=("ok","peace","flat","bottle","pointing")
    x=x.to_numpy()
    y = data["gesture_id"].to_numpy()
    pca.fit(x)
    X_pca=pca.transform(x)
    # print("x shape: {}".format(str(X_pca.shape)))

    # # drawing only two features
    # plt.figure(figsize=(8,8))
    # mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],y)
    # plt.gca().set_aspect("equal")
    # plt.legend(class_names,loc="best")
    # plt.xlabel("Pierwszy główny komponent")
    # plt.ylabel("Drugi główny komponent")

    # # drawing the map grid of wages
    # plt.matshow(pca.components_,cmap='viridis')
    # plt.yticks([0,1],["Pierwszy komponent","Drugi kopmponent"])
    # plt.colorbar()
    # plt.xticks(range(len(names)), names, rotation=60, ha="left")
    # plt.xlabel("cecha")
    # plt.ylabel("komponenty główne")
def draw_learn(x,y):
    fig, axes = plt.subplots(3, 3, figsize=(10, 15))

    ##################    SVC     ##################
    title = r"Learning Curves (SVC, c=5)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    estimator = LinearSVC( max_iter=2000)
    plot_learning_curve(
        estimator, title, x, y, axes=axes[:, 0], ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )

    ##################    KNeighbors     ##################
    title = r"Learning Curves (KNeighbors, n=5)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)
    estimator =  KNeighborsClassifier(n_neighbors=5)
    plot_learning_curve(
        estimator, title, x, y, axes=axes[:, 1], ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )

    ##################    mlp     ##################
    title = r"Learning Curves (mlp, hidden layers = 10)"
    # SVC is more expensive so we do a lower number of CV iterations:
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    estimator = MLPClassifier(activation='relu',max_iter=5000, alpha=1, random_state=42, hidden_layer_sizes=[10,10])
    plot_learning_curve(
        estimator, title, x, y, axes=axes[:, 2], ylim=(0.7, 1.01), cv=cv, n_jobs=4
        )
def main():
    global X_pca

    df = pd.read_csv('sensory_data.txt', sep='\t', skipinitialspace=True, engine='python')
    hist_drawing(df)
    y = df["gesture_id"]
    x = df.drop(columns=["gesture_id"])
    x = x.to_numpy()
    # draw_learn(x,y)
    # X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
    # grr = pd.plotting.scatter_matrix(X_train[:20], c=y_train[:20], figsize=(15, 15), marker='o',
    #                                  hist_kwds={'bins': 20},
    #                                  s=60, alpha=0.8, cmap=mglearn.cm3)

    # for i in [1,3,9]:
    #     print("################# Wynik dla {} sąsiada".format(i))
    #     k_neighbors(x,y,i)
    #     k_neighbors(X_pca,y,i)
    # mlp_func(x,y)
    # mlp_func(X_pca,y)
    # svm_func(x,y)
    svm_func(X_pca,y)




    
    plt.show()
if __name__ == "__main__":
    main()
