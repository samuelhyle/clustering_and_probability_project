from re import findall
from time import time

import numpy as np
import numpy.testing as np_testing
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_squared_error, confusion_matrix as calculate_confusion_matrix
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
from seaborn import heatmap

def main():

    X_1 = pd.read_csv('chooseK.csv').to_numpy()

    plt.scatter(X_1[:, 0], X_1[:, 1])
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    inertia = []
    SI = []

    for k in range(2, 8):

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X_1)

        inertia.append(kmeans.inertia_)

        si = metrics.silhouette_score(X_1, kmeans.labels_)

        SI.append(si)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(np.arange(2, 8), inertia)
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia')
        plt.xticks(np.arange(2, 8))
        plt.subplot(1, 2, 2)
        plt.plot(np.arange(2, 8), SI)
        plt.xlabel('Number of clusters')
        plt.ylabel('SI')
        plt.xticks(np.arange(2, 8))
        plt.show()

        km = KMeans(n_clusters=Best_k, n_init=10).fit(X_1)
        plt.scatter(X_1[:, 0], X_1[:, 1], c=km.labels_)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

        X_2 = pd.read_csv('chooseK.csv').to_numpy()

        xx, yy = np.meshgrid(np.linspace(-2, 3.0, 100), np.linspace(-2.5, 4, 100))
        xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
        XX = np.hstack((xx, yy))

        np.random.seed(1)
        it = 0
        inertia_prev = 1e12
        init = X_2[np.random.choice(np.arange(X_2.shape[0]), Best_k, replace=False), :]

        kmeans = KMeans(n_clusters=Best_k, init=init, n_init=1, max_iter=1).fit(X_2)
        init = kmeans.cluster_centers_
        it += 1

        if (np.isclose(kmeans.inertia_, inertia_prev, rtol=1e-3)):
            print('Stop Iteration at')
            print('{}th step'.format(it))
        else:
            inertia_prev = kmeans.inertia_

        plt.figure(figsize=(9, 7))
        plt.plot(init[:, 0], init[:, 1], 'ro', markersize=7)
        plt.scatter(xx, yy, c=kmeans.predict(XX), alpha=.8)
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.predict(X), edgecolors='black')
        plt.title('Red dots are current centroids, and other colors refer to cluster assignments')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.show()

main()