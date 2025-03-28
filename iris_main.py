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

    iris_data = load_iris()
    X = iris_data.data
    y = iris_data.target

    print(f'Sample shapes {X.shape}')
    print(f'Target names {iris_data.target_names}')

    kmeans = KMeans(n_clusters=3, random_state=0, n_init="auto")
    kmeans.fit(X)

    t0 = time()
    estimator = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)
    fit_time = time() - t0
    results = [fit_time, estimator.inertia_]

    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(y, estimator.labels_) for m in clustering_metrics]

    results += [
        metrics.silhouette_score(
            X,
            estimator.labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    formatter_result = (
        "{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(66 * "_")
    print("time\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")
    print(formatter_result.format(*results))
    print(66 * "_")

    labels = kmeans.labels_
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(float),
               edgecolor='k')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    plt.show()

    np.random.seed(0)
    n = 500

    Xdemo = np.random.multivariate_normal([0, 5 / 2], np.diag([20, 5 / 8]), n)
    Xdemo = np.vstack((Xdemo, np.random.multivariate_normal([0, -5 / 2], np.diag([20, 5 / 8]), n)))

    init = Xdemo[np.random.choice(np.arange(Xdemo.shape[0]), 2, replace=False), :]

    kmeans = KMeans(n_clusters=2, init=init, n_init=1).fit(Xdemo)

    plt.scatter(Xdemo[:, 0], Xdemo[:, 1], c=kmeans.labels_)
    plt.plot(init[:, 0], init[:, 1], 'bo', markersize=7)
    plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'ro', markersize=7)
    plt.xlim([-15, 15])
    plt.ylim([-6, 6])
    plt.title('Blue dots initial, Red dots final centroids')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

    Xs = StandardScaler().fit_transform(Xdemo)

    init = Xs[np.random.choice(np.arange(Xs.shape[0]), 2, replace=False), :]
    kmeans = KMeans(n_clusters=2, init=init, n_init=1).fit(Xs)

    plt.scatter(Xs[:, 0], Xs[:, 1], c=kmeans.labels_)
    plt.plot(init[:, 0], init[:, 1], 'bo', markersize=7)
    plt.plot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 'ro', markersize=7)
    plt.title('Blue dots initial, Red dots final centroids')
    plt.xlabel('X1 (scaled)')
    plt.ylabel('X2 (scaled)')
    plt.show()

    np.random.seed(6)

    DIST = metrics.pairwise_distances
    init = Xs[np.random.choice(np.arange(Xs.shape[0]), 2), :]
    total_deviations = (DIST(Xs, Xs) ** 2).sum() / Xs.shape[0]
    inertia = []
    between = []
    within = []
    plt.figure(figsize=(12, 8))

    inertia_prev = 1e12
    plt.subplot(2, 2, 1)
    for i in range(1, 100):

        kmeans = KMeans(n_clusters=2, init=init, n_init=1, max_iter=1).fit(Xs)
        plt.scatter(Xs[:, 0], Xs[:, 1], c=kmeans.labels_)
        plt.xlabel('X1 (scaled)')
        plt.ylabel('X2 (scaled)')
        if (i == 1):
            plt.plot(init[:, 0], init[:, 1], 'bo', markersize=9)
        else:
            plt.plot(init[:, 0], init[:, 1], 'ro', markersize=i)

        init = kmeans.cluster_centers_

        n0 = np.sum(kmeans.labels_ == 0)
        n1 = np.sum(kmeans.labels_ == 1)
        x0 = Xs[np.where(kmeans.labels_ == 0)[0], :]
        x1 = Xs[np.where(kmeans.labels_ == 1)[0], :]
        w0 = .5 * (DIST(x0, x0) ** 2).sum() / n0
        w1 = .5 * (DIST(x1, x1) ** 2).sum() / n1

        inertia.append(kmeans.inertia_)
        within.append(w0 + w1)
        between.append(total_deviations - w0 - w1)

        if (np.isclose(kmeans.inertia_, inertia_prev, rtol=1e-3)):
            plt.title('Blue dots initial, (Biggest) red dots final centroids')
            print('Stop Iteration at')
            print('{}th step'.format(i))
            print('Final centroids: \n{}'.format(np.round(kmeans.cluster_centers_, 2)))
            break
        else:
            inertia_prev = kmeans.inertia_

    plt.subplot(2, 2, 2)
    plt.plot(np.arange(1, len(inertia) + 1), inertia)
    plt.title('Inertia')
    plt.xlabel('iter')
    plt.subplot(2, 2, 3)
    plt.plot(np.arange(1, len(between) + 1), between)
    plt.title('Between cluster deviations')
    plt.xlabel('iter')
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(1, len(within) + 1), within)
    plt.title('Within cluster deviations')
    plt.xlabel('iter')
    plt.gcf().tight_layout(pad=3.0)
    plt.show()


main()



