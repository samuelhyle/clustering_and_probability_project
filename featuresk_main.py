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


class ClusteredFeatures:

    def __init__(self, n_clusters, add_constant=False, random_state=0):
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.add_constant = add_constant
        self.kmeans = None

    def fit(self, x):
        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state).fit(x)

    def transform(self, x):
        assert self.kmeans is not None, 'You have to call fit before transform!'

        k = self.kmeans.n_clusters
        assignments = self.kmeans.predict(x)

        x_new = np.zeros((x.shape[0], k))
        bool_ind = (np.arange(k).reshape(-1, 1) == assignments).T
        x_new[bool_ind] = x.reshape(-1)

        if (self.add_constant):
            for j in range(1, k):
                ones = np.zeros((x.shape[0], 1))
                ones[np.where(assignments == j)] = 1
                x_new = np.hstack((x_new, ones))

        return x_new

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

def main():

    data = pd.read_csv('featuresK.csv').to_numpy()
    x_train, y_train = data[:, 0].reshape(-1, 1), data[:, 1]

    km = KMeans(n_clusters=2, n_init=10).fit(x_train)
    plt.scatter(x_train, y_train, c=km.labels_)
    plt.xlabel('X (feature)')
    plt.ylabel('Y')
    plt.show()

    lm = LinearRegression().fit(x_train, y_train)
    x_eval = np.linspace(0, 9, 50).reshape(-1, 1)
    y_eval = lm.predict(x_eval)
    yhat_train = lm.predict(x_train)
    mse_train = MSE(y_train, yhat_train)

    plt.figure(figsize=(6, 6))
    plt.plot(x_eval, y_eval, 'k-')
    plt.scatter(x_train, y_train, c=km.labels_)
    plt.title('training error: {}'.format(round(mse_train, 2)))
    plt.xlabel('X (feature)')
    plt.ylabel('Y')
    plt.show()

    cf = ClusteredFeatures(n_clusters=2, add_constant=True, random_state=0)

    xnew_train = cf.fit_transform(x_train)

    xnew_eval = cf.transform(x_eval)

    lm = LinearRegression().fit(xnew_train, y_train)
    yhat_eval = lm.predict(xnew_eval)
    yhat_train = lm.predict(xnew_train)
    mse_train = MSE(y_train, yhat_train)

    plt.figure(figsize=(6, 6))
    plt.plot(x_eval, yhat_eval, 'k-')
    plt.scatter(x_train, y_train, c=km.labels_)
    plt.xlabel('X (feature)')
    plt.ylabel('Y')
    plt.title("training error: {}".format(round(mse_train, 2)))
    plt.show()

main()