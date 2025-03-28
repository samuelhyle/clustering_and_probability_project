import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import export_text, plot_tree, DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score

def main():

    data = pd.read_csv("mushroom_data.csv") # read the data
    print(data.shape)
    data.head()

    sns.countplot(x="y", data=data)
    plt.title("y")
    plt.show()

    sns.countplot(x="gill-color", data=data, hue="y")
    plt.xticks(rotation=45)
    plt.title("Gill color")
    plt.show()

    sns.countplot(x="odor", data=data, hue="y")
    plt.xticks(rotation=45)
    plt.title("Odor")
    plt.show()

    y = np.where(data['y'] == "edible", 1, 0)  # encode edibility as 1 or 0

    X = data[data.columns]
    X = X.drop(columns=["y"])

    print(y)

    X = pd.get_dummies(X, dtype=int)
    print(X.shape)
    X.head()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=2)

    print(X_train.shape)
    print(X_test.shape)

    clf = DecisionTreeClassifier(max_depth=2, random_state=0)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    confmat = (y_test, y_pred)

    print("Accuracy:", acc)

    ax = plt.subplot()
    sns.heatmap(confmat, annot=True, fmt='g', ax=ax)
    ax.set_xlabel('Predicted labels', fontsize=15)
    ax.set_ylabel('True labels', fontsize=15)
    plt.show()

    plot_tree(clf, feature_names=list(X.columns), filled=True, class_names=["poisonous", "edible"])
    plt.show()

    import pydotplus
    d_tree = export_graphviz(clf, feature_names=list(X.columns), filled=True, class_names=["poisonous", "edible"])
    pydot_graph = pydotplus.graph_from_dot_data(d_tree)
    pydot_graph.write_pdf('mushroom_tree.pdf')

main()

