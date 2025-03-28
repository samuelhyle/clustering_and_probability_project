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
from re import findall
import seaborn as sns
from sklearn.metrics import confusion_matrix as calculate_confusion_matrix

comments = pd.read_csv('youtube_comments.csv')

training_comments = comments[len(comments) // 3:]
testing_comments = comments[:len(comments) // 3]

print(training_comments[:5])

correct_posterior = 3

prior_spam_probability = (training_comments['CLASS'] == 1).mean()


def find_words(string):

    return findall("[a-z0-9äö']+", string)


def extract_frequencies(dataframe, column):

    return dataframe[column].str.lower().apply(find_words).explode().value_counts()


spam_word_frequencies = extract_frequencies(training_comments.query('CLASS == 1'), 'CONTENT')
ham_word_frequencies = extract_frequencies(training_comments.query('CLASS == 0'), 'CONTENT')

print(f'Frequency of the word "love" in spam comments: {spam_word_frequencies.get("love", 0)}')
print(f'Frequency of the word "love" in ham comments: {ham_word_frequencies.get("love", 0)}')


def likelihood_spam(word):

    if word in spam_word_frequencies:
        return spam_word_frequencies[word] / spam_word_frequencies.sum()
    return 1


def likelihood_ham(word):

    if word in ham_word_frequencies:
        return ham_word_frequencies[word] / ham_word_frequencies.sum()
    return 1


np.testing.assert_almost_equal(likelihood_spam('part'), 0.000363735564244794)
np.testing.assert_almost_equal(likelihood_ham('part'), 0.0016826518593303045)
np.testing.assert_equal(likelihood_spam('non-existant-word'), 1.0)
np.testing.assert_equal(likelihood_ham('non-existant-word'), 1.0)


def posterior_spam(comment):

    words = list(set(findall("[a-z0-9äö']+", comment.lower())))

    prod_spam = np.prod([likelihood_spam(w) for w in words])
    prod_ham = np.prod([likelihood_ham(w) for w in words])

    p_spam = prior_spam_probability
    p_ham = 1 - p_spam
    numerator = p_spam * prod_spam
    denominator = numerator + p_ham * prod_ham

    return numerator / denominator if denominator != 0 else 0


predictions = np.array([posterior_spam(comment) for comment in testing_comments['CONTENT']])
discretized_predictions = predictions > 0.5
labels = testing_comments['CLASS']

accuracy = np.sum(discretized_predictions == labels) / len(testing_comments)
print(f"Classification accuracy: {accuracy}")

confusion_matrix = calculate_confusion_matrix(labels, discretized_predictions)
sns.heatmap(
    confusion_matrix.T,
    square=True,
    annot=True,
    fmt='d',
    cbar=False,
    xticklabels=['ham', 'spam'],
    yticklabels=['ham', 'spam']
)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.title('Confusion Matrix')
plt.show()