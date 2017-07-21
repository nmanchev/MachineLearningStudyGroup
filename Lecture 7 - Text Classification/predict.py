"""
(C) 2017 Nikolay Manchev
[London Machine Learning Study Group](http://www.meetup.com/London-Machine-Learning-Study-Group/members/)

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

import numpy as np

import scipy

import timeit

np.random.seed(1234)

labels = np.fromfile("data/labels.csv", sep='\n')

tf_idf_matrix = scipy.io.mmread("data/training.mtx").todense()

X_train, X_test, y_train, y_test = train_test_split(tf_idf_matrix, labels, test_size=0.20, random_state=1234)

start_time = timeit.default_timer()

clf = MultinomialNB()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Elapsed time: %f sec" % (timeit.default_timer() - start_time))

print(accuracy_score(y_test, y_pred))
