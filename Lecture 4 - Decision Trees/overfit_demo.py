#!/usr/bin/env python
"""
(C) 2016 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

from sklearn import tree

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# Load the data set
# We use a modified version of the Auto MPG from UCI Machine Learning 
# Repository where the continuous MPG attribute has been converted to
# categorical as follows:
#
# [9;19) - BAD
# (9;26] - OK
# (26;47] - GOOD
#
# The original dataset is available at 
# https://archive.ics.uci.edu/ml/datasets/Auto+MPG

car_data = np.genfromtxt("auto-mpg-modified.data", usecols = range(8))
car_data = car_data[~np.isnan(car_data).any(axis = 1)]

# Assign MPG to y and all other attributes to x
data   = car_data[:,1:]
labels = car_data[:,0]

# Split the data into test/train subsets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1)

# Train a constrained Decision tree
dt = tree.DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(x_train, y_train)
pred_train = dt.predict(x_train)
pred_test  = dt.predict(x_test)
print("Prediction on training data :", accuracy_score(y_train, pred_train))
print("Prediction on test data     :", accuracy_score(y_test, pred_test))

