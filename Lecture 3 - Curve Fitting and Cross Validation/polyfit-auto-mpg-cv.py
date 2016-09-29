#!/usr/bin/env python
"""
(C) 2016 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

from sklearn.cross_validation import KFold


def y_hat(x, w):
    """
    Linear regression hypothesis: y_hat = X.w
    """   
    return x.dot(w)
    
def polyMatrix(v, order):       
    """
    Given a nx1 vector v, the function generates a matrix of the form:
        
        [ v[0] v[0]^2 ... v[0]^order ]
        [ v[1] v[1]^2 ... v[1]^order ]
        [             ...            ]
        [ v[n] v[n]^2 ... v[n]^order ]

    """   
    vector = v
    v_pow  = 2
    
    while v_pow <= order:        
      v = np.hstack((v, np.power(vector, v_pow)))            
      v_pow = v_pow + 1
        
    return v

# Load the data set
# We use Auto MPG from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Auto+MPG

car_data = np.genfromtxt("auto-mpg.data", usecols=(0, 3))
car_data = car_data[~np.isnan(car_data).any(axis=1)]

# Assign Horsepower attribute to x and MPG to y
x = car_data[:,1]
y = car_data[:,0]

x = np.array([x]).T
y = np.array([y]).T

# Set the order of the model and generate the X matrix
k = 1
x = polyMatrix(x, k)

# Set the number of folds
folds = 10

# Get the folds indexes
kf = KFold(x.shape[0], n_folds = folds, shuffle = True)

# Initialise an array to keep the errors from each iteration
sse = np.zeros(folds)

fold_index = 0

# Perform k-fold cross valdation
for train_index, test_index in kf:
       
    # Get the training and test subsets
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Add ones for w_0    
    x_train = np.hstack((np.array([np.ones(x_train.shape[0])]).T, x_train))
    x_test  = np.hstack((np.array([np.ones(x_test.shape[0])]).T, x_test))

    # Initialise model parameters    
    w = np.array([np.zeros(x_train.shape[1])]).T
    
    # Normal equations method
    xTx = np.linalg.inv(x_train.T.dot(x_train))
    xTy = x_train.T.dot(y_train)
    w = xTx.dot(xTy)

    # Compute error sum of squares
    sse[fold_index] = np.sum( (y_hat(x_test, w) - y_test) ** 2)    
    print("SSE[%i]: %.2f" % (fold_index, sse[fold_index]))    
    
    fold_index = fold_index + 1

# Print the average error from all folds
print("Average SSE : %.2f" % (np.average(sse)))
    
