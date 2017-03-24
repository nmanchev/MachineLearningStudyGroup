#!/usr/bin/env python
"""
(C) 2017 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

import matplotlib.pyplot as plt

x = np.array([[1,2,3,4,5,6,7,8,9,10]]).T
y = np.array([[0,0,0,0,0,1,1,1,1,1]]).T

# Normalize the inputs
x = (x - np.mean(x)) / np.std(x) 

# Add ones for w_0
mat_ones = np.ones(shape=(x.shape[0], 2))
mat_ones[:,1] = x[:,0]
x = mat_ones   

# Normal equations method
xTx = np.linalg.inv(x.T.dot(x))
xTy = x.T.dot(y)
w = xTx.dot(xTy)
    
print("Model parameters:\n")
print(w)

# Plot X and y
f, ax1 = plt.subplots(1, 1, figsize=(7,7))
ax1.scatter(x[:,1], y)        

# Make predictions on the training set
y_hat = w[0] + w[1]*x[:,1]

# Plot the regression line
ax1.plot(x[:,1], y_hat, color='r')
ax1.grid(True)