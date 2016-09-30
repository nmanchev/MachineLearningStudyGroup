#!/usr/bin/env python
"""
(C) 2016 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import matplotlib.pyplot as plt
import numpy as np

from numpy.polynomial.polynomial import polyval

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

# Set the order of the model and get the X matrix
k = 1
x = polyMatrix(x, k)

# Add ones for w_0    
x = np.hstack((np.array([np.ones(x.shape[0])]).T, x))

# Initialise model parameters    
w = np.array([np.zeros(x.shape[1])]).T
    
# Normal equations method
xTx = np.linalg.inv(x.T.dot(x))
xTy = x.T.dot(y)
w = xTx.dot(xTy)

print("Normal Equations Model parameters:\n")
print(w)

# Plot the data points
f, ax1 = plt.subplots(1, 1, figsize=(7,7))
ax1.scatter(x[:,1], y)       

# Plot a smooth curve using the fitted coefficients
x_smooth = np.linspace(x[:,1].min(), x[:,1].max(), 200)
f = np.squeeze(polyval(x_smooth, w))
ax1.plot(x_smooth, f, color='r')

ax1.set_title('Auto MPG - MPG vs Horsepower')
ax1.grid(True)
    