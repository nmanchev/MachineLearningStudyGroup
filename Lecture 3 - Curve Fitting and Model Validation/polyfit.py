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


def corr_vars( start=1, stop=10, step=1, mu=0, sigma=3, func=lambda x: x ):

    # Generate x
    x = np.arange(start, stop, step)  
    
    # Generate random noise
    e = np.random.normal(mu, sigma, x.size)
    
    # Generate y values as y = func(x) + e
    y = np.zeros(x.size)
    
    for ind in range(x.size):
        y[ind] = func(x[ind]) + e[ind]
    
    return (x,y)

def y_hat(x, w):
    """
    Linear regression hypothesis: y_hat = X.w
    """   
    return x.dot(w)
    
def polyMatrix(x, order):       
    """
    Given a nx1 vector x, the function generates a matrix of the form:
        
        [ x[0] x[0]^2 ... x[0]^order ]
        [ x[1] x[1]^2 ... x[1]^order ]
        [             ...            ]
        [ x[n] x[n]^2 ... x[n]^order ]

    """   
    vector = x
    x_pow  = 2
    
    while x_pow <= order:        
      x = np.hstack((x, np.power(vector, x_pow)))            
      x_pow = x_pow + 1
        
    return x

np.random.seed(100)

(x,y) = corr_vars(sigma=2, func=lambda x: 4*np.log2(x))

x = np.array([x]).T
y = np.array([y]).T

# Set the order of the model and get the X matrix
k = 1
x = polyMatrix(x, k)

# Add ones for w_0    
x = np.hstack((np.array([np.ones(x.shape[0])]).T, x))

# Initialise model parameters    
w = np.array([np.zeros(x.shape[1])]).T

# Print X and y
print(x,'\n')    
print(y,'\n')    

# Normal equations method
xTx = np.linalg.inv(x.T.dot(x))
xTy = x.T.dot(y)
w = xTx.dot(xTy)

# Print the model parameters
print("Normal Equations Model parameters:\n")
print(w)

# Plot the data points
f, ax1 = plt.subplots(1, 1, figsize=(7,7))
ax1.scatter(x[:,1], y)       

# Plot a smooth curve using the fitted coefficients
x_smooth = np.linspace(x[:,1].min(), x[:,1].max(), 200)
f = np.squeeze(polyval(x_smooth, w))
ax1.plot(x_smooth, f, color='r')

ax1.set_title('y = 4*log2(x) + e')
ax1.grid(True)
    
