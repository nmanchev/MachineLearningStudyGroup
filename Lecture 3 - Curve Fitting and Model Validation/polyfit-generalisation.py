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

from sklearn.cross_validation import train_test_split

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
    
def trainPolyFit(x, y, order):
    x = polyMatrix(x, order)

    # Add ones for w_0    
    x = np.hstack((np.array([np.ones(x.shape[0])]).T, x))

    # Hold out        
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)   

    # Initialise model parameters    
    w = np.array([np.zeros(x.shape[1])]).T
            
    # Normal equations method
    xTx = np.linalg.inv(x_train.T.dot(x_train))
    xTy = x_train.T.dot(y_train)
    w = xTx.dot(xTy)    

    # Compute error
    N = y_test.shape[0]

    train_err = np.sum( (y_hat(x_train, w) - y_train) ** 2 ) / (2 * N)
    test_err = np.sum( (y_hat(x_test, w) - y_test) ** 2 ) / (2 * N)
    
    return train_err, test_err


np.random.seed(100)

(x,y) = corr_vars(sigma=2, func=lambda x: 4*np.log2(x))

x = np.array([x]).T
y = np.array([y]).T

# Vary the order of the model and compute the
# training and test errors    
errors = np.zeros([5,2])
for order in range(1,len(errors) + 1):
    errors[order-1, ] = trainPolyFit(x, y, order)

print("Training and test errors:\n")
print(errors)

# Plot X and y
f, ax1 = plt.subplots(figsize=(7,7))

# Plot the regression line
x_plot = np.arange(1, len(errors)+1)
ax1.plot(x_plot, errors[:,0], color='b', label="Training")
ax1.plot(x_plot, errors[:,1], color='r', label="Test")
ax1.grid(True)

ax1.legend(loc="upper right", shadow=True)
  