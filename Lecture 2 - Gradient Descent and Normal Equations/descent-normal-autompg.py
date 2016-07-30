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

def y_hat(x, w):
    """
    Linear regression hypothesis: y_hat = X.w
    """   
    return x.dot(w)
    
def gradient_descent(x, y, w, max_iter, alpha = 0.05):
    N = y.shape[0]
    
    J_hist = np.zeros(max_iter)

    print("\nGradient descent starts\n")

    for i in range(0, max_iter):
        
        J = np.sum( (y_hat(x, w) - y) ** 2 ) / (2 * N)

        J_hist[i] = J
        
        print("Iteration %d, J(w): %f\n" % (i, J))
        
        gradient = np.dot(x.T, y_hat(x, w) - y) / N    
        
        w = w - alpha * gradient

    print("Gradient descent finished.\n")
        
    return (J_hist, w)

def main():

    np.random.seed(100)

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

    # Normalize
    x = (x - np.mean(x)) / np.std(x) 
        
    # Add ones for w_0    
    x = np.hstack((np.array([np.ones(x.shape[0])]).T, x))

    # Initialise model parameters    
    w = np.array([np.zeros(x.shape[1])]).T

    (j_hist, w) = gradient_descent(x, y, w, 20, 0.5)

    print("Gradient Descent Model parameters:\n")
    print(w, '\n')
   
    # Normal equations method
    xTx = np.linalg.inv(x.T.dot(x))
    xTy = x.T.dot(y)
    w = xTx.dot(xTy)

    print("Normal Equations Model parameters:\n")
    print(w)
    
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(12,8))
    ax1.scatter(x[:,1], y)        
    ax1.plot(x[:,1], y_hat(x, w), color='r')
    ax1.set_title("Horsepower vs MPG")
    ax1.grid(True)
    
    
    x = np.arange(1,j_hist.size + 1)
    y = j_hist
    
    ax2.plot(x, j_hist)
    ax2.set_title("J(w)")
    ax2.grid(True)
    
if __name__ == "__main__":
    main()    