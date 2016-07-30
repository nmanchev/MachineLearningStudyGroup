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

def corr_vars( start=1, stop=10, step=1, mu=0, sigma=3, func=lambda x: x ):
    """
    Generates a data set of (x,y) pairs with an underlying regularity. y is a
    function of x in the form of
    
    y = f(x) + e
    
    Where f(x) is specified by the *func* argument and e is a random Gaussian
    noise specified by *mu* and *sigma*.
    """
    
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

def gradient_descent(x, y, w, max_iter, alpha = 0.001):
    """
    Performs gradient descent to optimise w.

    Keyword arguments:
      
      *x* : Numpy array
        matrix of independent variables

      *y* : Numpy array
        columnar vector of target values

      *w* : Numpy array
        initial model parameters

      *max_iter* : int
        maximum number of iterations

      *alpha* : int, optional
        learning rate (defaults to 0.05)
        
    Returns: 

      *J_hist* : Numpy array
        values of J(w) at each iteration

      *w* : Numpy array
        estimated model parameters
    """
    
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

    # Initialise the data set

    np.random.seed(100)

    (x,y) = corr_vars(sigma=2, func=lambda x: 4*np.log2(x))

    x = np.array([x]).T
    y = np.array([y]).T
    
    # Add ones for w_0
    mat_ones = np.ones(shape=(9, 2))
    mat_ones[:,1] = x[:,0]
    x = mat_ones
    
    # Print the X and y
    print("X:")
    print(x)
    
    print("\nY:")
    print(y)
    
    m,n=np.shape(x)

    # Initialise w with zeros      
    w = np.array([np.ones(n)]).T    

    # Perform gradient descent
    (j_hist, w) = gradient_descent(x, y, w, 10, 0.1)

    print("Model parameters:\n")
    print(w)

    # Plot X and y
    f, (ax1,ax2) = plt.subplots(1, 2, figsize=(7,7))
    ax1.scatter(x[:,1], y)        
    
    # Plot the regression line
    ax1.plot(x[:,1], y_hat(x, w), color='r')
    ax1.grid(True)
    
    # Plot the change of J(w)
    x = np.arange(1,j_hist.size + 1)
    y = j_hist
    
    ax2.plot(x, j_hist)
    ax2.grid(True)       

if __name__ == "__main__":
    main()    