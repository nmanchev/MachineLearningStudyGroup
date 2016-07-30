#!/usr/bin/env python
"""
(C) 2016 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

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

    # Normal equations method
    xTx_inv = np.linalg.inv(x.T.dot(x))
    xTy = x.T.dot(y)
    w = xTx_inv.dot(xTy)
        
    print("Model parameters:\n")
    print(w)
    
if __name__ == "__main__":
    main()    