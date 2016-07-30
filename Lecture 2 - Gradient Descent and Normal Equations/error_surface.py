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

from mpl_toolkits.mplot3d import Axes3D

def J(x,y,w_0,w_1):
    N = y.shape[0]    
    J = 0
    for i in range(0, len(x)):
       J = J + ((w_0 + w_1 * x[i]) - y[i] ) ** 2
    return J / N   
   
def main():
    # Initialise the data set   
    
    car_data = np.genfromtxt("auto-mpg.data", usecols=(0, 3))
    car_data = car_data[~np.isnan(car_data).any(axis=1)]

    # Assign Horsepower attribute to x and MPG to y
    x = car_data[:,1]
    y = car_data[:,0]

    x = np.array([x]).T
    y = np.array([y]).T

    # Normalize
    x = (x - np.mean(x)) / np.std(x)    
    

    w_0 = np.linspace(-300.0, 300.0, 50)
    w_1 = np.linspace(-300.0, 300.0, 50)

    W0, W1 = np.meshgrid(w_0, w_1)
    
    E = np.array([J(x, y, w_0, w_1) for w_0, w_1 in zip(np.ravel(W0), np.ravel(W1))])

    E = E.reshape(W0.shape)
    
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(W0, W1, E, rstride=1, cstride=1, color='b', alpha=0.5)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticklabels([])

    ax.set_xlabel('$w_0$', fontsize=16)
    ax.set_ylabel('$w_1$', fontsize=16)
    ax.set_zlabel('$J(w_0, w_1)$', fontsize=16)
    
    plt.show()       
    '''
    plt.figure()
    CS = plt.contour(W0, W1, E)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.title('Simplest default with labels')
    '''
if __name__ == "__main__":
    main()    