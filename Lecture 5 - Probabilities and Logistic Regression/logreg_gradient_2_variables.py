"""
(C) 2017 Nikolay Manchev
[London Machine Learning Study Group](http://www.meetup.com/London-Machine-Learning-Study-Group/members/)

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

import matplotlib.pyplot as plt

def y_hat(x, w):
    """
    Logistic regression hypothesis: y_hat = 1 / (1 + e^(-x*w))
    """
    
    return (1/(1+np.exp(-x.dot(w))))

def gradient_ascent(x, y, w, max_iter, alpha = 0.01):
    """
    Performs gradient ascent to optimise L(w).

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
        learning rate (defaults to 0.01)
        
    Returns: 

      *L_hist* : Numpy array
        values of L(w) at each iteration

      *w* : Numpy array
        estimated model parameters
    """
    
    L_hist = np.zeros(max_iter)

    print("\nGradient ascent starts.\n")

    for i in range(0, max_iter):
        
        # Likelihood function
        L = np.sum(y.T.dot(np.log(y_hat(x, w))) + (1-y.T).dot(np.log(1-y_hat(x, w))))

        # Keep L(w) for each iteration (for the final plot)
        L_hist[i] = L
        
        print("Iteration %d, L(w): %f\n" % (i, L))
        
        # Compute the gradient and adjust the model parameters
        gradient = np.dot(x.T, y - y_hat(x, w)) 
        
        w = w + alpha * gradient

    print("Gradient ascent finished.\n")
        
    return (L_hist, w)


# Load the data set
# We use Auto MPG from UCI Machine Learning Repository
# https://archive.ics.uci.edu/ml/datasets/Auto+MPG

car_data = np.genfromtxt("auto-mpg.data", usecols=(4, 3, 7))
car_data = car_data[~np.isnan(car_data).any(axis=1)]

# Remove the data for Japan and recode US and Europe as 0 and 1
car_data = car_data[car_data[:,2]!=3]
car_data[:,1][car_data[:,1] == 1] = 0
car_data[:,1][car_data[:,1] == 2] = 1

# Assign Horsepower attribute to x and Origin to y
x = car_data[:,[0,1]]
y = car_data[:,2]
 
y = np.array([y]).T
    
# Normalize the inputs
weight_mean = np.mean(x[:,0])
weight_std  = np.std(x[:,0])
hp_mean = np.mean(x[:,1])
hp_std  = np.std(x[:,1])
x[:,0] = (x[:,0] - weight_mean) / weight_std 
x[:,1] = (x[:,1] - hp_mean) / hp_std

# Initialise w with ones      
m,n=np.shape(x)
w = np.array([np.ones(n)]).T    

# Perform gradient ascent
(l_hist, w) = gradient_ascent(x, y, w, 50)

print("Model parameters:\n")
print(w)

# Plot the data points and a gradient for the probability
# given by y_hat()
f, ax1 = plt.subplots(1, 1, figsize=(7,7))

x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, .1),
                     np.arange(y_min, y_max, .1))

Z = y_hat(np.c_[xx.ravel(), yy.ravel()], w) 
Z = Z.reshape(xx.shape)

ax1.contourf(xx, yy, Z, cmap=plt.cm.Blues)
ax1.scatter(x[:,0], x[:,1], c=y, cmap=plt.cm.bwr)        

# To make predictions use
# y_hat(np.array([((...weight_input...)-weight_mean)/weight_std,((...hp_input...)-hp_mean)/hp_std]), w)
