"""
(C) 2016 Nikolay Manchev
[London Machine Learning Study Group](http://www.meetup.com/London-Machine-Learning-Study-Group/members/)

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

import matplotlib.pyplot as plt

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
    
# Initialise the data set
np.random.seed(100)

# Generate data using one of the two functions (linear and non-linear)
#(x,y) = corr_vars(start=-3, stop=3, sigma=5, step = 0.005, func=lambda x:np.power(x,4) - 3*np.power(x,3) +8*np.power(x,2) +  + 7*x)
(x,y) = corr_vars(start=-3, stop=3, sigma=5, step = 0.005, func=lambda x:7+6*x)

x = np.array([x]).T
y = np.array([y]).T

# Add ones for w_0
mat_ones = np.ones(shape=(x.size, 2))
mat_ones[:,1] = x[:,0]
x = mat_ones

# Normal equations method
xTx_inv = np.linalg.inv(x.T.dot(x))
xTy = x.T.dot(y)
w = xTx_inv.dot(xTy)
    
print("Model parameters:\n")
print(w)    

# Make predictions on the training set
y_hat = w[0] + w[1]*x[:,1]

# Get the residuals
y_hat = y_hat.reshape(y_hat.shape[0],-1)
residuals = np.subtract(y_hat, y)

# Plot residuals vs predictions
plt.rcParams.update({'font.size': 15})

f, ax = plt.subplots( figsize=(10,8))

ax.scatter(y_hat, residuals, s=10)
ax.axhline(0, color='red')

ax.set_title("Residuals vs fitted")
ax.set_ylabel("Residuals")
ax.set_xlabel("$\hat y$",fontsize=20)
ax.grid(True)
