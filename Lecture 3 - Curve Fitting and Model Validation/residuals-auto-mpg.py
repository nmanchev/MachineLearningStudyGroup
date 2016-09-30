"""
(C) 2016 Nikolay Manchev
[London Machine Learning Study Group](http://www.meetup.com/London-Machine-Learning-Study-Group/members/)

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np

import matplotlib.pyplot as plt

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
    
# Add ones for w_0
mat_ones = np.ones(shape=(x.shape[0], 2))
mat_ones[:,1] = x[:,0]
x = mat_ones    

# Normal equations method
xTx_inv = np.linalg.inv(x.T.dot(x))
xTy = x.T.dot(y)
w = xTx_inv.dot(xTy)

# Print intercept and slope       
print("Model parameters:\n")
print(w)

# Make predictions on the training set
y_hat = w[0] + w[1]*x[:,1]

# Get the residuals
y_hat = y_hat.reshape(y_hat.shape[0],-1)
residuals = np.subtract(y_hat, y)

# Plot a histogram of the residuals
plt.figure(figsize=(10,8))
n, bins, patches = plt.hist(residuals, 30,  facecolor='green', alpha=0.75)

plt.title("Histogram of Residuals")
plt.grid(True)
plt.show()

# Plot residuals vs predictions
plt.rcParams.update({'font.size': 15})

f, ax = plt.subplots( figsize=(10,8))

ax.scatter(y_hat, residuals, s=10)
ax.axhline(0, color='red')

ax.set_title("Residuals vs fitted")
ax.set_ylabel("Residuals")
ax.set_xlabel("$\hat y$",fontsize=20)
ax.grid(True)
