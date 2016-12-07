#!/usr/bin/env python
"""
(C) 2016 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np
import pydotplus 

from sklearn import tree
from io import StringIO

# Load the data set
# We use a modified version of the Auto MPG from UCI Machine Learning 
# Repository where the continuous MPG attribute has been converted to
# categorical as follows:
#
# [9;19) - BAD
# (9;26] - OK
# (26;47] - GOOD
#
# The original dataset is available at 
# https://archive.ics.uci.edu/ml/datasets/Auto+MPG

car_data = np.genfromtxt("auto-mpg-modified.data", usecols = range(8))
car_data = car_data[~np.isnan(car_data).any(axis = 1)]

# Assign MPG to y and all other attributes to x
data   = car_data[:,1:]
labels = car_data[:,0]

# Uncomment to add some noise to the data
#noise = np.random.normal(0, 10, len(data))
#data[:,5] += noise.astype(int)

dt = tree.DecisionTreeClassifier(criterion = "entropy", max_depth=3)
dt = dt.fit(data, labels)

attributes =   ["CYLYNDERS", "DISPLACEMENT", "HORSEPOWER", "WEIGHT", "ACCELERATION", "MODEL_YEAR", "ORIGIN"]    
class_labels = ["BAD", "OK", "GOOD"]

out = StringIO()
tree.export_graphviz(dt,out_file=out, 
                     feature_names = attributes, 
                     class_names = class_labels, 
                     filled=True,
                     impurity = False) 

pydotplus.graph_from_dot_data(out.getvalue()).write_png("dtree.png")
