#!/usr/bin/env python
"""
(C) 2016 Nikolay Manchev, London Machine Learning Study Group

http://www.meetup.com/London-Machine-Learning-Study-Group/

This work is licensed under the Creative Commons Attribution 4.0 International
License. To view a copy of this license, visit
http://creativecommons.org/licenses/by/4.0/.
"""

import numpy as np
import math


def split(dataset, attribute, value):
    """
    Split a dataset in two by value of an attribute

    Parameters
    ----------
    dataset    : dataset for the split
    attribute  : attribute to split on     
    value      : threshold value for the split

    Returns
    -------
    a tuple containing the two splits   
    """
    set_one = dataset[dataset[:, attribute] > value]
    set_two = dataset[dataset[:, attribute] <= value]
    return (set_one, set_two)


def count_distinct(dataset):
    """
    Gets a list of unique values in a dataset and computes the
    frequency of occurrence for each unique value.

    Parameters
    ----------
    dataset    : a list of values

    Returns
    -------
    a dictionary of unique values and their respective frequency 
    of occurrence    
    """            
    counts = {}

    # Loop over all elements of the dataset    
    for item in dataset:        
        if (item in counts):
            # This value is already in the dictionary. 
            # Increase its count.
            counts[item] = counts[item] + 1
        else:
            # This is the first occurrence of the word.
            # Add it to the dictionary and set its count to 1
            counts[item] = 1            
    return counts


def entropy(dataset):
    """
    Computes the entropy for a dataset. The entropy is computed as    

      H = sum_{i} p(x_i) log_2 p(x_i)

    The sum is taken over all unique values in the set. The
    probability p(x_i) is computed as  
    
    p(x_i) = (frequency of occurrence of x_i) / (size of the dataset)

    Parameters
    ----------
    dataset    : a list of values

    Returns
    -------
    the entropy of the set          
    """    
    H = 0
    
    for freq in count_distinct(dataset).values():
        H += (-freq/len(dataset)) * math.log(freq/len(dataset), 2) 
        
    return H

   
def show_split_entropy(dataset, attr_index, split_value):
    """
    Splits a dataset on attribute and prints the frequency of occurrence
    and the entropy for each split.

    Parameters
    ----------
    dataset     : a list of values
    attr_index  : index of an attribute to split on
    split_value : threshold value for the split 
         
    """
    # Split the dataset in two subsets
    (x1, x2) = split(dataset,attr_index,split_value)

    # Print the frequencies and entropy for the first subset
    print("First split")
    print("**************")
    print("Value counts: ", count_distinct(x1[:,attr_index]))
    print("Entropy: ", entropy(x1[:,attr_index]), "\n")

    # Print the frequencies and entropy for the second subset
    print("Second split")
    print("**************")
    print("Value counts: ", count_distinct(x2[:,attr_index]))
    print("Entropy: ", entropy(x2[:,attr_index]))


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

# Set attribute names
attributes =   ["CYLYNDERS", "DISPLACEMENT", "HORSEPOWER", "WEIGHT", "ACCELERATION", "MODEL_YEAR", "ORIGIN"]
class_labels = ["BAD", "OK", "GOOD"]

# Look at the unique values for the MODEL_YEAR attribute
print("Unique values for MODEL_YEAR: ", count_distinct(data[:,5]), "\n")

# Split the dataset at the value 75
show_split_entropy(data, 5, 75)
