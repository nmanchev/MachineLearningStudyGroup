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


def mode(dataset):
    """
    Computes the mode (i.e. most frequent value) of the dataset 

    Parameters
    ----------
    dataset    : a list of values

    Returns
    -------
    the distinct value with highest frequency of occurrence    
    """    
    counts = count_distinct(dataset)
    return max(counts, key=counts.get)


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

def IG(dataset, attr_index, labels):
    """
    Computes the expected reduction of entropy if the dataset is
    split by a specific attribute.
    
    IG(dataset, attribute) = H(dataset) - H(dataset|attribute)    
    
    Parameters
    ----------
    dataset    : a list of values
    attr_index : index of an attribute to split on
    labels     : class labels for the examples in dataset

    Returns
    -------
    IG(dataset, attribute)
    """     
    # Get the dataset distinct values and their respective 
    # frequency of occurrence
    dataset_attributes = count_distinct(dataset[:,attr_index])    

    # Start with 0 entropy
    I = 0

    # Compute the entropy of the split
    # I(X, A) = \sum_{i=1}^{m} \frac{|X_i|}{|X|} \times H(X_i)
    for key in dataset_attributes.keys():
  
        # Compute the weighted average \frac{|X_i|}{|X|}   
        p = dataset_attributes[key] / sum(dataset_attributes.values())

        # Get the class labels for X_i
        subset_labels = labels[dataset[:,attr_index] == key] 

        # Add \frac{|X_i|}{|X|} \times H(X_i) to I(X,A)
        I = I + p * entropy(subset_labels)
    
    # Return H(dataset) - I(dataset, A)
    return entropy(dataset[:,attr_index]) - I


def select_best(dataset, attributes, labels):
    """
    Selects the best attribute to split on based on reduction of entropy.
      
    Parameters
    ----------
    dataset    : a list of values
    attributes : names of the attributes in the dataset
    labels     : class labels for the examples in dataset

    Returns
    -------
    The attribute that maximizes the decrease of entropy after
    splitting
    """         
    best_IG = 0
    best_attr = None
    
    # Go over all attributes of the set
    for attr in attributes:
        # Compute the expected Information Gain if we split on
        # that attribute
        gain = IG(dataset, attributes.index(attr), labels)
        # If the gain is higher than what we have so far select that attribute
        if (gain >= best_IG):
            best_IG = gain
            best_attr = attr
        
    # Return the attribute that produces the highest gain
    return best_attr

def build_tree(dataset, attributes, labels, default, verbose = False):
    
    if verbose:
        print("*****************")
        print("INPUT ATTRIBUTES:", attributes)
    
    # No data? Return default classification   
    if dataset.size == 0:
        return default
                
    # All examples have the same classification? Return this label
    if len(set(labels)) <= 1:

        if verbose:
            print("SAME CLASS    :", labels[0])
            print("*****************")

        return labels[0]

    # Attributes empty? Return MODE   
    if len(attributes) <= 1:
        return default

    # Choose best attribute
    attr = select_best(dataset, attributes, labels)
    
    if (attr == None):
        if verbose:
            print("NO ATTRIBUTE TO SPLIT ON")
            print("************************")
        return default
    
    if verbose:
        print("SPLITTING ON    :", attr)
        print("*****************")


    # Get distinct attribute index and values
    attr_index  = attributes.index(attr)
    attr_values = count_distinct(dataset[:,attributes.index(attr)]).keys()

    # Remove the selected attribute from the list of remaining attributes
    attributes = [x for x in attributes if x != attr]
        
    # Add a node for that attribute
    tree = {attr:{}}
        
    for v in attr_values:

        # Get the indexes of all examples that have value v for the
        # chosen attribute        
        indexes = dataset[:, attr_index] == v
        
        # Get all examples and their respective labels
        subtree_dataset = dataset[indexes]
        subtree_labels  = labels[indexes]    
        
        # Build a subtree using the selected examples
        subtree  = build_tree(subtree_dataset, attributes,
                              subtree_labels, mode(subtree_labels))
    
        # Attach the subtree
        tree[attr][v] = subtree
    
    return tree
       
def predict(tree, attributes, example):
    """
    Traverse a tree to make a prediction.
    
    Parameters
    ----------
    tree       : a dictionary containing a decision tree
    attributes : names of the attributes in the dataset
    example    : example to classify

    Returns
    -------
    The class label for this example.
    If the example cannot be classified, this function returns None.
    """
    # Get the attribute at the tree root
    for attr, value in tree.items():
      attr_index = attributes.index(attr)
      try:
        # Get the node that has the same value as in the example
        node = tree[attr][example[attr_index]]
      except KeyError:
        # No such node exists? We can't classify the example then
        return None
      if isinstance(node, dict):
        # Node exists, but it is a subtree. Traverse recursively.
        return predict(node, attributes, example)
      else:
        # Node exists and is a terminal node. Its value is the class label.
        return node         

def printTree(tree, attributes, offset = "|->"):
    """
    Prints a decision tree from dictionary.
    
    Parameters
    ----------
    tree       : a dictionary containing a decision tree
    attributes : names of the attributes in the dataset
    """
    for attr, value in tree.items():
      node = tree[attr]
      if isinstance(node, dict):
        print(offset,attr)
        printTree(node, attributes, ("    " + offset))
      else:
        print(offset,attr, "->", value)
      
    
# Load the data set

data   = np.array([[0,0,0],[1,0,1],[0,0,0],[0,0,0],[0,1,1],[1,0,0],[0,1,0],[1,1,1],[1,0,0],[1,0,0]])
#data   = np.array([[1,1,1],[2,1,2],[1,1,1],[1,1,1],[1,2,2],[2,1,1],[1,2,1],[2,2,2],[2,1,1],[2,1,1]])
#data   = np.array([[0,0,0],['A',0,'A'],[0,0,0],[0,0,0],['A','A','A'],['A',0,0],[0,'A',0],[0,'A','A'],['A',0,0],['A',0,0]])
labels = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 0])

#noise = np.random.normal(0, 0.5, len(data))
#data[:,1] += noise.astype(int)

# Set attribute names
attributes =   ["<2 YRS JOB", "MISSED PMNTS", "DEFAULTED"]    
class_labels = ["GOOD", "BAD"]

# Get the most frequent label
default = mode(labels)
    
tree = build_tree(data, attributes, labels, default)

printTree(tree, attributes)
#print(predict(tree, attributes, [1,0,1]))
    