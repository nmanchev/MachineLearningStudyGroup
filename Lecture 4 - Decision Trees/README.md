## Decision Trees

Code examples used in Lecture 4

* auto-mpg-modified.data - A modified version of the Auto MPG data set from UCI Machine Learning Repository, with the continues MPG attribute partitioned as follows:
	* [9;19) - BAD
	* (9;26] - OK
	* (26;47] - GOOD
* entropy.py - Splits a data set by attribute and threshold value and computes the entropy for each split
* dt-credit.py - An implementation of a decision tree algorithm against the credit rating data set
* scikit-dt-auto-mpg.py - A decision tree trained on the modified Auto MPG data set, using DecisionTreeClassifier from scikit-learn
* overfit_demo.py - Accuracy score for training/test subset against the modified Auto MPG data set, using DecisionTreeClassifier from scikit-learn

This repository contains materials from the London Machine Learning Study Group Meetups

The meetup page is available at [http://www.meetup.com/London-Machine-Learning-Study-Group](http://www.meetup.com/London-Machine-Learning-Study-Group).

(C) 2016 Nikolay Manchev, London Machine Learning Study Group

This work is licensed under the Creative Commons Attribution 4.0 International License. To view a copy of this license, visit [http://creativecommons.org/licenses/by/4.0](http://creativecommons.org/licenses/by/4.0).
