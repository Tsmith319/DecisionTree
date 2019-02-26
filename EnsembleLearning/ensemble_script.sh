#!/bin/bash

clear
echo "Running Ada Boost Algorithm"
echo " "
python AdaBoost.py
echo " "
echo "Running Bagged Trees algorithm"
echo " "
python Bagging.py
echo " "
echo "Running 100 predictors algorithm for Bagged Trees bias, variance, and squared error"
echo " "
python 100predictors.py
echo " "
echo "Running Random Forest algorithm"
echo " "
python random_tree.py
echo " "
echo "Running Algorithm to find bias, variance, and squared error terms on random forest algorithm"
echo " "
python randoForestPredictors.py
echo " "
echo "Done with script"