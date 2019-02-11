# DecisionTree

This is a machine learning library developed by Taylor Smith for
CS5350/6350 in University of Utah

[(1, {'Majority Error': [0.698, 0.7032967032967034], 'Entropy': [0.698, 0.7032967032967034], 'Gini Index': [0.698, 0.7032967032967034]}), (2, {'Majority Error': [0.698, 0.7032967032967034], 'Entropy': [0.778, 0.7774725274725275], 'Gini Index': [0.778, 0.7774725274725275]}), (3, {'Majority Error': [0.719, 0.6689560439560439], 'Entropy': [0.819, 0.8035714285714286], 'Gini Index': [0.824, 0.8159340659340659]}), (4, {'Majority Error': [0.857, 0.7980769230769231], 'Entropy': [0.918, 0.8489010989010989], 'Gini Index': [0.911, 0.8626373626373627]}), (5, {'Majority Error': [0.963, 0.9107142857142857], 'Entropy': [0.973, 0.9162087912087912], 'Gini Index': [0.973, 0.9162087912087912]}), (6, {'Majority Error': [1.0, 0.9107142857142857], 'Entropy': [1.0, 0.9162087912087912], 'Gini Index': [1.0, 0.9162087912087912]})]

Unknown is treated as Data
[(1, {'Majority Error': [0.9284, 0.8268], 'Entropy': [0.9284, 0.8268], 'Gini Index': [0.9284, 0.8268]}), 
(2, {'Majority Error': [0.9996, 0.8294], 'Entropy': [0.9996, 0.8294], 'Gini Index': [0.9996, 0.8294]}), 
(3, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(4, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(5, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(6, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(7, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(8, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(9, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(10, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(11, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(12, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(13, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(14, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(15, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]}), 
(16, {'Majority Error': [1.0, 0.8294], 'Entropy': [1.0, 0.8294], 'Gini Index': [1.0, 0.8294]})]

Unknown is treated as Missing Data
[(1, {'Majority Error': [0.9284, 0.8268], 'Entropy': [0.9284, 0.8268], 'Gini Index': [0.9284, 0.8268]}), 
(2, {'Majority Error': [0.9996, 0.828], 'Entropy': [0.9996, 0.828], 'Gini Index': [0.9996, 0.828]}), 
(3, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(4, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(5, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(6, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(7, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(8, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(9, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(10, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(11, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(12, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(13, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(14, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(15, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]}), 
(16, {'Majority Error': [1.0, 0.828], 'Entropy': [1.0, 0.828], 'Gini Index': [1.0, 0.828]})]
