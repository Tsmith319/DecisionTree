# DecisionTree

This is a machine learning library developed by Taylor Smith for
CS5350/6350 in University of Utah

The file carID3 contains the implementation of ID3 specifically for the data provided in car.zip under the name carID3.py. When ran, the program will print out the data the data in the format presented below for each type of approach for splitting the data set with the first index being the accuracy of the training data at the listed tree depth and the second index represents the testing accuracy of the training data. 
[(1, {'Majority Error': [0.698, 0.7032967032967034], 'Entropy': [0.698, 0.7032967032967034], 'Gini Index': [0.698, 0.7032967032967034]}), (2, {'Majority Error': [0.698, 0.7032967032967034], 'Entropy': [0.778, 0.7774725274725275], 'Gini Index': [0.778, 0.7774725274725275]}), (3, {'Majority Error': [0.719, 0.6689560439560439], 'Entropy': [0.819, 0.8035714285714286], 'Gini Index': [0.824, 0.8159340659340659]}), (4, {'Majority Error': [0.857, 0.7980769230769231], 'Entropy': [0.918, 0.8489010989010989], 'Gini Index': [0.911, 0.8626373626373627]}), (5, {'Majority Error': [0.963, 0.9107142857142857], 'Entropy': [0.973, 0.9162087912087912], 'Gini Index': [0.973, 0.9162087912087912]}), (6, {'Majority Error': [1.0, 0.9107142857142857], 'Entropy': [1.0, 0.9162087912087912], 'Gini Index': [1.0, 0.9162087912087912]})]

The file bankID3 contains the implementation of ID3 specicifically for the data provided in bank.zip under the name bankID3.py. When ran, the program will run the ID3 algorithm for both cases of where an unknown attribute is treated as part of the data set and where the attribute is treated as a missing attribute. The data will print to the terminal like so when ran where it will be presented in a similar format as the carID3.py file where at each depth it will present the training accuracy in the first index for each approach of splitting the data set and the testing accuracy in the second index. 
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
