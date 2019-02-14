# DecisionTree

This is a machine learning library developed by Taylor Smith for
CS5350/6350 in University of Utah

To run the program successfully you need to run it on the CADE machines and make sure you are inside the DecisionTree folder. Once inside the folder you run the algorithm by typing ./shell_script.sh which will run both python folders that are used with car.zip and bank.zip data. The format of the data is explained below. 

The file carID3 contains the implementation of ID3 specifically for the data provided in car.zip under the name carID3.py. When ran, the program will print out the data the data in the format presented below for each type of approach for splitting the data set with the first index being the accuracy of the training data at the listed tree depth and the second index represents the testing accuracy of the training data. 

Running Car Data with ID3 algorithm

Current Tree Depth: 1

{'Majority Error': [0.698, 0.7032967032967034], 'Entropy': [0.698, 0.7032967032967034], 'Gini Index': [0.698, 0.7032967032967034]}

Current Tree Depth: 2

{'Majority Error': [0.698, 0.7032967032967034], 'Entropy': [0.778, 0.7774725274725275], 'Gini Index': [0.778, 0.7774725274725275]}

Current Tree Depth: 3

{'Majority Error': [0.719, 0.6689560439560439], 'Entropy': [0.819, 0.8035714285714286], 'Gini Index': [0.824, 0.8159340659340659]}

Current Tree Depth: 4

{'Majority Error': [0.857, 0.7980769230769231], 'Entropy': [0.918, 0.8489010989010989], 'Gini Index': [0.911, 0.8626373626373627]}

Current Tree Depth: 5

{'Majority Error': [0.963, 0.9107142857142857], 'Entropy': [0.973, 0.9162087912087912], 'Gini Index': [0.973, 0.9162087912087912]}

Current Tree Depth: 6

{'Majority Error': [1.0, 0.9107142857142857], 'Entropy': [1.0, 0.9162087912087912], 'Gini Index': [1.0, 0.9162087912087912]}


The file bankID3 contains the implementation of ID3 specicifically for the data provided in bank.zip under the name bankID3.py. When ran, the program will run the ID3 algorithm for both cases of where an unknown attribute is treated as part of the data set and where the attribute is treated as a missing attribute. The data will print to the terminal like so when ran where it will be presented in a similar format as the carID3.py file where at each depth it will present the training accuracy in the first index for each approach of splitting the data set and the testing accuracy in the second index. 

Unknown is treated as Data

[(1, {'Majority Error': [0.8912, 0.8834], 'Entropy': [0.8808, 0.8752], 'Gini Index': [0.8912, 0.8834]}), (2, {'Majority Error': [0.8958, 0.8912], 'Entropy': [0.894, 0.8886], 'Gini Index': [0.8948, 0.8896]}), (3, {'Majority Error': [0.9036, 0.8878], 'Entropy': [0.899, 0.89], 'Gini Index': [0.9036, 0.8876]}), (4, {'Majority Error': [0.9156, 0.8836], 'Entropy': [0.9204, 0.883], 'Gini Index': [0.922, 0.877]}), (5, {'Majority Error': [0.9246, 0.881], 'Entropy': [0.9364, 0.8712], 'Gini Index': [0.9362, 0.8698]}), (6, {'Majority Error': [0.9314, 0.8778], 'Entropy': [0.9502, 0.863], 'Gini Index': [0.947, 0.8554]}), (7, {'Majority Error': [0.9396, 0.8776], 'Entropy': [0.9574, 0.8516], 'Gini Index': [0.9594, 0.8454]}), (8, {'Majority Error': [0.9456, 0.8762], 'Entropy': [0.968, 0.8462], 'Gini Index': [0.9708, 0.8404]}), (9, {'Majority Error': [0.9504, 0.872], 'Entropy': [0.9718, 0.8392], 'Gini Index': [0.9746, 0.8354]}), (10, {'Majority Error': [0.9544, 0.8694], 'Entropy': [0.9784, 0.838], 'Gini Index': [0.9806, 0.8338]}), (11, {'Majority Error': [0.9584, 0.8642], 'Entropy': [0.9836, 0.8392], 'Gini Index': [0.984, 0.8346]}), (12, {'Majority Error': [0.9642, 0.8588], 'Entropy': [0.9848, 0.8392], 'Gini Index': [0.9846, 0.8344]}), (13, {'Majority Error': [0.9702, 0.8492], 'Entropy': [0.985, 0.8392], 'Gini Index': [0.985, 0.8344]}), (14, {'Majority Error': [0.9762, 0.8402], 'Entropy': [0.985, 0.8392], 'Gini Index': [0.985, 0.8344]}), (15, {'Majority Error': [0.9812, 0.8386], 'Entropy': [0.985, 0.8392], 'Gini Index': [0.985, 0.8344]}), (16, {'Majority Error': [0.985, 0.84], 'Entropy': [0.985, 0.8392], 'Gini Index': [0.985, 0.8344]})]

Unknown is treated as Missing Data

[(1, {'Majority Error': [0.8912, 0.8834], 'Entropy': [0.8808, 0.8752], 'Gini Index': [0.8912, 0.8834]}), (2, {'Majority Error': [0.895, 0.8898], 'Entropy': [0.894, 0.8886], 'Gini Index': [0.8948, 0.8896]}), (3, {'Majority Error': [0.9024, 0.8848], 'Entropy': [0.8974, 0.8902], 'Gini Index': [0.899, 0.8914]}), (4, {'Majority Error': [0.9134, 0.882], 'Entropy': [0.9122, 0.8772], 'Gini Index': [0.911, 0.8776]}), (5, {'Majority Error': [0.9202, 0.883], 'Entropy': [0.93, 0.8702], 'Gini Index': [0.9274, 0.871]}), (6, {'Majority Error': [0.9248, 0.8792], 'Entropy': [0.9424, 0.8632], 'Gini Index': [0.9424, 0.8664]}), (7, {'Majority Error': [0.9292, 0.8724], 'Entropy': [0.9474, 0.8502], 'Gini Index': [0.9496, 0.847]}), (8, {'Majority Error': [0.937, 0.8684], 'Entropy': [0.9558, 0.848], 'Gini Index': [0.957, 0.8446]}), (9, {'Majority Error': [0.9426, 0.8632], 'Entropy': [0.9638, 0.8392], 'Gini Index': [0.9646, 0.8404]}), (10, {'Majority Error': [0.9486, 0.8544], 'Entropy': [0.9706, 0.8408], 'Gini Index': [0.9716, 0.8386]}), (11, {'Majority Error': [0.9564, 0.8496], 'Entropy': [0.9764, 0.8404], 'Gini Index': [0.9764, 0.8382]}), (12, {'Majority Error': [0.9592, 0.8444], 'Entropy': [0.9784, 0.8402], 'Gini Index': [0.9784, 0.8384]}), (13, {'Majority Error': [0.9628, 0.8396], 'Entropy': [0.9788, 0.8402], 'Gini Index': [0.9788, 0.8384]}), (14, {'Majority Error': [0.9678, 0.8338], 'Entropy': [0.9788, 0.8402], 'Gini Index': [0.9788, 0.8384]}), (15, {'Majority Error': [0.9736, 0.8316], 'Entropy': [0.9788, 0.8402], 'Gini Index': [0.9788, 0.8384]}), (16, {'Majority Error': [0.9788, 0.8332], 'Entropy': [0.9788, 0.8402], 'Gini Index': [0.9788, 0.8384]})]
