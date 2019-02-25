#Taylor Smith
#U0741392

import math
import random
import copy

# Node Class that contains the nodes of the decision tree such as its children nodes. If the node is a root node it will have a label and its split val is
# None and if it is a node of the tree or a root node, label = None and splitVal = the name of the current split in the node. The tree also contains the current
# tree depth of the node. 
class Node:
	def __init__(self, splitVal, treeDepth, label):
		self.splitVal = splitVal
		self.treeDepth = treeDepth
		self.branches = {}
		self.label = label

	def addBranch(self, branchName, childNode):
		self.branches[branchName] = childNode

# Creates a data set, list of labels, and a dictionary of the each attributes' values based upon the given file name. If unknownType = 1 then we treat any unknown attribute values as missing data otherwise
# they are treated as part of the data set. 
def openFile(filename, attributes, attributeDict, numericalAttributes, unknownType):
	f = open(filename, 'r')

	dataSet = list()
	labels = list()
	attributeValDict = copy.deepcopy(attributeDict)

	index = 0
	for line in f:
		terms = line.strip().split(',')
		count = 0;
		currExample = {}
		currLabel = None
		for term in terms:
			if count < 16:
				currExample[attributes[count]] = term

				if attributes[count] in numericalAttributes:
					attributeValDict[attributes[count]].add("yes")
					attributeValDict[attributes[count]].add("no")
				elif term != "unknown" and unknownType == 1:
					attributeValDict[attributes[count]].add(term)
				else:
					attributeValDict[attributes[count]].add(term)
			else:
				currLabel = term
			count = count + 1
			
		dataSet.append(currExample)
		labels.insert(index, currLabel)
		index = index + 1
	return (dataSet, labels, attributeValDict)

# Converts all of the numberical attribute values to binary and replaces the numerical attribute value with binary values in the dataset S. 
def findNumericalAttributeValues(S, numericalAttributes):
	attributeValCounts = {}
	for example in S:
		for numerical in numericalAttributes:
			if numerical in attributeValCounts.keys():
				attributeValCounts[numerical].append(int(example[numerical]))
			else:
				attributeValCounts[numerical] = list()
				attributeValCounts[numerical].append(int(example[numerical]))

	attributeThresholds = {}
	for attribute in attributeValCounts.keys():
		sortedList = attributeValCounts[attribute]
		sortedList.sort()
		medianIndex = len(sortedList)/2
		attributeThresholds[attribute] = sortedList[medianIndex]

	for example in S:
		for attribute in numericalAttributes:
			if int(example[attribute]) > attributeThresholds[attribute]:
				example[attribute] = "yes"
			elif int(example[attribute]) <= attributeThresholds[attribute]:
				example[attribute] = "no"

	return S

#Finds the majority attribute value for the current unknown attribute value and replaces the unknown attribute value with the majority attribute value in the data set S.
def findUnknownAttributeValues(S, unknownAttributes):
	attributeValCounts = {}

	for example in S:
		for unknown in unknownAttributes:
			if example[unknown] != "unknown":
				attributeVal = example[unknown]
				if unknown in attributeValCounts:
					if attributeVal in attributeValCounts[unknown]:
						currVal = attributeValCounts[unknown]
						currVal[attributeVal] += 1
						attributeValCounts[unknown] = currVal
					else:
						currVal = attributeValCounts[unknown]
						currVal[attributeVal] = 0
						currVal[attributeVal] += 1
						attributeValCounts[unknown] = currVal
				else:
					attributeValCounts[unknown] = {}
					if attributeVal in attributeValCounts[unknown]:
						currVal = attributeValCounts[unknown]
						currVal[attributeVal] += 1
						attributeValCounts[unknown] = currVal
					else:
						currVal = attributeValCounts[unknown]
						currVal[attributeVal] = 0
						currVal[attributeVal] += 1
						attributeValCounts[unknown] = currVal

	majorityCounts = dict()
	for attribute in attributeValCounts.keys():
		currentMajorityCount = 0
		currAttributeValCounts = attributeValCounts[attribute]
		for val in currAttributeValCounts.keys():
			if currAttributeValCounts[val] > currentMajorityCount:
				currentMajorityCount = currAttributeValCounts[val]
				majorityCounts[attribute] = val
	

	for example in S:
		for unknown in unknownAttributes:
			if example[unknown] == "unknown":
				example[unknown] = majorityCounts[unknown]

	return S

#Finds the most common label of the current data set.
def mostCommonLabel(labels):

	labelDictionary = {}
	for label in labels:
		if label in labelDictionary:
			labelDictionary[label] = labelDictionary[label] + 1
		else:
			labelDictionary[label] = 1

	count = 0
	commonLabel = None
	for label in labelDictionary.keys():
		if count == 0:
			commonLabel = label
			count = count + 1
		elif labelDictionary[label] > labelDictionary[commonLabel]:
			commonLabel = label

	return commonLabel

#Goes through all the labels in the current data set to check if all the labels are equal.
def checkAllLabelsAreEqual(labels):
	count = 0
	label = None
	returnVal = {
		"label": label,
		"equal": None
	}

	for example in labels:
		if count == 0:
			label = example
			count = count + 1
			returnVal["label"] = label
		elif label != example:
			returnVal["equal"] = "false";
			return returnVal

	returnVal["equal"] = "true"	
	return returnVal

# Creates a decision tree that splits the data in data set S using the Attributes list, as well as their values in attributeValues. Using the labels as well to accurately split the data using 
# majority error, gini index, and entropy. Based on whether informationGain equalling 1, 2 , or 3 will call one of the splitting algorithm functions as well the user can set the tree depth of the tree
# by setting the maxDepth value when calling the function. 
def ID3(S, Attributes, attributeValues, Labels, currentDepth, maxDepth, subsetSize):
	
	allLabelsEqual = checkAllLabelsAreEqual(Labels)
	if allLabelsEqual["equal"] == "true":
		return Node(None, currentDepth, allLabelsEqual["label"])
	
	if len(Attributes) == 0:
		return Node(None, currentDepth, mostCommonLabel(Labels))

	if currentDepth == maxDepth:
		return Node(None, currentDepth, mostCommonLabel(Labels))

	rootNode = Node(None, currentDepth, None)

	subsetAttributes = []
	chosenAttributes = []
	if len(Attributes) >= subsetSize:
		while len(subsetAttributes) < subsetSize:
			randIndex = random.randint(0, len(Attributes)-1)
			if(randIndex not in chosenAttributes):
				subsetAttributes.append(Attributes[randIndex])
				chosenAttributes.append(randIndex)
	else:
		subsetAttributes = Attributes


	splitAttribute = calculateSplitValueWithGiniIndex(S, subsetAttributes, attributeValues, Labels)
	rootNode.splitVal = splitAttribute

	currA = attributeValues[splitAttribute]

	for val in currA:
		branchNode = Node(None, None, None)
		subset, subListLabels = createSubset(S, splitAttribute, val, Labels)

		if len(subset) == 0:
			branchNode = Node(None, currentDepth, mostCommonLabel(Labels))
		else:
			newAttributes = list(Attributes)
			newAttributes.remove(splitAttribute)
			branchNode = ID3(subset, newAttributes, attributeValues, subListLabels, currentDepth + 1, maxDepth, subsetSize)
		rootNode.addBranch(val, branchNode)

	return rootNode

#Creates a subset of S based upon the attribute splitting value "val"
def createSubset(S, splitVal, val, labels):
	subset = []
	
	copyOfLabels = list()
	for example in S:
		if example[splitVal] == val:
			copyOfExample = dict(example)
			indexOfLabel = S.index(example)
			subset.append(copyOfExample)
			copyOfLabels.append(labels[indexOfLabel])
	return (subset, copyOfLabels)

# Finds the best splitting attribute based upon the given data set S using entropy approach.
def calculateSplitValueWithEntropy(S, attributes, attributeValues, labels):

	labelCounts = {}
	for label in labels:
		if label in labelCounts:
			labelCounts[label] = labelCounts[label] + 1
		else:
			labelCounts[label] = 1

	totalEntropy = 0
	for label in labelCounts.keys():
		labelCounts[label] = float(labelCounts[label]) / len(S)
		totalEntropy += - ((labelCounts[label]) * (math.log(labelCounts[label], 2)))

	attributeEntropies = {}
	for attribute in attributes:
		totalAttributeEntropy = 0
		for attributeVal in attributeValues[attribute]:
			attributeValLabelCounts = {}
			for example in S:
				if example[attribute] == attributeVal:
					indexOfLabel = S.index(example)
					currLabel = labels[indexOfLabel]
					if currLabel in attributeValLabelCounts:
						attributeValLabelCounts[currLabel] = attributeValLabelCounts[currLabel] + 1
					else:
						attributeValLabelCounts[currLabel] = 1

			totalLabelCount = 0
			attributeValEntropy = 0
			for label in attributeValLabelCounts.keys():
				totalLabelCount += attributeValLabelCounts[label]
			for label in attributeValLabelCounts.keys():
				attributeValLabelCounts[label] = float(attributeValLabelCounts[label]) / totalLabelCount
				attributeValEntropy += (-((attributeValLabelCounts[label]) * (math.log(attributeValLabelCounts[label], 2))))
			totalAttributeEntropy += (attributeValEntropy * (abs(float(totalLabelCount))/abs(len(S))))
		attributeEntropies[attribute] = totalEntropy - totalAttributeEntropy

	bestEntropy = 0
	splitAttribute = None
	for attribute in attributeEntropies.keys():
		if attributeEntropies[attribute] > bestEntropy:
			bestEntropy = attributeEntropies[attribute]
			splitAttribute = attribute

	return splitAttribute

# Finds the best splitting value using the majority error approach based upon the given data set S
def calculateSplitValueWithMajorityError(S, attributes, attributeValues, labels):

	labelCounts = {}
	for label in labels:
		if label in labelCounts:
			labelCounts[label] = labelCounts[label] + 1
		else:
			labelCounts[label] = 1

	totalMajorityError = float(min(labelCounts.values()))/len(S)

	attributeMajorityErrors = {}
	for attribute in attributes:
		totalAttributeMajorityError = 0
		for attributeVal in attributeValues[attribute]:
			attributeValLabelCounts = {}
			for example in S:
				if example[attribute] == attributeVal:
					indexOfLabel = S.index(example)
					currLabel = labels[indexOfLabel]
					if currLabel in attributeValLabelCounts:
						attributeValLabelCounts[currLabel] = attributeValLabelCounts[currLabel] + 1
					else:
						attributeValLabelCounts[currLabel] = 1

			totalLabelCount = 0
			for label in attributeValLabelCounts.keys():
				totalLabelCount += attributeValLabelCounts[label]

			if totalLabelCount != 0:
				attributeValMajorityError = 1 - (float(max(attributeValLabelCounts.values()))/totalLabelCount)
			else:
				attributeValMajorityError = 0

			totalAttributeMajorityError += (attributeValMajorityError * (abs(float(totalLabelCount)) / abs(len(S))))

		if((totalMajorityError - totalAttributeMajorityError) < 0):
			attributeMajorityErrors[attribute] = 0
		else:
			attributeMajorityErrors[attribute] = totalMajorityError - totalAttributeMajorityError

	bestMajorityError = 0
	splitAttribute = None
	count = 0
	for attribute in attributeMajorityErrors.keys():
		if count == 0:
			bestMajorityError = attributeMajorityErrors[attribute]
			splitAttribute = attribute
			count += 1
		elif attributeMajorityErrors[attribute] > bestMajorityError:
			bestMajorityError = attributeMajorityErrors[attribute]
			splitAttribute = attribute

	return splitAttribute

# Finds the best splitting value by using the gini index approach based upon the given data set S
def calculateSplitValueWithGiniIndex(S, attributes, attributeValues, labels):

	labelCounts = {}
	for label in labels:
		if label in labelCounts:
			labelCounts[label] = labelCounts[label] + 1
		else:
			labelCounts[label] = 1


	totalGiniIndex = 0
	for count in labelCounts.values():
		totalGiniIndex += (float(count)/len(S))**2
	totalGiniIndex = 1 - totalGiniIndex

	attributeGiniIndex = {}
	for attribute in attributes:
		totalAttributeGiniIndex = 0
		for attributeVal in attributeValues[attribute]:
			attributeValLabelCounts = {}
			index = 0
			for example in S:
				if example[attribute] == attributeVal:
					currLabel = labels[index]
					if currLabel in attributeValLabelCounts:
						attributeValLabelCounts[currLabel] = attributeValLabelCounts[currLabel] + 1
					else:
						attributeValLabelCounts[currLabel] = 1
				index += 1

			totalLabelCount = 0
			attributeValGiniIndex = 0
			for label in attributeValLabelCounts.keys():
				totalLabelCount += attributeValLabelCounts[label]
			for value in attributeValLabelCounts.values():
				attributeValGiniIndex += (float(value)/totalLabelCount)**2
			totalAttributeGiniIndex += ((float(1) - attributeValGiniIndex) * (abs(float(totalLabelCount)) / abs(len(S))))
		attributeGiniIndex[attribute] = totalGiniIndex - totalAttributeGiniIndex

	bestGiniIndex = 0
	splitAttribute = None
	if max(attributeGiniIndex.values()) == 0.0:
		randomIndex = random.randint(0,len(attributes)-1)
		splitAttribute = attributes[randomIndex]
	elif min(attributeGiniIndex.values()) < 0.0:
		randomIndex = random.randint(0,len(attributes)-1)
		splitAttribute = attributes[randomIndex]
	else:
		for attribute in attributeGiniIndex.keys():
			if attributeGiniIndex[attribute] > bestGiniIndex:
				bestGiniIndex = attributeGiniIndex[attribute]
				splitAttribute = attribute
	if splitAttribute == None:
		print(attributeGiniIndex.items())
	return splitAttribute

# Will go through each example given in S and recursively traverse the tree and find whether each label matches the out come of the tree and return the accuracy of
# the number of correctly predicted labels over the total number of labels.
def calculateBiasTerm(S, Attributes, Trees, labels):


	singleTree = Trees[random.randint(0,len(Trees)-1)]


	biasTerms = list()
	varianceTerms = list()
	averages = list()
	predictedValues = list()
	for example in S:
		predictionCounts = {}
		totalCount = 0
		for tree in Trees:
			currLabel = traverseTree(example, Attributes, tree)
			if(currLabel == "yes"):
				totalCount += 1
		predicttedTreeLabel = traverseTree(example, Attributes, tree)
		if(predicttedTreeLabel == "yes"):
			predictedValues.append(1)
		else:
			predictedValues.append(0)

		average = float(totalCount)/len(Trees)
		averages.append(average)

	index = 0
	for label in labels:
		val = 0
		if label == "yes":
			val = 1
		biasVal = (float(val) - averages[index])**2
		biasTerms.append(biasVal)
		varianceTerm = (float(averages[index]) - predictedValues[index])**2
		varianceTerms.append(varianceTerm)
		index += 1

	totalBiasCount = 0
	for bias in biasTerms:
		totalBiasCount += bias

	totalVarianceCount = 0
	for variance in varianceTerms:
		totalVarianceCount += variance


	generalBias = float(totalBiasCount)/len(biasTerms)
	generalVariance = float(totalVarianceCount)/len(varianceTerms)
	generalError = generalBias + generalVariance

	return (generalBias, generalVariance, generalError)

def calculateTerms(S, Attributes, Tree, labels):

	biasTerms = list()
	varianceTerms = list()
	averages = list()
	predictedValues = list()
	for example in S:
		totalCount = 0
		for i in range(0,100):
			currLabel = traverseTree(example, Attributes, Tree)
			if(currLabel == "yes"):
				totalCount += 1
		predicttedTreeLabel = traverseTree(example, Attributes, Tree)
		if(predicttedTreeLabel == "yes"):
			predictedValues.append(1)
		else:
			predictedValues.append(0)

		average = float(totalCount)/100
		averages.append(average)

	index = 0
	for label in labels:
		val = 0
		if label == "yes":
			val = 1
		biasVal = (float(val) - averages[index])**2
		biasTerms.append(biasVal)
		varianceTerm = (float(averages[index]) - predictedValues[index])**2
		varianceTerms.append(varianceTerm)
		index += 1

	totalBiasCount = 0
	for bias in biasTerms:
		totalBiasCount += bias

	totalVarianceCount = 0
	for variance in varianceTerms:
		totalVarianceCount += variance


	generalBias = float(totalBiasCount)/len(biasTerms)
	generalVariance = float(totalVarianceCount)/len(varianceTerms)
	generalError = generalBias + generalVariance

	return (generalBias, generalVariance, generalError)

# Takes in an example of the data set and recursively traverses the tree to its leaf nodes.
def traverseTree(example, Attributes, Tree):

	if (len(Tree.branches) == 0) or (len(Attributes) == 0):
		return Tree.label

	
	currBranch = example[Tree.splitVal]
	newTree = Tree.branches[currBranch]
	newAttributes = list(Attributes)
	newAttributes.remove(Tree.splitVal)
	label = traverseTree(example, newAttributes, newTree)

	return label

# Main function that gathers all the data in train.csv and test.csv and converts any missing data if necessary to a certain attribute value as well
# as change any numerical data to binary if necessary. Will run ID3 algorithm on training data with tree depths ranging from 1 to 16 and will run this algorithm
# twice, one for having the unknown attribute value as part of the data set and the other as treating unknown as being missing. Once completed building the tree
# using the makePredictions method we will pass in the training and testing data sets and find the number of correct predictions over the total number of labels 
# given in the data set.
def main():
	# List of attributes that contain numerical data that needs to be converted to binary
	numericalAttributes = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]

	# List of attributes that are known to have unknown attribute values
	unknownAttributes = ["job","education","contact","poutcome"]

	attributes = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]

	#Will contain all of the attributes' values for each attribute
	attributeDict = {
		"age": set(),
		"job": set(),
		"marital": set(),
		"education": set(),
		"default": set(),
		"balance": set(),
		"housing": set(),
		"loan": set(),
		"contact": set(),
		"day": set(),
		"month": set(),
		"duration": set(),
		"campaign": set(),
		"pdays": set(),
		"previous": set(),
		"poutcome": set()
	}

	
	# Runs the ID3 algorithm twice for treating unknown attribute value as missing and as part of the data set by first opening 
	#the training and testing files to create the data set and runs the algorithm for tree depths between 1 and 17 and splitting 
	#the data set using majority error, entropy, and gini index for each depth.
	unknownType = 0

	trainingDataSet, trainingLabels, trainingAttributeDict = openFile("train.csv", attributes, attributeDict, numericalAttributes, unknownType)
	trainingDataSet = findNumericalAttributeValues(trainingDataSet, numericalAttributes)
	testingDataSet, testingLabels, testingAttributeDict = openFile("test.csv", attributes, attributeDict, numericalAttributes, unknownType)
	testingDataSet = findNumericalAttributeValues(testingDataSet, numericalAttributes)
	subsetData = {}
	subsetSize = 4
	while subsetSize <= 6:
		print("SubsetSize: " + str(subsetSize))
		trees = []
		treeDepths = {}
		for tree in range(0,1000):
			sample = []
			sampleLabels = []
			while len(sample) != 1000:
				randomIndex = random.randint(0,4999)
				sample.append(trainingDataSet[randomIndex])
				sampleLabels.append(trainingLabels[randomIndex])

			print("Iteration: " + str(tree))
			
			currMaxDepth = 16
			node = ID3(sample, attributes, trainingAttributeDict, sampleLabels, 0, currMaxDepth, subsetSize)
			trees.append(node)

		subsetData[subsetSize] = trees
		subsetSize += 2

	forest = subsetData[6]
	trainingGeneralBias, trainingGeneralVariance, trainingGeneralError = calculateBiasTerm(trainingDataSet, attributes, forest, trainingLabels)
	testingGeneralBias, testingGeneralVariance, testingGeneralError = calculateBiasTerm(testingDataSet, attributes, forest, testingLabels)
	print("General Bias, General Variance, and General Error for Training Data on predictors: ")
	print("Bias: " + str(trainingGeneralBias) + " Variance: " + str(trainingGeneralVariance) + " General Error: " + str(trainingGeneralError))
	print("General Bias, General Variance, and General Error for Testing Data on predictors: ")
	print("Bias: " + str(testingGeneralBias) + " Variance: " + str(testingGeneralVariance) + " General Error: " + str(testingGeneralError))

	singleTree = forest[random.randint(0,len(forest)-1)]
	trainingSingleBias, trainingSingleVariance, trainingSingleGeneralError = calculateTerms(trainingDataSet, attributes, singleTree, trainingLabels)
	testingSingleBias, testingSingleVariance, testingSingleGeneralError = calculateTerms(testingDataSet, attributes, singleTree, testingLabels)
	print("General Bias, General Variance, and General Error for Training Data on single tree: ")
	print("Bias: " + str(trainingSingleBias) + " Variance: " + str(trainingSingleVariance) + " General Error: " + str(trainingSingleGeneralError))
	print("General Bias, General Variance, and General Error for Testing Data on single tree: ")
	print("Bias: " + str(testingSingleBias) + " Variance: " + str(testingSingleVariance) + " General Error: " + str(testingSingleGeneralError))



if __name__=="__main__":
	main()