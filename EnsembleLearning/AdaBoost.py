#Taylor Smith
#U0741392

import math
import copy
import sys

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

	def printTree(self, indent, last):
		print(indent),
		if(bool(last) == bool(True)):
			print("\\-"),
			indent += "  "
		else:
			print("|-"),
			indent += "|    "

		if(self.splitVal == None):
			print(self.label)
		if(self.label == None):
			print(self.splitVal)
		# print(self.treeDepth)
		count = 0
		for branch in self.branches.keys():
			currBranch = self.branches[branch]
			currBranch.printTree(indent, bool(count == (len(self.branches.keys()) - 1)))
			count += 1

# Creates a data set, list of labels, and a dictionary of the each attributes' values based upon the given file name. If unknownType = 1 then we treat any unknown attribute values as missing data otherwise
# they are treated as part of the data set. 
def openFile(filename, attributes, attributeDict, numericalAttributes, unknownType, initWeights):
	f = open(filename, 'r')

	dataSet = list()
	labels = list()
	weights = list()
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
				if term == "yes":
					currLabel = 1
				else:
					currLabel = -1
			count = count + 1
			
		dataSet.append(currExample)
		labels.insert(index, currLabel)
		weights.insert(index, float(1)/initWeights)
		index = index + 1
	return (dataSet, labels, attributeValDict, weights)

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

#Finds the most common label of the current data set.
def mostCommonLabel(labels, weights):

	labelDictionary = {}
	index = 0
	for label in labels:
		if label in labelDictionary:
			labelDictionary[label] = float(labelDictionary[label] + weights[index])
		else:
			labelDictionary[label] = float(weights[index])
		index += 1

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
def ID3(S, Attributes, attributeValues, Labels, currentDepth, maxDepth, informationGain, weights):
	
	allLabelsEqual = checkAllLabelsAreEqual(Labels)
	if allLabelsEqual["equal"] == "true":
		return Node(None, currentDepth, allLabelsEqual["label"])
	
	if len(Attributes) == 0:
		return Node(None, currentDepth, mostCommonLabel(Labels, weights))

	if currentDepth == maxDepth:
		return Node(None, currentDepth, mostCommonLabel(Labels, weights))

	rootNode = Node(None, currentDepth, None)

	if informationGain == 0:
		splitAttribute = calculateSplitValueWithGiniIndex(S, Attributes, attributeValues, Labels, weights)

	rootNode.splitVal = splitAttribute

	currA = attributeValues[splitAttribute]

	for val in currA:
		branchNode = Node(None, None, None)
		subset, subListLabels, subListWeights = createSubset(S, splitAttribute, val, Labels, weights)

		if len(subset) == 0:
			branchNode = Node(None, currentDepth, mostCommonLabel(Labels, weights))
		else:
			newAttributes = list(Attributes)
			newAttributes.remove(splitAttribute)
			branchNode = ID3(subset, newAttributes, attributeValues, subListLabels, currentDepth + 1, maxDepth, informationGain, subListWeights)
		rootNode.addBranch(val, branchNode)

	return rootNode

#Creates a subset of S based upon the attribute splitting value "val"
def createSubset(S, splitVal, val, labels, weights):
	subset = []
	
	copyOfLabels = list()
	copyOfWeights = list()
	index = 0
	for example in S:
		if example[splitVal] == val:
			copyOfExample = dict(example)
			subset.append(copyOfExample)
			copyOfLabels.append(labels[index])
			copyOfWeights.append(weights[index])
		index += 1

	return (subset, copyOfLabels, copyOfWeights)

# Finds the best splitting attribute based upon the given data set S using entropy approach.

def calculateSplitValueWithGiniIndex(S, attributes, attributeValues, labels, weights):

	labelCounts = {}
	index = 0
	for label in labels:
		if label in labelCounts:
			labelCounts[label] = float(labelCounts[label] + weights[index])
		else:
			labelCounts[label] = weights[index]

	totalWeightCount = 0;
	for weight in weights:
		totalWeightCount += weight


	totalGiniIndex = 0
	for count in labelCounts.values():
		totalGiniIndex += (float(count)/totalWeightCount)**2
	totalGiniIndex = 1 - totalGiniIndex

	attributeGiniIndex = {}
	for attribute in attributes:
		totalAttributeGiniIndex = 0
		for attributeVal in attributeValues[attribute]:
			attributeValLabelCounts = {}
			index = 0
			totalAttributeValWeightCount = 0
			for example in S:
				if example[attribute] == attributeVal:
					currLabel = labels[index]
					if currLabel in attributeValLabelCounts:
						attributeValLabelCounts[currLabel] = attributeValLabelCounts[currLabel] + weights[index]
					else:
						attributeValLabelCounts[currLabel] = 1
				totalAttributeValWeightCount += weights[index]
				index += 1
				

			attributeValGiniIndex = 0
			for value in attributeValLabelCounts.values():
				attributeValGiniIndex += (float(value)/totalAttributeValWeightCount)**2
			totalAttributeGiniIndex += ((1 - attributeValGiniIndex) * (abs(float(totalAttributeValWeightCount)) / abs(totalWeightCount)))
		attributeGiniIndex[attribute] = totalGiniIndex - totalAttributeGiniIndex

	bestGiniIndex = sys.maxsize
	splitAttribute = None
	for attribute in attributeGiniIndex.keys():
		if attributeGiniIndex[attribute] < bestGiniIndex:
			bestGiniIndex = attributeGiniIndex[attribute]
			splitAttribute = attribute

	return splitAttribute



# Will go through each example given in S and recursively traverse the tree and find whether each label matches the out come of the tree and return the accuracy of
# the number of correctly predicted labels over the total number of labels.
def makePrediction(S, Attributes, Tree, labels, weights):

	labelOutcomes = list()
	for example in S:
		labelOutcomes = traverseTree(example, Attributes, Tree, labelOutcomes)

	count = 0
	correctCount = 0
	totalError = 0
	errorCount = 0
	for label in labelOutcomes:
		if label != labels[count]:
			totalError += (weights[count])
			errorCount += float(weights[count])
		count += 1

	totalCount = 0
	for weight in weights:
		totalCount += weight

	totalError = float(totalError)/ totalCount

	alpha = (float(1)/2)*float(math.log(float(1-totalError)/float(totalError)))
	
	#Calculate new weight
	count = 0
	normalizedWeights = 0
	newWeights = list()
	for label in labelOutcomes:
		newWeight = float(weights[count] * math.exp(-1 * alpha * label * labels[count]))
		normalizedWeights += newWeight
		newWeights.insert(count, newWeight)
		count += 1

	for index in range(0, len(newWeights)):
		weight = float(newWeights[index])/normalizedWeights
		newWeights.insert(index, weight)

	accuracy = float(errorCount)/(float(len(S) + totalCount))
	return (accuracy, newWeights, totalError)

# Takes in an example of the data set and recursively traverses the tree to its leaf nodes.
def traverseTree(example, Attributes, Tree, labelOutcomes):

	if (len(Tree.branches) == 0) or (len(Attributes) == 0):
		labelOutcomes.append(Tree.label)
		return labelOutcomes

	
	currBranch = example[Tree.splitVal]
	newTree = Tree.branches[currBranch]
	newAttributes = list(Attributes)
	newAttributes.remove(Tree.splitVal)
	traverseTree(example, newAttributes, newTree, labelOutcomes)

	return labelOutcomes

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
	trainingDataSet, trainingLabels, trainingAttributeDict, trainingWeights = openFile("train.csv", attributes, attributeDict, numericalAttributes, unknownType, 5000)
	trainingDataSet = findNumericalAttributeValues(trainingDataSet, numericalAttributes)
	testingDataSet, testingLabels, testingAttributeDict, testWeights = openFile("test.csv", attributes, attributeDict, numericalAttributes, unknownType, 5000)
	testingDataSet = findNumericalAttributeValues(testingDataSet, numericalAttributes)
	print("TrainingAccuracy")
	trainingWeightsTemp = list(trainingWeights)
	finalHypothesisTrainingData = {}
	treeDepths = {}
	errors = {}
	for T in range(0,1001):
		print("Unknown is treated as Data")
		print("Iteration: " + str(T))

		currMaxDepth = 1

		informationGainType = 0
		accuracyDict = {}
		errorDict = {}
		accuracyList = list()
		informationGainKey = None

		accuracyDict[T] = accuracyList
		errorDict[T] = []

		node = ID3(trainingDataSet, attributes, trainingAttributeDict, trainingLabels, 0, currMaxDepth, unknownType, trainingWeights)
		trainingError, newWeights, trainingET = makePrediction(trainingDataSet, attributes, node, trainingLabels, trainingWeights)
		if(T % 50 == 0):
			accuracyDict[T].append(trainingError)
			errorDict[T].append(trainingET)
		trainingWeights = newWeights

		testingError, newWeights, testingET = makePrediction(testingDataSet, attributes, node, testingLabels, trainingWeights)
		if(T % 50 == 0):
			accuracyDict[T].append(testingError)
			errorDict[T].append(testingET)
			treeDepths[T] = accuracyDict
			errors[T] = errorDict
		if(T % 50 == 0):
			print(accuracyDict.items())
			print(errorDict.items())
			node.printTree("", bool(True))

		#Prints the entire tree for each depth displaying training and testing accuracies for the tree using majority error, entropy,
		# and gini index.
	for tree in treeDepths.keys():
		print("Decision Stumps Training and Testing Errors:")
		print(treeDepths[tree])
		print("Training and Testing Errors:")
		print(errors[tree])

if __name__=="__main__":
	main()