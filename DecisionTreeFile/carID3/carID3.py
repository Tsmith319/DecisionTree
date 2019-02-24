#Taylor Smith
#U0741392

import math

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

# Creates a data set, list of labels, and a dictionary of the each attributes' values based upon the given file name.
def openFile(filename, attributes):
	f = open(filename, 'r')

	dataSet = list()
	labels = list()

	index = 0
	for line in f:
		terms = line.strip().split(',')
		count = 0;
		currExample = {}
		currLabel = None
		for term in terms:
			if count < 6:
				currExample[attributes[count]] = term
			else:
				currLabel = term
			count = count + 1
			
		dataSet.append(currExample)
		labels.insert(index, currLabel)
		index = index + 1
	return (dataSet, labels)

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
def ID3(S, Attributes, attributeValues, Labels, currentDepth, maxDepth, informationGain):
	
	allLabelsEqual = checkAllLabelsAreEqual(Labels)
	if allLabelsEqual["equal"] == "true":
		return Node(None, currentDepth, allLabelsEqual["label"])
	
	if len(Attributes) == 0:
		return Node(None, currentDepth, mostCommonLabel(Labels))

	if currentDepth == maxDepth:
		return Node(None, currentDepth, mostCommonLabel(Labels))

	rootNode = Node(None, currentDepth, None)

	if informationGain == 0:
		splitAttribute = calculateSplitValueWithEntropy(S, Attributes, attributeValues, Labels)
	if informationGain == 1:
		splitAttribute = calculateSplitValueWithMajorityError(S, Attributes, attributeValues, Labels)
	if informationGain == 2:	
		splitAttribute = calculateSplitValueWithGiniIndex(S, Attributes, attributeValues, Labels)

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
			branchNode = ID3(subset, newAttributes, attributeValues, subListLabels, currentDepth + 1, maxDepth, informationGain)
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
			for example in S:
				if example[attribute] == attributeVal:
					indexOfLabel = S.index(example)
					currLabel = labels[indexOfLabel]
					if currLabel in attributeValLabelCounts:
						attributeValLabelCounts[currLabel] = attributeValLabelCounts[currLabel] + 1
					else:
						attributeValLabelCounts[currLabel] = 1

			totalLabelCount = 0
			attributeValGiniIndex = 0
			for label in attributeValLabelCounts.keys():
				totalLabelCount += attributeValLabelCounts[label]
			for value in attributeValLabelCounts.values():
				attributeValGiniIndex += (float(value)/totalLabelCount)**2
			totalAttributeGiniIndex += ((1 - attributeValGiniIndex) * (abs(float(totalLabelCount)) / abs(len(S))))
		attributeGiniIndex[attribute] = totalGiniIndex - totalAttributeGiniIndex

	bestGiniIndex = 0
	splitAttribute = None
	for attribute in attributeGiniIndex.keys():
		if attributeGiniIndex[attribute] > bestGiniIndex:
			bestGiniIndex = attributeGiniIndex[attribute]
			splitAttribute = attribute

	return splitAttribute

# Will go through each example given in S and recursively traverse the tree and find whether each label matches the out come of the tree and return the accuracy of
# the number of correctly predicted labels over the total number of labels.
def makePrediction(S, Attributes, Tree, labels):

	labelOutcomes = list()
	for example in S:
		traverseTree(example, Attributes, Tree, labelOutcomes)

	count = 0
	correctCount = 0
	for label in labelOutcomes:
		if label == labels[count]:
			correctCount += 1
		count += 1

	accuracy = float(correctCount) / len(labels)
	return accuracy

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

# Main function that gathers all the data in train.csv and test.csv and converts it to a data set, and a list of labels. It will then run the ID3 algorithm on 
#training data with tree depths ranging from 1 to 6 and will run this algorithm and will construct a tree with current given depth. Once the algorithm has completed 
# building the tree, the program will use the makePredictions method and pass in the training and testing data sets in order to find the number of correct predictions 
#over the total number of labels given in the data set. And will print out the accuracies for each splitting attribute approach (entropy, majority error, and gini index)
# along with the corresponding depth of the tree. 
def main():

	attributes = ["buying","maint","doors","persons","lug_boot","safety"]

	attributeDict = {
		"buying": ["vhigh", "high", "med", "low"], 
		"maint": ["vhigh", "high", "med", "low"], 
		"doors": ["2", "3", "4", "5more"], 
		"persons": ["2", "4", "more"], 
		"lug_boot": ["small", "med", "big"], 
		"safety": ["low", "med", "high"]
	}

	treeDepths = {}
	currMaxDepth = 1
	while currMaxDepth < 7:
		informationGainType = 0
		accuracyDict = {}
		while informationGainType < 3:
			accuracyList = list()
			informationGainKey = None
			if informationGainType == 0:
				informationGainKey = "Entropy"
			if informationGainType == 1:
				informationGainKey = "Majority Error"
			if informationGainType == 2:
				informationGainKey = "Gini Index"
			accuracyDict[informationGainKey] = accuracyList

			trainingDataSet, trainingLabels = openFile("train.csv", attributes)
			node = ID3(trainingDataSet, attributes, attributeDict, trainingLabels, 0, currMaxDepth, informationGainType)
			trainingAccuracy = makePrediction(trainingDataSet, attributes, node, trainingLabels)
			accuracyDict[informationGainKey].append(trainingAccuracy)

			testingDataSet, testingLabels = openFile("test.csv", attributes)
			testingAccuracy = makePrediction(testingDataSet, attributes, node, testingLabels)
			accuracyDict[informationGainKey].append(testingAccuracy)
			informationGainType += 1

		treeDepths[currMaxDepth] = accuracyDict
		currMaxDepth += 1
	
	for tree in treeDepths.keys():
		print("Current Tree Depth: " + str(tree))
		print(treeDepths[tree])

if __name__=="__main__":
	main()