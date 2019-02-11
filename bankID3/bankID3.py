#Taylor Smith
#U0741392

import math
import copy

class Node:
	def __init__(self, splitVal, treeDepth, label):
		self.splitVal = splitVal
		self.treeDepth = treeDepth
		self.branches = {}
		self.label = label

	def addBranch(self, branchName, childNode):
		self.branches[branchName] = childNode

def openFile(filename, attributes, attributeDict, unknownType):
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
				if term != "unknown" and unknownType == 1:
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

def traverseTree(example, Attributes, Tree, labelOutcomes):

	if (len(Tree.branches) == 0) or (len(Attributes) == 0):
		labelOutcomes.append(Tree.label)
		return labelOutcomes

	
	currBranch = example[Tree.splitVal]
	if currBranch in Tree.branches.keys():
		newTree = Tree.branches[currBranch]
		newAttributes = list(Attributes)
		newAttributes.remove(Tree.splitVal)
		traverseTree(example, newAttributes, newTree, labelOutcomes)
	else:
		if(len(labelOutcomes) != 0):
			labelOutcomes.append(mostCommonLabel(labelOutcomes))
		else: 
			labelOutcomes.append("no")
		return labelOutcomes

def main():

	unknownAttributes = ["job","education","contact","poutcome"]

	attributes = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome"]

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

	
	unknownType = 0
	while unknownType < 2:
		trainingDataSet, trainingLabels, trainingAttributeDict = openFile("train.csv", attributes, attributeDict, unknownType)
		testingDataSet, testingLabels, testingAttributeDict = openFile("test.csv", attributes, attributeDict, unknownType)
		if unknownType == 1:
			print("Unknown is treated as Missing Data")
			trainingDataSet = findUnknownAttributeValues(trainingDataSet, unknownAttributes)
			testingDataSet = findUnknownAttributeValues(testingDataSet, unknownAttributes)
		else:
			print("Unknown is treated as Data")
		treeDepths = {}
		currMaxDepth = 1
		while currMaxDepth < 17:
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

				node = ID3(trainingDataSet, attributes, trainingAttributeDict, trainingLabels, 0, currMaxDepth, informationGainType)
				trainingAccuracy = makePrediction(trainingDataSet, attributes, node, trainingLabels)
				accuracyDict[informationGainKey].append(trainingAccuracy)

				testingAccuracy = makePrediction(testingDataSet, attributes, node, testingLabels)
				accuracyDict[informationGainKey].append(testingAccuracy)
				informationGainType += 1

			treeDepths[currMaxDepth] = accuracyDict
			currMaxDepth += 1
		print(treeDepths.items())
		unknownType += 1

if __name__=="__main__":
	main()