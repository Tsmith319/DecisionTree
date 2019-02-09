import math

dataSet = list()
attributes = ["buying","maint","doors","persons","lug_boot","safety"]

attributeDict = {
	"buying": ["vhigh", "high", "med", "low"], 
	"maint": ["vhigh", "high", "med", "low"], 
	"doors": ["2", "3", "4", "5more"], 
	"persons": ["2", "4", "more"], 
	"lug_boot": ["small", "med", "big"], 
	"safety": ["low", "med", "high"]
}

labels = list()

class Node:
	def __init__(self, splitVal, treeDepth, label):
		self.splitVal = splitVal
		self.treeDepth = treeDepth
		self.branches = {}
		self.label = label

	def addBranch(self, branchName, childNode):
		self.branches[branchName] = childNode

	def printPretty(self, indent, last):
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

		count = 0
		for branch in self.branches.keys():
			currBranch = self.branches[branch]
			currBranch.printPretty(indent, bool(count == (len(self.branches.keys()) - 1)))
			count += 1

def openFile(filename):
	f = open(filename, 'r')

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

def ID3(S, Attributes, attributeValues, Labels, currentDepth):
	
	allLabelsEqual = checkAllLabelsAreEqual(Labels)
	if allLabelsEqual["equal"] == "true":
		return Node(None, currentDepth, allLabelsEqual["label"])
	
	if len(Attributes) == 0:
		return Node(None, currentDepth, mostCommonLabel(Labels))

	rootNode = Node(None, currentDepth, None)

	#splitAttribute = calculateSplitValueWithEntropy(S, Attributes, attributeValues, Labels)

	# splitAttribute = calculateSplitValueWithMajorityError(S, Attributes, attributeValues, Labels)
	# print(str(splitAttribute))

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
			branchNode = ID3(subset, newAttributes, attributeValues, subListLabels, currentDepth + 1)
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
			totalAttributeEntropy += (attributeValEntropy * (float(totalLabelCount)/len(S)))
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
	print("total ME")
	print(totalMajorityError)
	print("Attributes")
	print(attributes)
	print("labels")
	print(labels)
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

			print("Counts")
			print(attributeValLabelCounts.items())
			print(totalLabelCount)
			if totalLabelCount != 0:
				attributeValMajorityError = float(min(attributeValLabelCounts.values()))/totalLabelCount
			else:
				attributeValMajorityError = 0
				
			print(attributeValMajorityError)
			totalAttributeMajorityError += (attributeValMajorityError * (float(totalLabelCount) / len(S)))
			print("Total attr ME: ")
			print(totalAttributeMajorityError)

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
	print("Total Counts")
	print(attributeMajorityErrors.items())

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
			totalAttributeGiniIndex += ((1 - attributeValGiniIndex) * (float(totalLabelCount) / len(S)))
		attributeGiniIndex[attribute] = totalGiniIndex - totalAttributeGiniIndex

	bestGiniIndex = 0
	splitAttribute = None
	for attribute in attributeGiniIndex.keys():
		if attributeGiniIndex[attribute] > bestGiniIndex:
			bestGiniIndex = attributeGiniIndex[attribute]
			splitAttribute = attribute

	return splitAttribute

openFile("train.csv")
node = ID3(dataSet, attributes, attributeDict, labels, 0)
print(bool(True))
node.printPretty("", bool(True))

for branch in node.branches.keys():
	print("branch name: " + branch)
	print(str(node.branches[branch].splitVal))

