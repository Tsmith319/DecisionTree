import math

dataSet = list()
listOfAttributes = ["buying","maint","doors","persons","lug_boot","safety"]

attributes = {
	"buying": ["vhigh", "high", "med", "low"], 
	"maint": ["vhigh", "high", "med", "low"], 
	"doors": ["2", "3", "4", "5more"], 
	"persons": ["2", "4", "more"], 
	"lug_boot": ["small", "med", "big"], 
	"safety": ["low", "med", "high"]
}

labels = ["unacc", "acc", "good", "vgood"]

class Node:
	def __init__(self, splitVal, treeDepth, label):
		self.splitVal = splitVal
		self.treeDepth = treeDepth
		self.branches = {}
		self.label = label

	def addBranch(self, branchName, childNode):
		self.branches[branchName] = childNode

def openFile(filename):
	f = open(filename, 'r')

	for line in f:
		terms = line.strip().split(',')
		count = 0;
		currExample = {}
		for term in terms:
			if count < 6:
				currExample[listOfAttributes[count]] = term
			else:
				currExample["label"] = term
			count = count + 1
		dataSet.append(currExample)



def ID3(S, Attributes, Label, treeDepth, currentDepth):
	
	allLabelsEqual = checkAllLabelsAreEqual(S)
	if allLabelsEqual["equal"] == "true":
		return Node(None, currentDepth, allLabelsEqual["label"])
	
	if len(Attributes) == 0:
		return Node(None, currentDepth, mostCommonLabel(S))

	rootNode = Node(None, currentDepth, None)

	splitAttribute = informationGain(S, Attributes, Label)
	print("Split on: " + splitAttribute)
	rootNode.splitVal = splitAttribute

	currA = Attributes[splitAttribute]

	for val in currA:
		print(str(val))
		branchNode = Node(None, None, None)
		valS = createSubset(S, splitAttribute, val)
	
		if len(valS) == 0:
			branchNode = Node(None, currentDepth, mostCommonLabel(S))
		else:
			newAttributes = dict(Attributes)
			newAttributes.pop(splitAttribute)
			print("Attributes: ")
			print(Attributes.items())
			print("new attributes: ")
			print(newAttributes.items())
			branchNode = ID3(valS, newAttributes, Label, treeDepth, currentDepth + 1)
		rootNode.addBranch(val, branchNode)

	return rootNode



def createSubset(S, splitVal, val):
	subset = []
	
	for example in S:
		if example[splitVal] == val:
			copyOfExample = dict(example)
			copyOfExample.pop(splitVal)
			subset.append(copyOfExample)
	return subset


def informationGain(S, Attributes, Label):
	print("current Data Set")
	print(S)
	targetLabelCount = 0
	oppositeLabelCount = 0
	for example in S:
		if example["label"] == Label:
			targetLabelCount = targetLabelCount + 1
		else:
			oppositeLabelCount = oppositeLabelCount + 1

	#Calculate Total Information Gain using Entropy
	targetLabelFraction = float(targetLabelCount)/len(S)
	oppositeLabelFraction = float(oppositeLabelCount)/len(S)
	totalEntropy = (-(targetLabelFraction) * (math.log(targetLabelFraction, 2))) - ((oppositeLabelFraction) * (math.log(oppositeLabelFraction, 2)))

	#Calculate Total Information Gain using Majority Error
	totalMajorityError = 0
	if oppositeLabelCount < targetLabelCount:
		totalMajorityError = float(oppositeLabelCount) / (targetLabelCount + oppositeLabelCount)
	elif oppositeLabelCount >= targetLabelCount:
		totalMajorityError = float(targetLabelCount) / (targetLabelCount + oppositeLabelCount)

	#Calculate Total Information Gain using Gini Index
	totalGiniIndex = 1 - ((float(targetLabelCount)/len(S))**2 + (float(oppositeLabelCount)/len(S))**2)

	attributeEntropies = {} #dictionary for attribute entropies
	attributeME = {} #dictionary for attribute Majority Error
	attributeGI = {} #dictionary for attribute Gini Index
	print("IG: Attributes: ")
	print(Attributes)
	for attribute in Attributes:
		totalAttributeEntropy = 0
		totalAttributeMajorityError = 0
		totalAttributeGiniIndex = 0
		print("ATTRI: " + attribute)
		for attributeVal in Attributes[attribute]:
			attributeTargetLabelCount = 0
			attributeOppositeLabelCount = 0
			print("attri: val: " + attributeVal)
			for example in S:
				if example[attribute] == attributeVal:
					if example["label"] == Label:
						attributeTargetLabelCount = attributeTargetLabelCount + 1
					else:
						attributeOppositeLabelCount = attributeOppositeLabelCount + 1

			attributeValFractionOfTotal = float(attributeTargetLabelCount + attributeOppositeLabelCount) / len(S)
			print("Attribute Total Fraction")
			print(attributeValFractionOfTotal)

			#calculate each individual attribute value entropy
			attributeTargetFraction = float(attributeTargetLabelCount)/ (attributeTargetLabelCount + attributeOppositeLabelCount)
			nonAttributeTargetFraction = float(attributeOppositeLabelCount)/ (attributeTargetLabelCount + attributeOppositeLabelCount)

			if ((nonAttributeTargetFraction == 0 and attributeTargetFraction == 1) or (nonAttributeTargetFraction == 1 and attributeTargetFraction == 0)):
				attributeEntropy = 0
			else:
				attributeEntropy = 	(-(attributeTargetFraction) * (math.log(attributeTargetFraction, 2))) - ((nonAttributeTargetFraction) * (math.log(nonAttributeTargetFraction, 2)))

			totalAttributeEntropy = totalAttributeEntropy + (attributeEntropy * (attributeValFractionOfTotal))

			#calculate each individual attribute value majority error
			attributeMajorityError = 0
			if attributeOppositeLabelCount < attributeTargetLabelCount:
				attributeMajorityError = float(attributeOppositeLabelCount) / (attributeTargetLabelCount + attributeOppositeLabelCount)
			elif attributeOppositeLabelCount >= attributeTargetLabelCount:
				attributeMajorityError = float(attributeTargetLabelCount) / (attributeTargetLabelCount + attributeOppositeLabelCount)

			totalAttributeMajorityError = totalAttributeMajorityError + (attributeMajorityError * attributeValFractionOfTotal)

			#calculate each individual attribute value gini index
			attributeGiniIndex = 1 - ((float(attributeTargetLabelCount)/(attributeTargetLabelCount + attributeOppositeLabelCount))**2 + (float(attributeOppositeLabelCount)/(attributeTargetLabelCount + attributeOppositeLabelCount))**2)
			totalAttributeGiniIndex = totalAttributeGiniIndex + (attributeGiniIndex * attributeValFractionOfTotal)

		
		#Calculate information gain using entropy
		attributeInformationGain = totalEntropy - totalAttributeEntropy
		attributeEntropies[attribute] = attributeInformationGain

		#Calculate information gain using Majority Error
		attributeInformationGainME = totalMajorityError - totalAttributeMajorityError

		if(attributeInformationGainME < 0):
			attributeInformationGainME = 0

		attributeME[attribute] = attributeInformationGainME

		#Calculate information gain using Gini Index
		attributeInformationGainGini = totalGiniIndex - totalAttributeGiniIndex
		attributeGI[attribute] = attributeInformationGainGini

	# Finding the best information gain using entropy
	bestSplitAttributeEntropy = None
	bestEntropy = 0
	count = 0
	for entropy in attributeEntropies.keys():
		if count == 0:
			bestSplitAttributeEntropy = entropy
			bestEntropy = attributeEntropies[entropy]
			count = count + 1
		elif attributeEntropies[entropy] > bestEntropy:
			bestEntropy = attributeEntropies[entropy]
			bestSplitAttributeEntropy = entropy


	# Finding the best information gain using majority error
	bestSplitAttributeME = None
	bestMajorityError = 0
	count = 0
	for currMajorityError in attributeME.keys():
		if count == 0:
			bestSplitAttributeME = currMajorityError
			bestMajorityError = attributeME[currMajorityError]
			count = count + 1
		elif attributeME[currMajorityError] > bestMajorityError:
			bestMajorityError = attributeME[currMajorityError]
			bestSplitAttributeME = currMajorityError


	# Finding the best information gain using gini index
	bestSplitAttributeGini = None
	bestGiniIndex = 0
	count = 0
	for currGiniIndex in attributeGI.keys():
		if count == 0:
			bestSplitAttributeGini = currGiniIndex
			bestGiniIndex - attributeGI[currGiniIndex]
			count = count + 1
		elif attributeGI[currGiniIndex] > bestGiniIndex:
			bestGiniIndex = attributeGI[currGiniIndex]
			bestSplitAttributeGini = currGiniIndex

	#Find the largest of the three information gain variations
	maxInformationGain = max(bestMajorityError, bestGiniIndex, bestEntropy)

	if maxInformationGain == bestEntropy:
		return bestSplitAttributeEntropy
	if maxInformationGain == bestMajorityError:
		return bestSplitAttributeME
	if maxInformationGain == bestSplitAttributeGini:
		return bestSplitAttributeGini



def mostCommonLabel(S):

	labelDictionary = {}
	for example in S:
		if example["label"] in labelDictionary:
			labelDictionary[example["label"]] = labelDictionary[example["label"]] + 1
		else:
			labelDictionary[example["label"]] = 0

	count = 0
	commonLabel = None
	for label in labelDictionary.keys():
		if count == 0:
			commonLabel = label
			count = count + 1
		elif labelDictionary[label] > labelDictionary[commonLabel]:
			commonLabel = label

	return commonLabel



def checkAllLabelsAreEqual(S):
	count = 0
	label = None
	returnVal = {
		"label": label,
		"equal": None
	}

	for example in S:
		if count == 0:
			label = example["label"]
			count = count + 1
			returnVal["label"] = label
		elif label != example["label"]:
			returnVal["equal"] = "false";
			return returnVal

	returnVal["equal"] = "true"		
	return returnVal



openFile("train.csv")
ID3(dataSet, attributes, "unacc" ,6,0)
ID3(dataSet, attributes, "acc", 6, 0)