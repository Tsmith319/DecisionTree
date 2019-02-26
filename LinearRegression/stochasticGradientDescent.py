#Taylor Smith
#U0741392

import math
import random


# Creates a data set, list of labels, and a dictionary of the each attributes' values based upon the given file name. If unknownType = 1 then we treat any unknown attribute values as missing data otherwise
# they are treated as part of the data set. 
def openFile(filename):
	f = open(filename, 'r')

	dataSet = list()
	labels = list()

	index = 0
	for line in f:
		terms = line.strip().split(',')
		count = 0;
		currExample = []
		currLabel = None
		for term in terms:
			if count < 7:
				currExample.append(term)
			else:
				currLabel = term
			count = count + 1
			
		dataSet.append(currExample)
		labels.insert(index, currLabel)
		index = index + 1
	return (dataSet, labels)

def stochasticGradientDescent(Data, labels):
	t = 0
	w = [0,0,0,0,0,0,0]
	prevW = w
	threshold = math.exp(-6)
	r = 0.1
	costVals = {}
	count = 0
	Jw_t = {}
	while t >= 0:

		wDiffVector = []
		normW = 0
		index = 0

		for val in w:
			wDiffVector.append(float(val) - float(prevW[index]))
			index += 1
		for val in wDiffVector:
			normW += float((val)**2)


		normW = float(math.sqrt(normW))

		if count != 0 and normW < threshold:
			return (costVals, w) 
		if count == 0:
			count += 1

		Jterms = []
		currCostVal = 0
		index = 0
		randomSampleIndex = random.randint(0,len(Data)-1)
		x = Data[randomSampleIndex]
		y_i = float(labels[randomSampleIndex])
		wTx = 0
		w_index = 0
		for val in w:
			wTx += float(val * float(x[w_index]))
			w_index += 1

		Xij = []
		scalarVal = float(y_i - wTx)

		currCostVal += float((scalarVal)**2)
		currCostVal = (float(1)/2)*currCostVal
		costVals[t] = currCostVal

		scalarVal = float(r)*scalarVal

		for val in x:
			Xij.append(float(float(val) * scalarVal))

		prevW = list(w)	
		index = 0
		newW = [0,0,0,0,0,0,0]
		for val in w:
			newW[index] = val + Xij[index]
			index += 1
		w = newW
		t += 1

		
		

	
def calculateFinalCost(data,labels,w):

	totalCost = 0
	index = 0
	for x in data:
		y_i = float(labels[index])
		wTx = 0
		w_index = 0
		for val in w:
			wTx += float(val * float(x[w_index]))
			w_index += 1
		totalCost += float((y_i - wTx)**2)
		index += 1

	totalCost = (float(1)/2) * totalCost
	return totalCost





def main():

	trainingData, trainingLabels = openFile("train.csv")
	testingData, testingLabels = openFile("test.csv")
	
	costFunctionVals, finalWeightVector = stochasticGradientDescent(trainingData, trainingLabels)
	print("Final Weight Vector: ")
	print(finalWeightVector)
	print("Cost Function Values:")
	print(costFunctionVals.items())

	finalCost = calculateFinalCost(testingData, testingLabels, finalWeightVector)
	print("Final Cost of test Data: " + str(finalCost))


if __name__=="__main__":
	main()