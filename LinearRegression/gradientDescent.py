#Taylor Smith
#U0741392

import math


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

def gradientDescent(Data, labels):
	t = 0
	w = [0,0,0,0,0,0,0]
	prevW = w
	threshold = math.exp(-6)
	r = 0.0125
	costVals = {}
	count = 0
	Jw_t = {}
	while t >= 0:

		wDiffVector = []
		normW = 0
		index = 0
		for val in w:
			wDiffVector.append(float(val - prevW[index]))
			index += 1
		for val in wDiffVector:
			normW += (val)**2

		normW = math.sqrt(normW)

		print("Norm: " + str(normW))
		if count != 0 and normW < threshold:
			return (costVals, w) 
		if count == 0:
			count += 1

		Jterms = []
		currCostVal = 0
		index = 0
		for x in Data:
			y_i = float(labels[index])
			wTx = 0
			w_index = 0
			for val in w:
				wTx += float(val * float(x[w_index]))
				w_index += 1

			Xij = []
			scalarVal = float(y_i - wTx)
			currCostVal += float((scalarVal)**2)

			for val in x:
				Xij.append(float(float(val) * scalarVal))
			Jterms.append(Xij)

			index += 1

		currCostVal = (float(1)/2)*currCostVal
		currJterm = [0,0,0,0,0,0,0]
		for term in Jterms:
			index = 0
			for val in term:
				currJterm[index] += float(val)
				index += 1

		index = 0
		for term in currJterm:
			currJterm[index] = float(currJterm[index] * -1)
			index += 1

		Jw_t[t] = currJterm
		costVals[t] = currCostVal
		prevW = list(w)

		index = 0
		newW = [0,0,0,0,0,0,0]
		for val in w:
			temp = [0,0,0,0,0,0,0]
			tempIndex = 0
			for term in currJterm:
				temp[tempIndex] = float(term * r)
				tempIndex += 1

			newW[index] = val - temp[index]
			index += 1
		w = newW
		t += 1

	





def main():

	trainingData, trainingLabels = openFile("train.csv")
	testingData, testingLabels = openFile("test.csv")
	
	costFunctionVals, finalWeightVector = gradientDescent(trainingData, trainingLabels)
	print("Final Weight Vector: ")
	print(finalWeightVector)
	print("Cost Function Values:")
	print(costFunctionVals.items())



if __name__=="__main__":
	main()