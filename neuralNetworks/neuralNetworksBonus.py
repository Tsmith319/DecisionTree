from __future__ import print_function
import tensorflow as tf 

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
			if count < 4:
				currExample.append(float(term))
			else:
				currExample.append(float(1))
				if(int(term) == 0):
					currLabel = float(-1.0)
				else:
					currLabel = float(term)
			count = count + 1
			
		dataSet.append(currExample)
		labels.insert(index, currLabel)
		index = index + 1
	return (dataSet, labels)

def MAP(dataSet, labels)

def main():
	trainingData, trainingLabels = openFile("train.csv")
	testingData, testingLabels = openFile("test.csv")

if __name__=="__main__":
	main()