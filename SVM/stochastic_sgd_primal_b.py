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

def shuffleData(data,labels):
	numOfItems = len(labels)
	shuffled_Data = list()
	shuffled_Labels = list()
	indecesAlreadyUsed = set()

	while ((len(shuffled_Data) != numOfItems) and (len(shuffled_Labels) != numOfItems)):
		currIndex = random.randint(0,numOfItems-1)
		if currIndex not in indecesAlreadyUsed:
			shuffled_Data.append(data[currIndex])
			shuffled_Labels.append(labels[currIndex])
			indecesAlreadyUsed.add(currIndex)

	return (shuffled_Data, shuffled_Labels)

def stochastic_sgd(data,labels,testData,testLabels):
	w = [0,0,0,0]
	epoch = 1
	T = 100
	C_vals = [float(1)/float(873),float(10)/float(873),float(50)/float(873),float(100)/float(873),float(300)/float(873),float(500)/float(873),float(700)/float(873)]
	C_strings = ["1/873","10/873","50/873","100/873","300/873","500/873","700/873"]
	c_count = 0;
	N = 872
	gamma_o = 0.0005

	for C in C_vals:
		w = [0,0,0,0,0]
		w_0 = [0,0,0,0]
		trainingPredictionError = 0
		testingPredictionError = 0
		epoch = 1
		while epoch != T:
			shuffData, shuffLabels = shuffleData(data,labels)
			index = 0
			for example in shuffData:
				y_i = shuffLabels[index]
				
				wTx = [(float(example[i]) * float(w[i])) for i in range(0,5)]
				wTx = sum(wTx)
					
				w_0 = w[0:4]
				gamma_t = float(gamma_o)/(float(1 + epoch))
				if (float(y_i) * float(wTx)) <= 1.0:
					
					ytCNyi = float(y_i * C * N * gamma_t)
					loss = [(float(x) * float(ytCNyi)) for x in example]

					w_0_copy = w_0[:]
					w_0_copy.append(0)
					w_0_copy = [(float(weight) * float(1-gamma_t)) for weight in w_0_copy]

					vector_Index = 0
					for x in loss:
						w[vector_Index] = float(w_0_copy[vector_Index]) + float(x)
						vector_Index += 1

				else: 
					w_0 = [(float(1 - gamma_t) * float(weight)) for weight in w_0]
				index += 1


			epoch += 1

			trainingPredictionError += predictData(data,labels,w)
			testingPredictionError += predictData(testData,testLabels,w)
		
		print("C value: "+C_strings[c_count])
		print("Training Errors")
		print(float(trainingPredictionError)/float(T))
		print("Testing Errors")
		print(float(testingPredictionError)/float(T))
		print("Vector")
		print(w)
		c_count += 1
	return w

def predictData(data,labels,w):
	predictedError = 0.0
	totalData = len(labels)
	index = 0
	for example in data:
		y_i = labels[index]
		vector_Index = 0
		wTx = 0.0
		for x in example:
			wTx += (float(x) * float(w[vector_Index]))
			vector_Index += 1
		if(float(y_i) * float(wTx)) < 0:
			predictedError += 1
		index += 1

	return (float(predictedError)/float(totalData))



def main():
	trainingData, trainingLabels = openFile("train.csv")
	testingData, testingLabels = openFile("test.csv")

	w_Vector = stochastic_sgd(trainingData, trainingLabels, testingData, testingLabels)



if __name__=="__main__":
	main()