import random
import math

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

def ML_Estimation(data,labels,testData,testLabels):

		w = [0,0,0,0,0]
		gamma_t = 0.0051
		d = 0.05
		epochs = 100
		t = 1
		while t != epochs:
			s_data, s_labels = shuffleData(data,labels)
			
			index = 0
			for example in s_data:
				curr_label = s_labels[index]
				gamma_t = float(gamma_t) / (1 + ((float(gamma_t)/d) * (index+1)))

				wTx = [w[i] * example[i] for i in range(len(example))]
				wTx_val = sum(wTx)

				y_i = curr_label

				yi_wTx = -curr_label * wTx_val

				e_val = math.exp(yi_wTx)

				yixi = [-curr_label * x for x in example]


				numerator_eVal = [val * e_val for val in yixi]
				denomintator_eVal = 1 + e_val 

				ML = [float(val) / denomintator_eVal for val in numerator_eVal]

				gamma_Jw = [gamma_t * val for val in ML] 

				w = [w[i] - gamma_Jw[i] for i in range(len(w))]
				index+= 1
			t += 1

		print("Vector found")
		print(w)
		print("Training Accuracy")
		print(predict(data,labels,w))
		print("Testing Accuracy")
		print(predict(testData,testLabels,w))

def predict(data,labels,w):

	count = 0
	index = 0
	for example in data:
		output = labels[index]
		wTx = sum([w[i] * example[i] for i in range(len(example))])
		if(output < 0 and wTx < 0):
			count += 1
		elif(output > 0 and wTx > 0):
			count += 1

	return float(count)/len(labels)


def main():
	trainingData, trainingLabels = openFile("train.csv")
	testingData, testingLabels = openFile("test.csv")

	ML_Estimation(trainingData,trainingLabels,testingData,testingLabels)

if __name__=="__main__":
	main()