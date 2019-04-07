import random
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter
from numpy import linalg as la

# Creates a data set, list of labels, and a dictionary of the each attributes' values based upon the given file name. If unknownType = 1 then we treat any unknown attribute values as missing data otherwise
# they are treated as part of the data set. 
N = 872
x = np.zeros(shape=(N,4))
y = np.zeros(shape=(N,1))
k = 0

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
				# currExample.append(float(1))
				if(int(term) == 0):
					currLabel = float(-1.0)
				else:
					currLabel = float(term)
			count = count + 1
			
		dataSet.append(currExample)
		labels.insert(index, currLabel)
		index = index + 1

	N = len(labels)
	dataSetArray = np.zeros(shape=(len(labels),4))
	labelsArray = np.zeros(shape=(len(labels),1))

	index = 0
	for data in dataSet:
		dataSetArray[index] = [data[0],data[1],data[2],data[3]]
		index += 1

	index = 0
	for label in labels:
		labelsArray[index] = [label]
		index += 1

	return (dataSetArray, labelsArray, dataSet, labels)


def dualSVM(alpha):
	yyT = np.dot(y,np.transpose(y))
	aaT = np.dot(alpha,np.transpose(alpha))
	firstTerm = np.multiply(yyT,aaT)
	firstTerm = np.multiply(firstTerm,k)
	firstTerm = np.sum(firstTerm)
	firstTerm = np.multiply((float(1)/float(2)),firstTerm)
	secondTerm = np.sum(alpha)
	return (firstTerm - secondTerm)

def dual_example(currC):
	eq_cons = {'type': 'eq','fun' : lambda alpha: np.dot(alpha,y)}
	alpha_0 = np.ones(shape=(N,1))
	bds = [ [ 0, currC]] * N
	jac_Val = False
	res = minimize(dualSVM, alpha_0, method='SLSQP',
	                constraints=eq_cons, options={'disp': True},
	                bounds=bds)
	return res

def recover_w(res):
	alpha = res.x
	xiyi = np.multiply(y,x)

	for index in range(N):
		currAlpha = alpha[index]
		xiyi[index] = [currAlpha * element for element in xiyi[index]]

	w = [0,0,0,0]
	for element in xiyi:
		w = [element[index] + w[index] for index in range(4)]

	return w

def recover_b(w):
	count = N
	b = 0

	index = 0
	for example in x:
		yj = y[index]
		wTx = [example[index] * w[index] for index in range(4)]
		wTx = sum(wTx)
		b += yj - wTx
		index += 1

	b = float(b)/float(count)
	return b

def calculateKernel(gamma, x):
	xixj = x
	index = 0
	for example in xixj:
		for j_index in range(4):
			curr_j = example[j_index]
			example = [element - curr_j for element in example]
		xixj[index] = example
		index += 1
	xixj = la.norm(xixj)
	k = np.multiply(-1.0, xixj)
	k = np.divide(k,gamma)
	k = np.exp(k)
	kernel = k
	return kernel

def makePrediction(data, labels, w, b):

	errorCount = 0
	index = 0
	for example in data:
		currLabel = labels[index]
		wTx = np.dot(w,example)
		prediction = wTx + b
		if(prediction >= 0 and currLabel < 0):
			errorCount += 1
		elif(prediction < 0 and currLabel >= 0):
			errorCount += 1
		index += 1

	return float(errorCount)/float(len(labels))




def main():
	(xVal, yVal, trainingData, trainingLabels) = openFile("train.csv")
	(xTestVal, yTestVal,testingData, testingLabels) = openFile("test.csv")
	global x  
	global y 
	x = xVal
	y = yVal

	C = [float(100)/float(873), float(500)/float(873), float(700)/float(873)]
	Cvals = ["100/873","500/873","700/873"]
	gamma = [0.01,0.1,0.5,1,2,5,10,100]
	gammaVals = ["0.01","0.1","0.5","1","2","5","10","100"]
	indexOfC = 0
	for c in C:
		gammaIndex = 0
		for gammaVal in gamma:

			kernel = calculateKernel(gammaVal, x)
			global k
			k = kernel
			res = dual_example(c)
			w = recover_w(res)
			b = recover_b(w)

			print("Current C " + Cvals[indexOfC])
			print("Current Gamma: " + gammaVals[gammaIndex])
			print("The weight vector:")
			print(w)
			print("The bias point:")
			print(b)

			trainingError = makePrediction(trainingData, trainingLabels, w, b)
			print("Training Error: " + str(trainingError))
			testingError = makePrediction(testingData, testingLabels, w, b)
			print("Testing Error: " + str(testingError))
			gammaIndex += 1
		indexOfC += 1


if __name__=="__main__":
	main()