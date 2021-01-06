import sklearn as sk
import numpy as np
import sys
from sklearn import datasets
from numpy import median
from random import seed
from random import randrange
from collections import defaultdict
import math
import copy
import matplotlib
import matplotlib.pyplot as plt

#We use this function to turn our y labels from regression values into a class (0 or 1) based on our given tau value
def classificationify(tau):
	X,t=sk.datasets.load_boston(return_X_y=True)
	divide = np.percentile(t, tau)
	#boston 
	for i in range(len(t)):
		if(t[i]>=divide):
			t[i] = 1
		else:
			t[i] = 0

	return X, t

#This function defines our two datasets, Boston50 and Boston75
def defineDataset(dataset):
	#we load our boston datasets using our classificationify method
	if(dataset=='Boston50'):
		X, t = classificationify(50)
		return X, t
	elif(dataset=='Boston75'):
		X, t = classificationify(75)
		return X, t
	else:
		X, t = sk.datasets.load_digits(return_X_y=True)
		return X, t

#This function divides our data into a dictionary containing an array of X data per class
#This makes calculating means and such per class far simpler
def divideByClass(X, t):
	data=[(t[i], X[i]) for i in range(len(X))]
	dataDict=defaultdict(list)
	for k, v in data:
		dataDict[k].append(v)

	return dataDict

#Split our data into a number of test and training splits, specified by our input
def trainTestSplit(numSplits, X, t):
	classDict = divideByClass(X, t)
	splits = []
	for j in range(numSplits):
		X_train_list = []
		t_train_list = []
		X_test_list = []
		t_test_list = []
		copyOfDict = copy.deepcopy(classDict)
		for c in copyOfDict:
			#Split into 80% train and 20% test
			len_train_split = math.ceil(0.8*(len(copyOfDict[c])))
			#Assign a random 80% to the training set
			for i in range(len_train_split):
				index = randrange(len(copyOfDict[c]))
				randX = copyOfDict[c].pop(index)
				X_train_list.append(randX)
				t_train_list.append(c)
			#Assign the rest to our test set
			for i in range(len(copyOfDict[c])):
				testX= copyOfDict[c].pop()
				X_test_list.append(testX)
				t_test_list.append(c)
		splits.append([np.array(X_train_list), np.array(t_train_list), np.array(X_test_list), np.array(t_test_list)])
	return splits

#This function was used while building the model for a sanity check
def accuracy(t, predictions):
	sum=0
	for i in range(len(t)):
		if(t[i]==predictions[i]):
			sum+=1
	return sum/len(t)

#This function calculates our error rate
def errorRate(t, predictions):
	sum=0
	for i in range(len(t)):
		if(t[i]!=predictions[i]):
			sum+=1
	return sum/len(t)

#This function calculates the standard deviation of our error
def stdDevOfError(t, predictions):
	mean=np.mean(t)
	sumTotal=0
	for i in range(len(predictions)):
		sumTotal+=(predictions[i]-mean)**2
	return math.sqrt(sumTotal/len(predictions))/math.sqrt(len(t))

#The all-great logistic function
def sigmoid(z):
	return 1/ (1+np.exp(-z))

#Calculate cross entropy loss and the gradient of our loss
def cost(theta, x, t):
	h = sigmoid(np.matmul(x,theta))
	avoidDivideByZero = np.zeros(np.shape(h))+1e-15
	h=h+avoidDivideByZero
	m = len(t)
	temp = (-1*t)*np.log(h)
	costSum = np.sum(temp-(1-t)*np.log(1-h))
	cost = 1/m*costSum
	grad = 1/m*(np.matmul((t-h),x))
	return cost, grad

#Fit our function by iterating through and approaching our optimal solution using gradient descent
def fit(x, t, iterations, alpha):
	#initialize
	x = np.insert(x, 0, 1, axis=1)
	thetas=[]
	classes=np.unique(t)
	costs = np.zeros(iterations)

	#Because we have more than 2 classes with our digits dataset we implement a one vs rest binary classification
	for c in classes:
		binarizedt = np.where(t==c, 1, 0)
		theta = np.zeros(np.shape(x)[1])
		#iterate over our determined number of iterations
		for i in range(iterations):
			costs[i], grad = cost(theta, x, binarizedt)
			theta+=alpha*grad
		thetas.append(theta)

	return thetas, costs

#Given our thetas and our input make a prediction for the class
def predict(thetas, x):
	x = np.insert(x, 0, 1, axis=1)
	return [np.argmax([sigmoid(np.matmul(xi, theta)) for theta in thetas])for xi in x]

#Split our training data by some given percentage
def splitTrainByPercent(percentage, trainX, trainT):
	#We need to make sure that we still keep our classes fairly balanced within our dataset
	classDict = divideByClass(trainX, trainT)
	copyOfDict = classDict.copy()
	keepXList = []
	keepTList = []
	for c in copyOfDict:
		len_split = math.floor((percentage/100)*(len(copyOfDict[c])))
		for i in range(len_split):
			index = randrange(len(copyOfDict[c]))
			randX = copyOfDict[c].pop(index)
			keepXList.append(randX)
			keepTList.append(c)

	return np.array(keepXList), np.array(keepTList)

#Split our training data into a set of training data subsets based on a given vector of percentages
def splitTrainByVector(percentage_vector, trainX, trainT):
	trainingSplits = []
	for i in range(len(percentage_vector)):
		trainingSplits.append(splitTrainByPercent(int(percentage_vector[i]), trainX, trainT))
	return trainingSplits

def experiment(num_splits, percentage_vector):
	#Define datasets
	datasets={}
	datasets['Boston50']=(defineDataset('Boston50'))
	datasets['Boston75']=(defineDataset('Boston75'))
	datasets['Digits']=(defineDataset('Digits'))

	#We want to run our experiment for each dataset
	for d in datasets:
		totalErrorRate = 0
		count=0
		errorRatesSum={}
		stdDevsSum={}
		#initialize errorRates
		for element in percentage_vector:
			errorRatesSum[element]=0
			stdDevsSum[element]=0
		print("I am in dataset ", d)
		#Split our data into testing and training data and do this num_splits times
		testTrainSplits = trainTestSplit(num_splits, datasets[d][0], datasets[d][1])
		#Now that we have our training and testing data we want to split our training data and perform experiments
		for i in range(len(testTrainSplits)):
			print("I am in split ", i+1)
			#For clarity
			X_test = testTrainSplits[i][2]
			t_test = testTrainSplits[i][3]
			trainingSplits = splitTrainByVector(percentage_vector, testTrainSplits[i][0], testTrainSplits[i][1])
			for j in range(len(trainingSplits)):
				print("I have the following training data percentage: ", percentage_vector[j])
				X_train = trainingSplits[j][0]
				t_train = trainingSplits[j][1]
				thetas, costs = fit(X_train, t_train, 1000, 0.1)
				predictions=predict(thetas, X_test)
				error=errorRate(predictions,t_test)
				print("My error rate is ", error)
				errorRatesSum[percentage_vector[j]]+=error
				stdDevsSum[percentage_vector[j]]+=stdDevOfError(t_test, predictions)
				totalErrorRate+=error
				count+=1
				print("...........................")
			print("-----------------------------")
			print("The overall error rate is", totalErrorRate/count)
		print("\n")
		print("\n")

		#convert error rates into an array to be plotted
		errorRates=[]
		for percent in errorRatesSum:
			errorRates.append(errorRatesSum[percent]/10)
		stdDevs=[]
		for percent in stdDevsSum:
			stdDevs.append(stdDevsSum[percent]/10)

		#Uncomment if you'd like to make your own plots
		#fig = plt.figure()
		#axes = fig.add_axes([0.1,0.1,0.8,0.8])
		#axes.plot(percentage_vector, errorRates)
		#axes.set_xlabel('Percentage of Training Data')
		#axes.set_ylabel('Error Rate')
		#axes.set_title(d)
		#axes.errorbar(percentage_vector, errorRates, yerr=stdDevs)
		#fig.savefig('logisticRegression'+d+'.png')





if __name__ == '__main__':
	num_splits = int(sys.argv[1])
	percentage_vector = sys.argv[2:]
	percentage_vector[0] = percentage_vector[0].replace("[","")
	percentage_vector[len(percentage_vector)-1]=percentage_vector[len(percentage_vector)-1].replace("]","")
	experiment(num_splits, percentage_vector)