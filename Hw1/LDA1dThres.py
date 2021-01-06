import sklearn as sk
import numpy as np
import sys
from sklearn import datasets
from numpy import median
from random import seed
from random import randrange
import math

#We use this function to turn our y labels from regression values into a class (0 or 1) based on our given tau value
def classificationify(tau):
	X,t=sk.datasets.load_boston(return_X_y=True)
	divide = np.percentile(t, tau)
	for i in range(len(t)):
		if(t[i]>=divide):
			t[i] = 1
		else:
			t[i] = 0

	return X, t

#This function defines our two dataset
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

#This function splits our dataset into k folds
def kFoldCrossVal(X,t,k):
	sizeOfFold = int(len(X)/k)
	dimX1, dimX2 = np.shape(X)
	#Create copies of our inital datasets 
	copyOfX = np.copy(X)
	copyOfT = np.copy(t)
	#Establish our split datasets
	t_split = np.zeros((k, sizeOfFold))
	x_split = np.zeros((k, sizeOfFold, dimX2))
	for i in range(k):
		#Establish our fold for this iteration and populate it
		xFold = np.zeros((sizeOfFold, dimX2))
		tFold = np.zeros(sizeOfFold)
		for j in range(len(xFold)):
			#Introduce a form of shuffling our data by choosing a random sample in our dataset
			index = randrange(len(copyOfX))
			#Populate our fold
			xFold[j] = copyOfX[index]
			tFold[j] = copyOfT[index]
			#Delete the random sample we just put into our fold to avoid duplicates
			copyOfX=np.delete(copyOfX,index, 0)
			copyOfT=np.delete(copyOfT, index, 0)
		#Add our fold to our dataset
		x_split[i] = xFold
		t_split[i] = tFold
	return x_split, t_split

#This function calculates the means of our two classes.  It's done more gracefully in LDA2dGaussGM.py after I'd shaken the dust off.
def calculateMeans(X, t):
	#Establish our summing variables to calculate our means
	dim1, dim2 = np.shape(X)
	sum1 = np.zeros(dim2)
	sum0 = np.zeros(dim2)
	num1 = 0
	num0 = 0
	for i in range(len(t)):
		#We have class 0, so add that matrix to our sum0 and increase our number of 0s by 1
		if(t[i]==0):
			sum0 = np.add(sum0, X[i])
			num0+=1
		#We have class 1, so add that matrix to our sum1 and increase our number of 1s by 1	
		else:
			sum1 = np.add(sum1, X[i])
			num1+=1

	#If we don't have any of a particular class we have to make sure that we aren't dividing by 0		
	if(num1==0):
		num1=1
	if(num0==0):
		num0=1

	#Reshape to # features x 1
	sum0 = sum0.reshape(np.shape(sum0)[0], 1)
	sum1 = sum1.reshape(np.shape(sum1)[0], 1)

	#Divide by the sum by the number of each class and return our mean matrices
	return sum0*(1/num0), sum1*(1/num1)

#This function calculates our between class covariance
def calculateSB(m1, m2):
	left = np.subtract(m2, m1)
	right = left.transpose()
	return np.matmul(left, right)

#This function calculates our within-class covariance
def calculateSW(X, t, m1, m2):
	xDim1, xDim2 = np.shape(X)
	sumLeft = np.zeros((xDim2, xDim2))
	sumRight = np.zeros((xDim2, xDim2))

	for i in range(len(t)):
		if(t[i]==0):
			temp = np.subtract(X[i].reshape(xDim2, 1), m1)
			sumLeft=np.add(sumLeft, np.matmul(temp, temp.transpose()))
		else:
			temp = np.subtract(X[i].reshape(xDim2, 1), m2)
			sumRight=np.add(sumRight, np.matmul(temp, temp.transpose()))
	sumTotal = np.add(sumLeft, sumRight)
	#multiply sumTotal by the identity matrix times a very small number to avoid singularity issue
	return np.add(sumTotal, np.identity(xDim2)*(1e-15))

#This function calculates our transformational w
def calculateW(X, t):
	#Calculate means
	m1, m2 = calculateMeans(X, t)
	#Calculate between class variance
	SB = calculateSB(m1, m2)
	#Calculate within class variance
	SW = calculateSW(X, t, m1, m2)
	#Calculate eigenvalues and eigenvectors
	eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))
	#In order to reduce to 1 dimension we need to find the largest eigenvalue and its corresponding eigenvector
	eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
	eigenPairs = sorted(eigenPairs, key=lambda x:x[0], reverse=True)
	w = eigenPairs[0][1]
	return w.reshape(np.shape(w)[0], 1)

#This function transforms an x by multiplying it by the transpose of w
def fit(X, w):
	y = []
	for i in range(len(X)):
		y.append(np.dot(w.transpose(), X[i]))
	return np.array(y)

#This function "trains" our model by fitting to training data and determining a threshold
def train(X, t):
	w = calculateW(X, t)
	y = fit(X, w)
	m0, m1 = calculateMeans(y,t)
	#This will be our threshold
	averageMean = np.add(m0, m1)*(1/2)
	#We need to know which class sits above the threshold and which class sits below in order to correctly classify
	if(m0[0][0]>m1[0][0]):
		return averageMean[0][0], 0, 1, w
	else:
		return averageMean[0][0], 1, 0, w

#This function outputs our prediction array
def classify(X, w, threshold, higherMeanClass, lowerMeanClass):
	y = fit(X, w)
	prediction = []
	for i in range(len(y)):
		if(y[i][0]>threshold):
			prediction.append(higherMeanClass)
		else:
			prediction.append(lowerMeanClass)
	return np.array(prediction)

#This function calculates accuracy and was used in building the model as a sanity check
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

#This is the big kahuna - it takes in our number of folds and performs our cross validation, printing our error rates and standard deviation of error
def experiment(num_crossVal):
	X, t = defineDataset('Boston50')
	X_split, t_split = kFoldCrossVal(X, t, num_crossVal)
	overallErrorRate = 0
	avgStdDev = 0

	for i in range(len(X_split)):
		#Test data will be the chosen iteration
		X_test = X_split[i]
		t_test = t_split[i]
		#Training data will be the remaining folds
		X_train_folds = np.delete(np.copy(X_split), i, 0)
		t_train_folds = np.delete(np.copy(t_split), i, 0)
		Xdim1, Xdim2, Xdim3 = np.shape(X_train_folds)
		tdim1, tdim2= np.shape(t_train_folds)
		X_train = np.reshape(X_train_folds, (Xdim1*Xdim2, Xdim3))
		t_train = np.reshape(t_train_folds, tdim1*tdim2)

		#Train and get threshold
		threshold, higherMeanClass, lowerMeanClass, w = train(X_train, t_train)
		#Make predictions
		predictions = classify(X_test, w, threshold, higherMeanClass, lowerMeanClass)
		#Get error rate and standard deviation
		errorRateForFold = errorRate(t_test, predictions)
		stdDevForFold = stdDevOfError(t_test, predictions)
		overallErrorRate+=errorRateForFold
		avgStdDev+=stdDevForFold
		print("The error rate for this fold is: ", errorRateForFold)
		print("The standard deviation for this fold is: ", stdDevForFold)
	print("The overall error rate is: ", overallErrorRate/num_crossVal)
	print("The average standard deviation is: ", avgStdDev/num_crossVal)

#Main, which grabs the input number of cross validation folds from our terminal and performs LDA
if __name__ == '__main__':
	num_crossVal = int(sys.argv[1])
	experiment(num_crossVal)


