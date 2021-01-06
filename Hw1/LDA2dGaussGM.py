import sklearn as sk
import numpy as np
import sys
import math
from sklearn import datasets
from numpy import median
from random import seed
from random import randrange
from collections import defaultdict
#We use this function to turn our y labels from regression values into a class (0 or 1) based on our given tau value
#It isn't used in this problem because we're only dealing with the Digits dataset but was left in for generality
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

#This function divides our data into a dictionary containing an array of X data per class
#This makes calculating means and such per class far simpler
def divideByClass(X, t):
	data=[(t[i], X[i]) for i in range(len(X))]
	dataDict=defaultdict(list)
	for k, v in data:
		dataDict[k].append(v)

	return dataDict

#This function takes in our class organized dictionary and returns a dictionary containing means per class
#Because it was simple to also calculate our overall mean and count of samples per class that was also done here for use in future calculations
def calculateMeans(dividedByClass, X):
	#Establish our summing variables to calculate our means
	means={}
	dim1, dim2 = np.shape(X)
	overallMean = np.zeros(dim2)
	niPerClass={}
	#For each class calculate the mean and count of samples
	for c in dividedByClass:
		sumC = np.zeros(dim2)
		numC = 0
		for i in range(len(dividedByClass[c])):
			sumC=np.add(sumC, dividedByClass[c][i])
			numC+=1
			overallMean = np.add(overallMean, dividedByClass[c][i])
		if numC==0:
			numC=1
		sumC = np.reshape(np.shape(sumC)[0], 1)
		means[c]=sumC*(1/numC)
		niPerClass[c]=numC

	overallMean = overallMean.reshape(np.shape(overallMean)[0], 1)
	#Return our means dictionary, overall mean, and counts dictionary 
	return means, overallMean*(1/len(X)), niPerClass

#Calculate our between-class covariance matrix
def calculateSB(X, means, overallMean, niPerClass):
	xDim1, xDim2 = np.shape(X)
	sumTotal = np.zeros((xDim2, xDim2))
	for c in means:
		temp = np.subtract(means[c], overallMean)
		multTemp = np.dot(temp, temp.transpose())
		sumTotal = np.add(sumTotal, multTemp*niPerClass[c])

	return sumTotal

#Calculate our within-class covariance matrix
def calculateSW(X, dividedByClass, means):
	xDim1, xDim2 = np.shape(X)
	sums={}
	for c in dividedByClass:
		sums[c] = np.zeros((xDim2, xDim2))
		for i in range(len(dividedByClass[c])):
			temp = np.subtract(dividedByClass[c][i], means[c])
			sums[c] = np.add(sums[c], np.dot(temp, temp.transpose()))
	sumTotal = np.zeros((xDim2, xDim2))
	for c in sums:
		sumTotal = np.add(sumTotal, sums[c])
	return sumTotal

#Using Sw and Sb calculate our w
def calculateW(k, X, t):
	dividedByClass = divideByClass(X, t)
	#Calculate means
	means, overallMean, niPerClass = calculateMeans(dividedByClass, X)
	#Calculate between class variance
	SB = calculateSB(X, means, overallMean, niPerClass)
	#Calculate within class variance
	SW = calculateSW(X, dividedByClass, means)
	#Add some noise to SW to prevent singular matrix issues when taking its inverse
	SW = np.add(SW, np.identity(np.shape(SW)[0])*(1e-9))
	#Calculate eigenvalues and eigenvectors
	eigenValues, eigenVectors = np.linalg.eig(np.linalg.inv(SW).dot(SB))
	#In order to reduce to 2 dimensions we need to find the largest eigenvalues and its corresponding eigenvectors
	eigenPairs = [(np.abs(eigenValues[i]), eigenVectors[:,i]) for i in range(len(eigenValues))]
	eigenPairs = sorted(eigenPairs, key=lambda x:x[0], reverse=True)
	w = np.hstack((eigenPairs[0][1].reshape(np.shape(X)[1], 1), eigenPairs[1][1].reshape(np.shape(X)[1], 1))).real
	return w

#Transform our input x by multiplying it by the transpose of w
def fit(k, X, w):
	y = []
	for i in range(len(X)):
		y.append(np.dot(w.transpose(), X[i]))
	return np.array(y)

#Calculate our gaussian parameters: means per class, covariances per class, prior probabilities per class
def train(X, t, w):
	dividedByClass = divideByClass(X, t)
	means = {}
	covariance={}
	priors={}
	for c, v in dividedByClass.items():
		#project our values down to 2 dimensions
		y = np.dot(v, w)
		#Calculate the means for our projection
		means[c] = np.mean(y, axis=0)
		#Calculate the covariances for our projections
		covariance[c] = np.cov(y, rowvar=False)
		#Estimate our priors by grabbing the number of samples in this class and dividing it by our total number of samples
		priors[c] = np.shape(v)[0]/len(X)
	return means, covariance, priors

#Create our Gaussian distribution from our mean and covariance
def gaussian_dist(x, u, cov):
	#Add some noise to prevent singular matrix issues
	cov = np.add(cov, np.identity(np.shape(cov)[0])*(1e-9))
	temp = (1.0/((2*np.pi)**(x.shape[0]/2.0)))
	temp = temp * (1/np.sqrt(np.linalg.det(cov)))
	xMinusMean = np.subtract(x, u)
	return temp*np.exp((-1/2)*np.dot(np.dot(xMinusMean.transpose(), np.linalg.inv(cov)), xMinusMean))

#Given our number of classes, training data, and testing input output our predictions
def predict(k, XTrain, XTest, tTrain):
	#Calculate our w vector
	w = calculateW(k, XTrain, tTrain)
	#Transform our testing input down to 2 dimensions 
	y = fit(k, XTest, w)
	gaussianLikelihoods = []
	#Calculate our gaussian parameters per class from our training data
	means, covariances, priors = train(XTrain, tTrain, w)
	#Calculate our likelihoods P(C_k|x)
	for x in y:
		row = []
		for i in range(k):
			res = priors[i] * gaussian_dist(x, means[i], covariances[i])
			row.append(res)
		gaussianLikelihoods.append(row)

	gaussianLikelihoods = np.array(gaussianLikelihoods)
	#Our prediction per input becomes the class with the highest likelihood
	predictions = np.argmax(gaussianLikelihoods, axis=1)
	return predictions

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


#Where the magic happens - puts everything together and performs cross-validation for an input number of cross validation folds
#It prints out our error rate and standard deviation per fold as well as our average error rate and average standard deviation
def experiment(num_crossVal):
	X, t = defineDataset('Digits')
	X_split, t_split = kFoldCrossVal(X, t, num_crossVal)
	overallErrorRate=0
	avgStdDev=0

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

		#Make predictions
		predictions = predict(10, X_train, X_test, t_train)
		#Calculate and output error rates and standard deviations
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
