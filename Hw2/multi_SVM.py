import pandas
import numpy as np
import math
from cvxopt import matrix as cvxMatrix
from cvxopt import solvers as cvsSolve
from sklearn import metrics
from sklearn.model_selection import KFold
import itertools
import copy

#Create dataset from six files
def createDataset():
	#47 Zernike moments
	fZer = open("mfeat-zer", "r")
	arrZer =[x.split() for x in fZer.readlines()]
	#240 pixel averages in 2x3 windows
	fPix = open("mfeat-pix", "r")
	arrPix =[x.split() for x in fPix.readlines()]
	#6 morphological features
	fMor = open("mfeat-mor", "r")
	arrMor =[x.split() for x in fMor.readlines()]
	#64 Karhunen-Love coefficients
	fKar = open("mfeat-kar", "r")
	arrKar =[x.split() for x in fKar.readlines()]
	#76 Fourier coefficients of the character shapes
	fFou = open("mfeat-fou", "r")
	arrFou =[x.split() for x in fFou.readlines()]
	#216 profile correlations
	fFac = open("mfeat-fac", "r")
	arrFac =[x.split() for x in fFac.readlines()]

	#Concatenate arrays of features
	features = arrZer.copy()
	arrRemainingReaders=[arrPix, arrMor, arrKar, arrFou, arrFac]
	for reader in arrRemainingReaders:
		features = np.concatenate((features, reader), axis=1)

	#Create y value array
	yRange=[y for y in range(10)]
	yValues=[[v] for v in list(itertools.chain.from_iterable(itertools.repeat(y, 200) for y in yRange))]

	#Concatenate into one dataset
	dataset = np.concatenate((features, yValues), axis=1)
	#Write csv file
	pandas.DataFrame(dataset).to_csv("multi_SVM_dataset.csv", index=False)

	#Close the files
	fZer.close()
	fPix.close()
	fMor.close()
	fKar.close()
	fFou.close()
	fFac.close()




#Load file from CSV file and split into 80% training and 20% testing
def loadData(dataset):
	df = pandas.read_csv(dataset)
	numpyData = df.to_numpy()
	np.random.shuffle(numpyData)

	partition = math.floor(0.8*(len(numpyData)))
	train, test = numpyData[0:partition], numpyData[partition:]

	X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]

	return X_train, y_train.reshape(-1,1), X_test, y_test.reshape(-1,1)

#Kernel functions
#Linear kernel is simply the dot product
def linear_kernel(x, x_prime):
	return np.dot(x, x_prime)

#RBF is exp(-||x-x'||^2_2 / (2 sigma^2)) where sigma is a hyperparameter
def rbf_kernel(x, x_prime, sigma):
	return np.exp(-np.linalg.norm(x-x_prime)**2 / (2*(sigma**2)))

#Calculate our P matrix, which is yiyjk(x_i, x_j)
def calculateGramMatrix(X, y, m, kernel, sigma):
	#Create gram matrix
	G = np.zeros((m,m))
	for i in range(m):
		for j in range(m):
			if kernel=="linear":
				G[i,j] = linear_kernel(X[i], X[j])
			else:
				G[i,j] = rbf_kernel(X[i], X[j], sigma)

	return np.outer(y, y)*G

#Convert to cvxopt form and solve using the solver
def solve(C, X, y, K):

	m, n = np.shape(X)
	#Convert to optimizer matrices
	P=cvxMatrix(K)
	q=cvxMatrix(-1*np.ones((m,1)))
	A=cvxMatrix(y.T)
	b=cvxMatrix(np.zeros(1)) 
	G=cvxMatrix(np.vstack((np.identity(m)*-1, np.identity(m))))
	h=cvxMatrix(np.hstack((np.zeros(m), np.ones(m)*C)))

	#SOLVE!
	solver=cvsSolve.qp(P,q,G,h,A,b)
	lambdas = np.array(solver['x'])

	return lambdas

#Calculate the decision boundary for our SVM, which will be used to make predictions
def calculateDecisionBoundary(C, sigma, X, y, kernel):
	#Calculate our gram matrix
	m, n = np.shape(X)
	K = calculateGramMatrix(X, y, m, kernel, sigma)

	#Calculate our lambdas
	lambdas = solve(C, X, y, K)

	#Calculate our support vector set
	vectors = (lambdas>1e-5).flatten()
	sv = lambdas[vectors]
	sv_y = y[vectors]
	sv_x = X[vectors]


	#Trying this a new way according to what is in the book
	b = 0 
	for n in range(len(sv)):
		b+=sv_y[n]
		tempSum=0
		for m in range(len(sv)):
			if(kernel=="linear"):
				tempSum+=sv[m]*sv_y[m]*linear_kernel(sv_x[n],sv_x[m])
			else:
				tempSum+=sv[m]*sv_y[m]*rbf_kernel(sv_x[n],sv_x[m], sigma)
		b-=tempSum


	return sv, b, sv_y, sv_x


#the final prediction function becomes:
#Sum y_i lambda_i k(x_i, x) where k(x_i, x_j) = phi(x_i)^Tphi(x_j)
def predict(X, b, lambdas, sv_y, sv_x, kernel, sigma):
	predictions=np.zeros(len(X))
	for i in range(len(X)):
		totalSum=0
		for lambda_i, sv_y_i, sv_x_i in zip(lambdas, sv_y, sv_x):
			if kernel=="linear":
				totalSum+=lambda_i*sv_y_i*linear_kernel(X[i], sv_x_i)
			else:
				totalSum+=lambda_i*sv_y_i*rbf_kernel(X[i], sv_x_i, sigma)
		predictions[i] = totalSum
	predictions = predictions + b
	return np.sign(predictions)

#For each class we perform a binary classification of not this class vs. this class
def binaryClassification(X_train, X_test, y_train, sigma, c, kernel):
	#Calculate our decision boundary based on our fold training data
	sv, b, sv_y, sv_x = calculateDecisionBoundary(c, sigma, X_train, y_train, kernel)
	#Make predictions on our data
	trainPredictions=predict(X_train, b, sv, sv_y, sv_x, kernel, sigma)
	testPredictions=predict(X_test, b, sv, sv_y, sv_x, kernel, sigma)
	return trainPredictions, testPredictions

#Run through the k classes doing a binary classification problem for each of them
def multiClassClassification(X_train, X_test, y, sigma, c, kernel):
	#get number of classes
	k = len(np.unique(y))

	#If a class is never positively classified it will have a default class of 0
	#This could lead to an overclassification of the 0 class
	finalTrainPredictions=np.zeros(len(X_train))
	finalTestPredictions=np.zeros(len(X_test))

	for i in range(k):
		yCopy = copy.deepcopy(y)
		#Set anything that isn't the class we're looking at to -1 (one vs. all)
		yCopy[yCopy!=i]=-1
		#Set anything that is the class we're looking at to a +1 (one vs. all)
		yCopy[yCopy==i]=1
		trainPredictions, testPredictions = binaryClassification(X_train, X_test, yCopy, sigma, c, kernel)

		#The one vs. all approach leads to if a point is classified into multiple different classes
		#I make the choice to classify a point by the largest class it is classified into
		#For example, if a point is classified as 2 and 8 it will have a final classification of 8
		#This could lead to an overclassification of larger digits and underclassification of small digits
		for j in range(len(finalTrainPredictions)):
			if(trainPredictions[j]==1):
				finalTrainPredictions[j]=i
		for p in range(len(finalTestPredictions)):
			if(testPredictions[p]==1):
				finalTestPredictions[p]=i
	return finalTrainPredictions, finalTestPredictions


#Take in the name of a dataset
#Print out the average train and validation error rates and standard deviations
#Print out the optimal hyperparameters and the test set error for the best model
def multi_SVM(dataset:str):
	#Set up data
	X_train, y_train, X_test, y_test = loadData(dataset)
	cArr = [0.01,0.1,1]
	sigmaArr=[np.var(X_train)/10, np.var(X_train), np.var(X_train)*10]
	kernelArr = ['linear', 'rbf']
	#Get 10 folds using k-folds
	kf = KFold(n_splits=10)

	for kernel in kernelArr:
		#Get optimal C through cross validation on training data
		optimalC=0
		optimalSigma=0
		lowestErrorRate=1
		hyperparameters=[]
		if kernel=="linear":
			hyperparameters=[(0, 0.01), (0, 0.1), (0, 1)]
		else:
			for s in sigmaArr:
				for c in cArr:
					hyperparameters.append((s, c))

		for sigma, c in hyperparameters:
			validErrorRates=[]
			trainErrorRates=[]
			validErrorRate=0
			trainErrorRate=0

			for train_index, test_index in kf.split(X_train):
				#Get our fold train and test data
				X_k_train, X_k_test = X_train[train_index], X_train[test_index]
				y_k_train, y_k_test = y_train[train_index], y_train[test_index]

				#Make predictions on our fold train and validation data using our learned w and b
				trainPredictions, testPredictions=multiClassClassification(X_k_train, X_k_test, y_k_train, sigma, c, kernel)

				#Calculate our error rates and record them
				testErrorRate = 1-metrics.accuracy_score(y_k_test, testPredictions)
				trainErrorRate=1-metrics.accuracy_score(y_k_train, trainPredictions)
				validErrorRates = np.append(validErrorRates, testErrorRate)
				trainErrorRates = np.append(trainErrorRates, trainErrorRate)
				validErrorRate+=testErrorRate
				trainErrorRate+=trainErrorRate

			#Average our error rates	
			validErrorRate = validErrorRate/10
			trainErrorRate = trainErrorRate/10

			#Choose our optimal C and sigma with the lowest validation error rate
			if(validErrorRate<lowestErrorRate):
				lowestErrorRate = validErrorRate
				optimalC = c
				optimalSigma = sigma

			print("The C value is: ", c)
			print("The sigma value is: ", sigma)
			print("The average training error rate is: ", trainErrorRate)
			print("The standard deviation of training error rates is: ", np.std(trainErrorRates))
			print("The average validation error rate is: ", validErrorRate)
			print("The standard deviation of testing error rates is: ", np.std(validErrorRates))
			print("\n")

		print("The kernel is: ", kernel)
		print("The optimal C is: ", optimalC)
		print("The optimal sigma is: ", optimalSigma)
		print("The error rate for the optimal C is: ", lowestErrorRate)
		print("\n")

		#Train SVM with all train data and apply to test data with optimal C and Sigma
		finalTrainPredictions, finalTestPredictions=multiClassClassification(X_train, X_test, y_train, optimalSigma, optimalC, kernel)
		errorRate = 1-metrics.accuracy_score(y_test, finalTestPredictions)
		print("The final error rate for the test data with the optimal values of C and sigma for kernel ", kernel, "IS...",errorRate,"!!!")






if __name__ == '__main__':
	#Create the csv file by uncommenting out the below command
	#createDataset()
	multi_SVM('multi_SVM_dataset.csv')
