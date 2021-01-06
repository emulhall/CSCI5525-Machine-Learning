import pandas
import numpy as np
import math
from cvxopt import matrix as cvxMatrix
from cvxopt import solvers as cvsSolve
from sklearn import metrics
from sklearn.model_selection import KFold
import itertools

#Load file from CSV file and split into 80% training and 20% testing
def loadData(dataset):
	df = pandas.read_csv(dataset)
	numpyData = df.to_numpy()
	np.random.shuffle(numpyData)
	partition = math.floor(0.8*(len(numpyData)))
	train, test = numpyData[0:partition], numpyData[partition:]
	X_train, y_train, X_test, y_test = train[:,:-1], train[:,-1], test[:,:-1], test[:,-1]
	y_train[y_train==0] = -1
	y_test[y_test==0] = -1

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

#Convert to cvxopt matrix form and use solver to get lagrange multipliers
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


	#Calculating our intercept
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


#Takes in a name of a dataset
#Prints out the average train and validation error rates and standard deviations
#Prints out optimal hyperparameter values
#Prints out the test set error rate for the best model
def kernel_SVM(dataset:str):

	#Set up data
	X_train, y_train, X_test, y_test = loadData(dataset)
	cArr = [0.01,0.1,1]
	sigmaArr=[np.var(X_train)/100, np.var(X_train)/10, np.var(X_train), np.var(X_train)*10, np.var(X_train)*100]
	kernelArr = ['linear', 'rbf']
	#Get 10 folds using k-folds
	kf = KFold(n_splits=10)

	for kernel in kernelArr:
		#Get optimal C and sigma through cross validation on training data
		optimalC=0
		optimalSigma=0
		lowestErrorRate=1
		hyperparameters=[]
		#Don't need to run extra sigma values for linear, since its only hyperparameter is C
		if kernel=="linear":
			hyperparameters=[(0, 10e-4),(0, 0.01), (0, 0.1), (0, 1), (0, 10), (0, 100)]
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

				#Calculate our decision boundary based on our fold training data
				sv, b, sv_y, sv_x = calculateDecisionBoundary(c, sigma, X_k_train, y_k_train, kernel)

				#Make predictions on our fold train and validation data using our learned w and b
				testPredictions=predict(X_k_test, b, sv, sv_y, sv_x, kernel, sigma)
				trainPredictions=predict(X_k_train,b, sv, sv_y, sv_x, kernel, sigma)

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

			#Choose our optimal C, with the lowest validation error rate
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

		#Train SVM with all train data and apply to test data with optimal C
		sv_star, b_star, sv_y_star, sv_x_star= calculateDecisionBoundary(optimalC, optimalSigma, X_train, y_train, kernel)
		finalTestPredictions = predict(X_test, b_star, sv_star, sv_y_star, sv_x_star, kernel, optimalSigma)
		errorRate = 1-metrics.accuracy_score(y_test, finalTestPredictions)
		print("The final error rate for the test data with the optimal values of C and sigma for kernel ", kernel, "IS...",errorRate,"!!!")


if __name__ == '__main__':
	kernel_SVM('hw2_data_2020.csv')