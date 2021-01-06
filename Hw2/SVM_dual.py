import pandas
import numpy as np
import math
from cvxopt import matrix as cvxMatrix
from cvxopt import solvers as cvsSolve
from sklearn import metrics
from sklearn.model_selection import KFold
#According to the cvxopt API:
#min1/2(x^T)(Px) + (q^x)
#s.t. Gx <= h
#Ax = b

#Our original problem is:
#(max alpha, lambda over sum lambda_i) - 1/2 (sum (lambda_i)(lambda_j)(y_i) (y_j)(x_i^T)(x_j))
#s.t. C = lambda_i+alpha_i
#lambda_i, alpha_i >=0

#Translating our original problem to cvxopt API gives us:
# min lambda 1/2 (lambda^T)(H)(lambda)-1^T(lambda)
# s.t. -lambda_i, alpha_i <= 0
#lambda_i <= C
# y^T(lambda)=0
#where H = (y_i) (y_j)(x_i^T)(x_j)

#Use above as a guide to the below code

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

#P = (y_i) (y_j)(x_i^T)(x_j)
def calculateP(X, y):
	temp = y*X
	P = np.dot(temp, temp.T)
	return P

#P = P
#q = -1
#G is a diagonal matrix of -1s stacked on top of a matrix of ones (because -lambda_i>=0 AND lambda_i<=C)
#h is a vector of zeros of size mx1 stacked on top of a matrix of 1*C (again, because -lambda_i>=0 AND lambda_i<=C)
#We use G and h to capture both of these inequalities
#A is y^T 
#b is 0
def solve(C, X, y):
	#Calculate H
	m, n = np.shape(X)

	#Convert to optimizer matrices
	P=cvxMatrix(calculateP(X, y))
	q=cvxMatrix(-1*np.ones((m,1)))
	G=cvxMatrix(np.vstack((np.identity(m)*-1, np.identity(m))))
	h=cvxMatrix(np.hstack((np.zeros(m), np.ones(m)*C)))
	A=cvxMatrix(y.T)
	b=cvxMatrix(np.zeros(1)) 

	#SOLVE!
	solver=cvsSolve.qp(P,q,G,h,A,b)
	lambdas = np.array(solver['x'])

	return lambdas

#w=sum(y_i)(lambda_i)(x_i)
def calculateDecisionBoundary(C, X, y):
	#Calculate our support vectors
	lambdas = solve(C, X, y)

	#Calculate our support vector set
	supportVectors = (lambdas>0).flatten()

	#Calculate our w and b to form our predictor
	w = np.sum(((y*lambdas)*X), axis=0)
	w = w.reshape(-1, 1)
	b=y[supportVectors]-np.dot(X[supportVectors], w)
	return supportVectors, w, b

#Use our learned decision boundary to make predictions
def predict(w, X, b):
	predictions=[]
	for i in range(len(X)):
		predictions.append(w.T@X[i]+b[i])
	return np.sign(predictions)


#Takes in our dataset and prints out the average train and
#validation error rates and standard deviations, the optimal value of C, and the 
#test set error rate for the model with the lowest validation error rate
def SVM_dual(dataset:str):

	#Set up data
	cArr = [10e-4,10e-3,10e-2,0.1,1,10,100,1000]
	X_train, y_train, X_test, y_test = loadData(dataset)

	#Get 10 folds using k-folds
	kf = KFold(n_splits=10)
	#Get optimal C through cross validation on training data
	optimalC=0
	lowestErrorRate=1
	#For each C value perform cross validation
	for c in cArr:
		cTestErrorRates = []
		cTrainErrorRates=[]
		cTestErrorRate=0
		cTrainErrorRate=0
		#for each fold train and validate a model
		for train_index, test_index in kf.split(X_train):
			#Get our fold train and test data
			X_k_train, X_k_test = X_train[train_index], X_train[test_index]
			y_k_train, y_k_test = y_train[train_index], y_train[test_index]
			#Calculate our decision boundary based on our fold training data
			supportVectors, w, b = calculateDecisionBoundary(c, X_k_train, y_k_train)
			#Make predictions on our fold train and validation data using our learned w and b
			testPredictions = predict(w, X_k_test, b)
			trainPredictions=predict(w, X_k_train,b)
			#Calculate our error rates and record them
			testErrorRate = 1-metrics.accuracy_score(y_k_test, testPredictions)
			trainErrorRate=1-metrics.accuracy_score(y_k_train, trainPredictions)
			cTestErrorRates = np.append(cTestErrorRates, testErrorRate)
			cTrainErrorRates = np.append(cTrainErrorRates, trainErrorRate)
			cTestErrorRate+=testErrorRate
			cTrainErrorRate+=trainErrorRate

		#Average our error rates	
		cTestErrorRate = cTestErrorRate/10
		cTrainErrorRate = cTrainErrorRate/10

		#Choose our optimal C, with the lowest validation error rate
		if(cTestErrorRate<lowestErrorRate):
			lowestErrorRate = cTestErrorRate
			optimalC = c

		print("The C value is: ", c)
		print("The average training error rate is: ", cTrainErrorRate)
		print("The standard deviation of training error rates is: ", np.std(cTrainErrorRates))
		print("The average validation error rate is: ", cTestErrorRate)
		print("The standard deviation of testing error rates is: ", np.std(cTestErrorRates))
		print("\n")


	print("The optimal C is: ", optimalC)
	print("The error rate for the optimal C is: ", lowestErrorRate)
	print("\n")


	#Train SVM with all train data and apply to test data with optimal C
	supportVectors_star, w_star, b_star = calculateDecisionBoundary(optimalC, X_train, y_train)
	finalTestPredictions = predict(w_star, X_test, b_star)
	errorRate = 1-metrics.accuracy_score(y_test, finalTestPredictions)
	print("The final error rate for the test data with the optimal value of C IS...",errorRate,"!!!")



if __name__ == '__main__':
	SVM_dual('hw2_data_2020.csv')



