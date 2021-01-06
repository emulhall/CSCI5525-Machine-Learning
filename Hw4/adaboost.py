import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
from matplotlib import pyplot as plt

#Load in the cancer dataset
def load_data(dataset:str):
	#Read in dataset
	df = pd.read_csv(dataset, header=None,index_col=False, skip_blank_lines=True, na_values='?')

	for c in df:
			if(df[c].isnull().values.any()):
				df[c].fillna(df[c].mode()[0], inplace=True)

	#We are removing the ID column because while it has high information gain, it leads to poor test performance			
	df=df.drop(0,axis=1)

	#Replace the 2 and 4 with -1 and 1 for later calculations
	df[10].replace(2,-1,inplace=True)
	df[10].replace(4,1,inplace=True)

	#Split into train and test
	train, test = train_test_split(df, test_size=0.2, shuffle=True)
	return train, test

#Calculate the entropy of a dataset
def entropy(df):
	classes,class_counts=np.unique(df[10],return_counts=True)
	h=0
	for i in range(len(classes)):
		h+=(class_counts[i]/np.sum(class_counts)*np.log2(class_counts[i]/np.sum(class_counts)))
	return -h

#Calculate the information gain of a split
def information_gain(df_before_split, splits):
	total=entropy(df_before_split)
	for s in splits:
		total-=entropy(s)*len(s)
	return total

#Split a dataset by some feature at some value
def split(df, feature, value):
	left=df[df[feature]<=value]
	right=df[df[feature]>value]
	return left, right

#Find the split with the highest information gain
def get_best_split(df, features):
	max_info_gain=-999
	split_feature=None
	split_value=None
	for f in features:
		f_values=np.unique(df[f])
		for v in f_values:
			splits=split(df, f, v)
			gain=information_gain(df, splits)
			if gain>max_info_gain:
				split_feature=f
				split_value=v
				max_info_gain=gain
	return {'feature':split_feature, 'value':split_value, 'branches':split(df,split_feature, split_value)}

#Get the classification of a leaf.  It will be the class of the samples that occurs most.
def terminal_node(df):
	return df[10].mode()[0]

#Branch out our decision tree
def branch_downward(node, features, max_depth, min_size, depth):
	left, right=node['branches']
	del(node['branches'])
	#Check for max depth
	if depth>=max_depth:
		node['left']=terminal_node(left)
		node['right']=terminal_node(right)
		return
	#If we're not at our max depth, let's keep branching
	#Process left branch
	if len(left)<=min_size:
		node['left']=terminal_node(left)
	else:
		node['left'] = get_best_split(left)
		branch_downward(node['left'], features, max_depth, min_size, depth+1)

	#Process right branch
	if len(right)<=min_size:
		node['right']=terminal_node(right)
	else:
		node['right'] = get_best_split(right)
		branch_downward(node['right'], features, max_depth, min_size, depth+1)

#Build our tree by starting at the root and branching downward
def build_tree(df, max_depth, min_size):
	columns=df.columns.values
	features=columns[:-1]
	root=get_best_split(df, features)
	branch_downward(root, features, max_depth, min_size, 1)
	return root

#Use our tree to classify a sample
def decision_tree_predict(node, sample):
	if sample[node['feature']]<node['value']:
		if isinstance(node['left'], dict):
			return decision_tree_predict(node['left'], sample)
		else:
			return node['left']

	else:
		if isinstance(node['right'], dict):
			return decision_tree_predict(node['right'], sample)
		else:
			return node['right']

#Takes in a dataset and runs the adaboost algorithm on it
def adaboost(dataset:str):
	train, test = load_data(dataset)
	train = train.reset_index(drop=True)
	test=test.reset_index(drop=True)
	T=100
	test_error_rates=[]
	train_error_rates=[]

	#Initialize our arrays
	n=len(train)
	sample_weights=np.zeros(shape=(T,n))
	stumps=[]
	stump_weights=np.zeros(shape=T)
	errors=np.zeros(shape=T)

	#Initialize weights uniformly
	sample_weights[0]=np.ones(shape=n)/n

	#Train our model with our number of weak learners
	for t in range(T):

		current_sample_weights = sample_weights[t]

		#get a sample to train on
		train_sample=train.sample(frac=1, replace=True, weights=current_sample_weights)

		#Train weak learner
		tree=build_tree(train_sample, 1, 1)
		output=[]
		for index, row in train.iterrows():
			output.append(decision_tree_predict(tree, row))


		#Calculate error
		err=0
		for index, row in train.iterrows():
			w=current_sample_weights[index]
			if output[index]!=row[10]:
				err+=w

		#Calculate stump weight
		stump_weight=0.5*np.log((1-err)/err)

		#Update weights
		new_sample_weights=np.zeros(shape=len(current_sample_weights))
		for index, row in train.iterrows():
			new_sample_weights[index]=current_sample_weights[index]*(np.exp(-stump_weight*output[index]*row[10]))

		new_sample_weights/=new_sample_weights.sum()

		if t+1<T:
			sample_weights[t+1]=new_sample_weights

		stumps.append(tree)
		stump_weights[t]=stump_weight
		errors[t]=err

		#Now that the  model has been built, let's finally make a prediction
		test_prediction=np.zeros(shape=len(test))
		train_prediction=np.zeros(shape=len(train))
		for s in range(len(stumps)):
			temp_test_pred=[]
			for index, row in test.iterrows():
				temp_test_pred.append(decision_tree_predict(stumps[s], row))
			test_prediction=test_prediction+np.array(temp_test_pred)*stump_weights[s]


			temp_train_pred=[]
			for index, row in train.iterrows():
				temp_train_pred.append(decision_tree_predict(stumps[s], row))
			train_prediction=train_prediction+np.array(temp_train_pred)*stump_weights[s]

		for j in range(len(test_prediction)):
			test_prediction[j]=np.sign(test_prediction[j])

		for j in range(len(train_prediction)):
			train_prediction[j]=np.sign(train_prediction[j])

		#Final error
		train_error=1-accuracy_score(train[10], train_prediction)
		print("For %s weak learners, the train error rate is %s" % (t,train_error))
		train_error_rates.append(train_error)


		test_error=1-accuracy_score(test[10], test_prediction)
		print("For %s weak learners, the test error rate is %s" % (t,test_error))
		test_error_rates.append(test_error)

	#Plot our results.  Please comment out if you do not want your own plots to be generated.	
	test_fig=plt.figure()
	test_fig_axes=test_fig.add_axes([0.1,0.1,0.8,0.8])
	test_fig_axes.plot(test_error_rates)
	test_fig_axes.set_xlabel('Number of Weak Learners')
	test_fig_axes.set_ylabel('Error Rate')
	test_fig.savefig('test_adaboost_plot')

	train_fig=plt.figure()
	train_fig_axes=train_fig.add_axes([0.1,0.1,0.8,0.8])
	train_fig_axes.plot(train_error_rates)
	train_fig_axes.set_xlabel('Number of Weak Learners')
	train_fig_axes.set_ylabel('Error Rate')
	train_fig.savefig('train_adaboost_plot')



if __name__ == '__main__':
	adaboost('breast-cancer-wisconsin.data')