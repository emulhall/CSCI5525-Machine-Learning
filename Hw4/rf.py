import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
from matplotlib import pyplot as plt
from scipy.stats import mode

#Load in the cancer dataset
def load_data(dataset:str):
	#Read in dataset
	df = pd.read_csv(dataset, header=None,index_col=False, skip_blank_lines=True, na_values='?')

	for c in df:
			if(df[c].isnull().values.any()):
				df[c].fillna(df[c].mode()[0], inplace=True)

	#We are removing the ID column because while it has high information gain, it leads to poor test performance			
	df=df.drop(0,axis=1)

	#Replace the 2 and 4 with -1 and 1
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

#Build our tree downward by splitting from our given node
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

#Build up our tree starting at the root
def build_tree(df, num_features, max_depth, min_size):
	columns=df.columns.values
	options=np.array(columns[:-1])
	features=np.random.choice(options, num_features, replace=False)
	root=get_best_split(df, features)
	branch_downward(root, features, max_depth, min_size, 1)
	return root

#Use our decision tree to make a prediction
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


#Build a forest of learners and make predictions on our training and testing data
def build_forest(train, test, num_learners, num_features):
	T=num_learners
	train_predictions=[]
	test_predictions=[]
	train_error=[]
	test_error=[]
	for t in range(T):
		#Get a sample to train on
		train_sample=train.sample(frac=1, replace=True)

		#Create a decision tree
		tree=build_tree(train_sample, num_features, 1, 1)
		temp_train_pred=[]
		temp_test_pred=[]

		#Get predictions on training data
		for index, row in train.iterrows():
			temp_train_pred.append(decision_tree_predict(tree,row))
		train_predictions.append(temp_train_pred)

		#Get predictions on testing data
		for index, row in test.iterrows():
			temp_test_pred.append(decision_tree_predict(tree,row))
		test_predictions.append(temp_test_pred)

		t_train_predictions=mode(train_predictions,axis=0)[0][0]
		t_test_predictions=mode(test_predictions, axis=0)[0][0]

		t_train_error=1-accuracy_score(train[10],t_train_predictions)
		train_error.append(t_train_error)
		print("For %s weak learners, the train error rate is %s" % (t,t_train_error))

		t_test_error=1-accuracy_score(test[10],t_test_predictions)
		test_error.append(t_test_error)
		print("For %s weak learners, the test error rate is %s" % (t,t_test_error))

	return train_error, test_error


def rf(dataset:str):
	train, test = load_data(dataset)
	train = train.reset_index(drop=True)
	test=test.reset_index(drop=True)

	#Part 1 - Use three randomly sampled features to split our trees
	m_3_train_error, m_3_test_error=build_forest(train, test, 100, 3)

	m_3_test_fig=plt.figure()
	m_3_test_fig_axes=m_3_test_fig.add_axes([0.1,0.1,0.8,0.8])
	m_3_test_fig_axes.plot(m_3_test_error)
	m_3_test_fig_axes.set_xlabel('Number of Weak Learners')
	m_3_test_fig_axes.set_ylabel('Error Rate')
	m_3_test_fig.savefig('part_1_test_rf_plot')

	m_3_train_fig=plt.figure()
	m_3_train_fig_axes=m_3_train_fig.add_axes([0.1,0.1,0.8,0.8])
	m_3_train_fig_axes.plot(m_3_train_error)
	m_3_train_fig_axes.set_xlabel('Number of Weak Learners')
	m_3_train_fig_axes.set_ylabel('Error Rate')
	m_3_train_fig.savefig('part_1_train_rf_plot')

	m_possibilities=list(range(1,9))

	train_error_rates=[]
	test_error_rates=[]

	#Part 2 - Vary the number of features used to split the trees
	for m in m_possibilities:
		print('\n')
		print("The number of features is %s" % m)
		print('\n')
		train_error, test_error=build_forest(train, test, 100, m)
		train_error_rates.append(train_error[-1])
		test_error_rates.append(test_error[-1])

	test_fig=plt.figure()
	test_fig_axes=test_fig.add_axes([0.1,0.1,0.8,0.8])
	test_fig_axes.plot(m_possibilities, test_error_rates)
	test_fig_axes.set_xlabel('Number of Features')
	test_fig_axes.set_ylabel('Error Rate')
	test_fig.savefig('part_2_test_rf_plot')

	train_fig=plt.figure()
	train_fig_axes=train_fig.add_axes([0.1,0.1,0.8,0.8])
	train_fig_axes.plot(m_possibilities, train_error_rates)
	train_fig_axes.set_xlabel('Number of Features')
	train_fig_axes.set_ylabel('Error Rate')
	train_fig.savefig('part_2_train_rf_plot')


if __name__ == '__main__':
	rf('breast-cancer-wisconsin.data')