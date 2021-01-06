import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

#1 channel input 28x28
#Fully connected layer 1: input with bias; output 128 nodes
#ReLU activation function
#Fully connected layer 2 input - 128 nodes; output 10 nodes
#Softmax activation function
#Use cross entropy as the loss function
#Use SGD as optimizer
#Set mini batch size as 32

def neural_net():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#Convert samples from integers to floats
	x_train=x_train/255.0
	x_test=x_test/255.0

	#Add our layers to our model
	#First layer is our 28x28 input layer
	#Second layer is our hidden layer with 128 nodes and relu activation
	#Output layer is 10 nodes and softmax activation function
	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=(28,28)),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dense(10,activation='softmax')])

	#Apply our model to our training data
	predictions=model(x_train).numpy()
	#Define our cross entropy loss
	ce_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	#Compile our model with our SGD optimizer and cross-entropy loss
	model.compile(optimizer='sgd', loss=ce_loss, metrics=['accuracy'])
	#Add a callback to stop when our validation loss is no longer improving
	#because we do not want to overfit to our training data I've chosen to stop based on a validation subset of the training data
	callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
	#now we can make our number of epochs quite large since our model will stop itself when our loss stops decreasing
	#train our model
	history=model.fit(x_train,y_train, batch_size=32, validation_split=0.2, callbacks=[callback], epochs=100000000)
	#evaluate our model on our testing data
	test_loss, test_acc=model.evaluate(x_test,y_test,verbose=2)

	#uncomment below if you would not like your own plots
	loss_fig=plt.figure()
	loss_axes=loss_fig.add_axes([0.1,0.1,0.8,0.8])
	loss_axes.plot(history.history['loss'])
	loss_axes.set_xlabel('Epochs')
	loss_axes.set_ylabel('Loss')
	loss_fig.savefig('nnLoss.png')

	accuracy_fig=plt.figure()
	accuracy_axes=accuracy_fig.add_axes([0.1,0.1,0.8,0.8])
	accuracy_axes.plot(history.history['accuracy'])
	accuracy_axes.set_xlabel('Epochs')
	accuracy_axes.set_ylabel('Accuracy')
	accuracy_fig.savefig('nnAccuracy.png')

if __name__ == '__main__':
	neural_net()