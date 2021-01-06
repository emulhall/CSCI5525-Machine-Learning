import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from TimeHistory import TimeHistory

def buildModel(x_train, y_train, x_test, y_test, batch_size, optimizer):
		#Add layers to our model
		#input channel of 28x28
		#convolution layer: convolution kernel size is (3,3) with stride 1. Input channels - 1; output channels - 20 nodes
		#number of filters is not defined in the assignment, so we will go with 1
		#Max-pool 2x2 max pool
		#dropout layer with probability p=.50
		#flatten input for feed to fully connected layers
		#fully connected layer 1: flattened input with bias; output - 128 nodes
		#ReLU activation function
		#dropout layer with probability .5
		#fully connected layer 2: input - 128 nodes; output - 10 nodes
		#softmax activation function
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(1, kernel_size=3, activation='relu', input_shape=(28,28,1), strides=(1,1)),
		tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(128, activation='relu'),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(10,activation='softmax')])

	#Apply our model to our training data
	predictions=model(x_train).numpy()
	#Define our cross entropy loss
	ce_loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	#Compile our model with our SGD optimizer and cross-entropy loss
	model.compile(optimizer=optimizer, loss=ce_loss, metrics=['accuracy'])
	#Add a callback to stop when our validation loss is no longer improving
	#because we do not want to overfit to our training data I've chosen to stop based on a validation subset of the training data
	early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=2)
	#Add a callback to track the amount of time it takes to converge
	time_callback = TimeHistory()
	#train our model
	history=model.fit(x_train,y_train, batch_size=batch_size, validation_split=0.2,epochs=100000000, callbacks=[early_stop_callback, time_callback])
	#evaluate our model on our testing data
	test_loss, test_acc=model.evaluate(x_test,y_test,verbose=2)

	return history, test_loss, test_acc, time_callback.convergence_time


def cnn():
	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	#Convert samples from integers to floats
	x_train=x_train/255.0
	x_test=x_test/255.0
	#Reshape it to fit our cnn - we need to add our channel dimension, which is 1 because it is grayscale
	x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
	x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_train.shape[2],1)

	#first train using SGD as the optimizer and mini batches of size 32. Plot the cumulative training loss and accuracy for every epoch.
	#evaluate our model on our testing data
	history1, test_loss1, test_acc1, convergence_time=buildModel(x_train,y_train,x_test, y_test,32,'sgd')

	#comment out below if you would not like your own plots
	loss_fig1=plt.figure()
	loss_axes1=loss_fig1.add_axes([0.1,0.1,0.8,0.8])
	loss_axes1.plot(history1.history['loss'])
	loss_axes1.set_xlabel('Epochs')
	loss_axes1.set_ylabel('Loss')
	loss_fig1.savefig('cnnLoss1.png')

	accuracy_fig1=plt.figure()
	accuracy_axes1=accuracy_fig1.add_axes([0.1,0.1,0.8,0.8])
	accuracy_axes1.plot(history1.history['accuracy'])
	accuracy_axes1.set_xlabel('Epochs')
	accuracy_axes1.set_ylabel('Accuracy')
	accuracy_fig1.savefig('cnnAccuracy1.png')

	#Second, train your network using mini batch sizes of [32, 64, 96, 128] 
	#and plot the convergence run time vs mini batch sizes for each of the following optimizers: SGD, Adagrad, and Adam.
	batch_sizes=[32,64,96,128]
	optimizers=['sgd','adagrad','adam']

	for o in optimizers:
		run_times=[]
		for b in batch_sizes:
			history, test_loss, test_acc, convergence_time=buildModel(x_train,y_train,x_test, y_test,b,o)
			run_times.append(convergence_time)

		#comment out below if you would not like your own plots
		fig=plt.figure()
		axes=fig.add_axes([0.1,0.1,0.8,0.8])
		axes.plot(batch_sizes,run_times)
		axes.set_xlabel('Batch Size')
		axes.set_ylabel('Run time (s)')
		axes.set_title(o)
		fig.savefig(o+'.png')



if __name__ == '__main__':
	cnn()