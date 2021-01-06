import tensorflow as tf
import time

class TimeHistory(tf.keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.start_time=time.time()

	def on_train_end(self, logs={}):
		self.convergence_time=time.time()-self.start_time