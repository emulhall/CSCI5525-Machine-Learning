import random
import numpy as np
from matplotlib import pyplot as plt

def load_dataset(image:str):
	img=plt.imread(image)

	#let's just get rid of the A value, since we don't need that
	img=img[:,:,:3]
	return img

#Calculate the cumulative loss
def calculate_distortion(r, mu, x):
	N=len(x)
	K=len(mu)

	J=0

	for n in range(N):
		for k in range(K):
			if r[n]==k:
				J+=np.square(np.linalg.norm(x[n]-mu[k]))
	return J

#Randomly initialize the centers
def initialize_mu(X, k):
	mu_list=random.sample(list(X),k)
	return mu_list

#Compute the difference between two spots
def euclidean_distance(x,y):
	total=0
	for i in range(len(x)):
		total+=np.square(x[i]-y[i])
	return np.sqrt(total)

#Update which center the points are assigned to by computing the euclidean distance
def update_r(X, mu):
	r=np.zeros(len(X))
	for i in range(len(X)):
		min_dist=9999
		closest_center=None
		for k in range(len(mu)):
			k_dist=euclidean_distance(X[i],mu[k])
			if k_dist<min_dist:
				min_dist=k_dist
				closest_center=k
		r[i]=closest_center
	return r


#Update the mean values by taking the average of the points assigned to that class
def update_mu(X, r, k):
	mu=np.zeros((k,3))
	for i in range(k):
		c=np.zeros(3)
		class_count=0
		for n in range(len(X)):
			#Sum over the points assigned to that class
			if(r[n]==i):
				c=c+X[n]
				class_count+=1
		#Preventing divide by zero
		if class_count==0:
			class_count=1
		#Divide by the number of points assigned to that class to get the average
		c/=class_count
		mu[i]=c
	return mu


#Run the k-means algorithm over an image and print the loss over iterations
def kmeans(image:str):
	img=load_dataset(image)
	rows=img.shape[0]
	cols=img.shape[1]
	X=np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))


	max_iterations=100000000000000
	k_list=[3,5,7]
	for k in k_list:
		loss=[]
		centers=initialize_mu(X,k)
		loss_threshold=0.001
		for i in range(max_iterations):
			r=update_r(X, centers)
			j=calculate_distortion(r,centers,X)
			centers=update_mu(X, r, k)
			print("For iteration:%s, number of centers:%s, the distortion measure is %s" % (i,k,j))
			loss.append(j)
			if(len(loss)>1):
				if(abs(loss[i]-loss[i-1])<loss_threshold):
					break

		#Save the compressed image			
		final_r=update_r(X, centers)
		recovered=centers[final_r.astype(int)]
		recovered=np.reshape(recovered, (rows, cols, 3))
		plt.imsave('q3_compressed_image_'+str(k)+'.png',recovered)

		#Plot the cumulative loss over the iterations for this particular number of prototypes
		fig=plt.figure()
		fig_axes=fig.add_axes([0.1,0.1,0.8,0.8])
		fig_axes.plot(loss)
		fig_axes.set_xlabel('Number of Iterations')
		fig_axes.set_ylabel('Cumulative Loss')
		fig_axes.set_title('Loss Over Iterations For '+ str(k)+' Prototypes')
		fig.savefig('q3_k'+str(k))


if __name__ == '__main__':
	kmeans('umn_csci.png')