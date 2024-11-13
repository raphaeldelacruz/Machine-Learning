

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

#############################################################################
# Generate three seperate clusters
#############################################################################
n = 1000
x1 = np.random.randn(n,2) + np.matlib.repmat([4,4], n, 1)
x2 = np.random.randn(n,2) + np.matlib.repmat([4,12], n, 1)
x3 = np.random.randn(n,2) + np.matlib.repmat([10,8], n, 1)
data = np.concatenate((x1, x2, x3), axis=0)

def get_centroids(data, K, centroids):
    n_samples = data.shape[0]
    n_features = data.shape[1]
    new_centroids = np.zeros((K, n_features))
    classes = np.zeros(n_samples, dtype=int)
    d = np.zeros((K, n_samples))
    for i in range(K):
        d[i] = np.sum(np.square(data - centroids[i]), axis=1)
        classes = np.argmin(d, axis=0)
    for i in range(K):
        new_centroids[i] = np.mean(data[classes == i], axis=0)
    return new_centroids, classes

#############################################################################
#                       Main Function
#############################################################################
# Hyper-parameters
K = 3
epochs = 100

# Initialize the centroids
data_min = np.min(data)
data_max = np.max(data)
print('the min is %.2f' % (data_min))
print('the max is %.2f' % (data_max))
centroids = np.random.randint(low=data_min,high=data_max,size = (K,2))
print(centroids)

# Define the centroids vector for visualizing the training procedure
centroids_vector = np.zeros((epochs,K,2))

# Training loop for K-means clustering
for epoch in range(epochs):
	centroids,classes = get_centroids(data,K,centroids)
	centroids_vector[epoch] = centroids

#############################################################################
# Visualizing the dataset and the learned K-mean model
group_colors = ['skyblue', 'coral', 'lightgreen']
colors = [group_colors[j] for j in classes]

fig, _axs = plt.subplots(nrows=1, ncols=2,figsize=(8,6))
ax = _axs.flatten()

ax[0].scatter(data[:,0], data[:,1])
ax[0].set_title('The orignial dataset with 3 clusters')


ax[1].scatter(data[:,0], data[:,1], color=colors, alpha=0.5)
ax[1].scatter(centroids_vector[epochs-1][:,0], centroids_vector[epochs-1][:,1], color=['blue', 'darkred', 'green'], marker='o', lw=2)

l31 = ax[1].scatter(centroids_vector[:,0,0],centroids_vector[:,0,1], c=range(epochs), vmin=1, vmax=epochs, cmap='autumn',marker = 'v')
l32 = ax[1].scatter(centroids_vector[:,1,0],centroids_vector[:,1,1], c=range(epochs), vmin=1, vmax=epochs, cmap='autumn',marker = 'v')
l33 = ax[1].scatter(centroids_vector[:,2,0],centroids_vector[:,2,1], c=range(epochs), vmin=1, vmax=epochs, cmap='autumn',marker = 'v')

ax[1].plot(centroids_vector[:,0,0],centroids_vector[:,0,1],alpha=0.5,color ='k')
ax[1].plot(centroids_vector[:,1,0],centroids_vector[:,1,1],alpha=0.5,color ='k')
ax[1].plot(centroids_vector[:,2,0],centroids_vector[:,2,1],alpha=0.5,color ='k')
ax[1].set_xlabel('$x_0$')
ax[1].set_ylabel('$x_1$');
cbar = fig.colorbar(l31, ax=ax[1])
cbar.set_label('Epoch')
ax[1].set_title('The final dataset with 3 clusters')
#cbar.ax.set_yticklabels(['Start', '0', 'End'])


plt.show()