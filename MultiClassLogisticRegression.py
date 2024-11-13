
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt


#############################################################################
# Preparing the MNIST dataset for logistic regression
train_digits = sio.loadmat('data_minst.mat')
# The dataset has two compoents: train_data and train_labels
# Train_data has 5000 digits images, and the dimension of each image is 28 * 28 (784 * 1)
# Train_lables has 5000 labels, 
train_data = train_digits['train_feats']
# There are totally 5000 images and the dimension of each image is 784 * 1
num_train_images = train_data.shape[0]
dim_images = train_data.shape[1]

train_bias = np.ones([num_train_images,1])
data_x = np.concatenate((train_data, train_bias), axis=1)

# Generating the labels for the one-verus-all logistic regression
labels = train_digits['train_labels']
data_y = np.zeros([num_train_images,10])
for i in range(num_train_images):
	data_y[i,labels[i][0]-1] = 1

print(data_x.shape)
print(data_y.shape)

#############################################################################
# The logistic function
def logistic(x,beta):
	y_pred = np.matmul(x,beta)
	logistic_prob = 1/(1 + np.exp(-y_pred))
	return logistic_prob

#############################################################################
# Here you create you own LOGISTIC REGRESSION algorith for the logistic regression model
# Hint: please check the slides of the logistic regression 
#############################################################################

def logistic_regression(beta, lr, x_batch, y_batch,lambda1):
	dNLL_dB = np.matmul(np.transpose(x_batch), (y_batch - logistic(x_batch, beta))) + lambda1 * beta
	beta_next = lr * dNLL_dB + beta
	cost_batch = np.matmul(x_batch, beta_next)
	cost_batch[cost_batch > 0] = 1
	cost_batch[cost_batch <= 0] = 0
	cost = np.sum(np.square(cost_batch - y_batch))
	return cost, beta_next

#############################################################################
 
def classifcation_ratio(beta,x_batch,y_batch):
	ratio = 0
	logistic_prob = logistic(x_batch,beta)
	clssifcation_result = logistic_prob > 0.5
	true_result = y_batch == 1
	comparison = np.logical_and(clssifcation_result,true_result)
	ratio = float(np.sum(comparison))/float(np.sum(true_result))
	return ratio 


#############################################################################
#                       Main Function
#############################################################################

# Hyper-parameters
training_epochs = 100
learning_rate = 0.0005          # The optimization initial learning rate
lambda1 = 0						# The regularization parameter
cost = 0


for i in range(1):
	current_label = data_y[:,0]
	print(current_label.shape)
	beta = np.random.randn(dim_images + 1)
	for epoch in range(training_epochs):
		cost, beta_next = logistic_regression(beta,learning_rate,data_x,current_label, lambda1)
		ratio = classifcation_ratio(beta,data_x,current_label)
		print('Class %d Epoch %3d, cost %.3f, the classification accuracy is %.2f%%' % (i+1, epoch+1,cost,ratio*100))
		beta = beta_next


#############################################################################
# Visualizing five images of the MNIST dataset
fig, _axs = plt.subplots(nrows=1, ncols=5)
axs = _axs.flatten()

for i in range(5):
	axs[i].grid(False)
	axs[i].set_xticks([])
	axs[i].set_yticks([])
	image = data_x[i*10,:784]
	image = np.reshape(image,(28,28))
	aa = axs[i].imshow(image,cmap=plt.get_cmap("gray"))

fig.tight_layout()
plt.show()

