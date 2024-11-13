
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import cvxopt
import cvxopt.solvers

cvxopt.solvers.options["show_progress"] = False
#############################################################################
# Preparing the MNIST dataset for SVM classifcation
#############################################################################
train_digits = sio.loadmat('data_minst.mat')
# The dataset has two compoents: train_data and train_labels
# Train_data has 5000 digits images, and the dimension of each image is 28 * 28 (784 * 1)
# Train_lables has 5000 labels, 
train_data = train_digits['train_feats']
train_data -= np.mean(train_data,axis=0)
train_data /= np.std(train_data)
# There are totally 5000 images and the dimension of each image is 784 * 1
num_train_images = train_data.shape[0]
dim_images = train_data.shape[1]

train_bias = np.ones([num_train_images,1])
data_x = np.concatenate((train_data, train_bias), axis=1)

# Generating the labels for the one-verus-all logistic regression
labels = train_digits['train_labels']
data_y = np.zeros([num_train_images,10])-1
for i in range(num_train_images):
	data_y[i,labels[i][0]-1] = 1


train_samples = int(round(data_x.shape[0]*0.95))
train_data_x = data_x[:train_samples]
train_data_y = data_y[:train_samples]

print('The dimension of train dataset is [%d, %d]' %(train_data_x.shape[0],train_data_x.shape[1]))
print('The dimension of train labels  is [%d, %d]' %(train_data_y.shape[0],train_data_y.shape[1]))

test_data_x = data_x[train_samples:]
test_data_y = data_y[train_samples:]
#print(test_data_y)
#print(labels[train_samples:]-1)

print('The dimension of testing dataset is [%d, %d]' %(test_data_x.shape[0],test_data_x.shape[1]))
print('The dimension of testing labels  is [%d, %d]' %(test_data_y.shape[0],test_data_y.shape[1]))



#############################################################################
# The SVM Wolfe dual problem solution
# The input dataset: x_batch with dimesnion (n_samples, n_features) 
# The input dataset: y_batch with dimension (n_samples,1)
# The input value: C is the margin defined in 3.2 slides page 8 to 9
# The output vector: alpha with deimension (n_smaples,1) is defined in 3.2 slides page 4
# Note: the svm_dual function is implement based on the cone programming function in the cvxopt package (https://cvxopt.org/userguide/coneprog.html). 
# Note: comparing 3.2 slide page 4, you can understand how to solve SVM Wolfe dual based on the cone programming.
#############################################################################

def svm_dual(x_batch, y_batch, C):

    n_samples = x_batch.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i,j] = np.dot(x_batch[i,:2], x_batch[j,:2])
    
    P = cvxopt.matrix(np.outer(y_batch,y_batch) * K)
    q = cvxopt.matrix(np.ones(n_samples) * -1)
    A = cvxopt.matrix(y_batch, (1,n_samples))
    b = cvxopt.matrix(0.0)

    tmp1 = np.diag(np.ones(n_samples) * -1)
    tmp2 = np.identity(n_samples)
    G = cvxopt.matrix(np.vstack((tmp1, tmp2)))
    tmp1 = np.zeros(n_samples)
    tmp2 = np.ones(n_samples) * C
    h = cvxopt.matrix(np.hstack((tmp1, tmp2)))

    # solve QP problem
    print('SVM Wolfe Dual problem is solving ...')
    solution = cvxopt.solvers.qp(P, q, G, h, A, b)
    alpha = np.ravel(solution['x'])

    return alpha

#############################################################################
# The classifcation_ratio function
# The input weights: beta with dimension (n_features, 1). It includes the bias
# The input dataset: x_batch with dimesnion (n_samples, n_features) 
# The input dataset: y_batch with dimension (n_samples,1)
# The output scalar: accuracy is the SVM classifcation accuracy
# The output vector: svm_classifer is a vector wit dimension (n_samples,1). If a sample belongs to the first linear model, svm_classier = 1, otherwise svm_classifer = 0
# Note: 3.1 slide page 5 shows how to derive SVM classifcation based on signed distances
#############################################################################

#Taken from svm_example.py
def multiclass_ratio(p_batch, y_batch):
    actual_batch = np.argmax(y_batch, axis=1)
    class_batch = np.argmax(p_batch, axis=1)
    numerator = np.sum(actual_batch == class_batch)
    denominator = (actual_batch == class_batch).size
    ratio = numerator / float(denominator)
    return ratio
#############################################################################
#                       Main Function
#############################################################################
# In the computer assignment, you should complete the main functino by yourself
# In the main function, you need to finish three tasks
# (1) use the one-vs-all method to train SVM models to get the corresponding weights.
# (2) use the one-vs-all method to test the learned SVM modesl to check the learned models.
# (3) obtain the training accuracy and testing accuracy during the above two tasks.
# Note:
# (0) you should install the cvxopt package into your environment, the command is 'conda install -c conda-forge cvxopt'
# (1) you can reuse the code in the svm_example.py file to solve Wolfe Dual problem for learning weights.
# (2) MNIST dataset is non-sperable case, you can refer 3-2 slides Page 7-8 to complete the classifcation.
# (3) you can refer 3-1 slides Page 5-8 to design classifcation accuracy function.
# (4) The training accuracy should be around 80% and the testing accuarcy should be around 60%.
# (5) you can resue the code from Model 2 to solve the one-vs-all problem.
# (6) you should use the pre-defined training dataset and testing dataset to train and to test SVM models

C = [.00001, 0.05, 10]
K = 10
beta = [np.reshape(np.zeros(K * train_data_x.shape[1]), (K, train_data_x.shape[1])) for i in range(len(C))]
beta = np.array(beta)
train_svm_score = [0] * len(C)
test_svm_score = [0] * len(C)

for i in range(len(C)):
    ratios = np.zeros(K)
    for j in range(K):
        class_label = train_data_y[:, j]
        alpha = svm_dual(train_data_x, class_label, C[i])
        sv = alpha > 1e-5
        alpha_nz = alpha[sv]
        data_nz = train_data_x[sv]
        label_nz = class_label[sv]

        # Derive the weights of the SVM classifer based on SVM Wolfe dual (3.2 Slides, Page 4)
        n_features = data_nz.shape[1]
        n_samples = data_nz.shape[0]
        for k in range(n_samples):
            beta[i][j] += alpha_nz[k] * label_nz[k] * data_nz[k]

#############################################################################
# Training SVM models to get the weights
for i in range(len(C)):
    svm_train_accuracy = [np.zeros(K) for _ in range(train_data_y.shape[0])]
    svm_train_accuracy = np.array(svm_train_accuracy)
    for j in range(K):
        svm_train_accuracy[:, j] = np.matmul(train_data_x, beta[i][j])
    train_ratio = multiclass_ratio(svm_train_accuracy, train_data_y)
    print(f'C: {C[i]}, The svm training classification accuracy is %.2f%%'%(train_ratio*100))
# Testing SVM models
for i in range(len(C)):
    svm_test_accuracy = [np.zeros(K) for _ in range(test_data_y.shape[0])]
    svm_test_accuracy = np.array(svm_test_accuracy)
    for j in range(K):
        svm_test_accuracy[:, j] = np.matmul(test_data_x,beta[i][j])
    test_ratio = multiclass_ratio(svm_test_accuracy, test_data_y)
    print(f'C: {C[i]}, The svm testing classification accuracy is %.2f%%'%(test_ratio*100))