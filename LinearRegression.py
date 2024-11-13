

import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#############################################################################
# ***************************************************************************
# you should create you own GRADIENT DESCENT algorithm in the location specified below
# ***************************************************************************
#############################################################################

#############################################################################
# Dataset Prepration 
data = pd.read_csv('Salary_Data.csv')
x_batch = data['YearsExperience']
y_batch = data['Salary']
n=30
def generate_dataset(W,b,n):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n)
    y_batch = W * x_batch + b + np.random.randn(*x_batch.shape)
    return x_batch, y_batch

#############################################################################

print(x_batch.shape)
print(y_batch.shape)

#############################################################################
# Hyper-parameters
training_epochs = 500
learning_rate = 0.0001            # The optimization initial learning rate

#############################################################################
# Here you create you own GRADIENT DESCENT algorith for the linear regression model
# Hint: please check the slides of the gradient descent (1.3, the 9th to 10th page)
#############################################################################

def gradient_descent(beta, lr, x_batch, y_batch):
    # Mean Squared Error Cost Function: cost
    cost = 0
    # The learned coefficients: beta_next
    weight_next = beta[0]
    bias_next = beta[1]
    beta_next = [weight_next,bias_next]
    
    y_pred = x_batch * weight_next + bias_next
    cost = (1/(2* y_batch.size)) * sum((y_pred - y_batch) ** 2)
    weight_next = weight_next - (lr * sum((y_pred - y_batch) * x_batch))
    bias_next = bias_next - (lr * sum(y_pred - y_batch))
    beta_next = [weight_next, bias_next]

    return cost, beta_next

#############################################################################


#############################################################################
#                       Main Function
#############################################################################
beta = np.random.randn(2)
cost = 0

for epoch in range(training_epochs):
    cost, beta_next = gradient_descent(beta,learning_rate,x_batch,y_batch)
    print('Epoch %3d, cost %.3f, weight %.3f bias is %.3f' % (epoch+1,cost,beta_next[0],beta_next[1]))
    beta = beta_next

#############################################################################
# Visualizing the dataset and the learned linear model
Data_s = sorted(zip(x_batch,y_batch))
X_s, Y_s = zip(*Data_s)
plt.plot(X_s,Y_s,'o')
plt.ylabel('Salary')
plt.xlabel('Years of Experience')
plt.title('The blue points are the dataset, the red line is the learned linear model')
plt.hold(True)

max_x = np.max(x_batch)
min_x = np.min(x_batch)
xx = np.linspace(min_x, max_x, n)

weight = beta[0]
bias = beta[1]

yy = weight * xx + bias
plt.plot(xx,yy,'r')
plt.show()



