
import numpy as np
import matplotlib.pyplot as plt

#############################################################################
# ***************************************************************************
# you should create you own LASSO algorithm in the location specified below
# ***************************************************************************
#############################################################################

#############################################################################
# The function to generate the dataset
def generate_dataset(n,W,b=0):
    #x_batch = np.linspace(0, 2, n)
    x_batch = np.random.randn(n,W.shape[0])
    y_batch =  x_batch.dot(W) + b + np.random.randn(n)*2
    return x_batch, y_batch

#############################################################################
# Preparing the dataset for linear regression
# The dimension of the dataset
n = 30
# The weight of the linear model y = Bx
Beta_true = np.array([10, -2, 0, -3, 0, 5, 0, 0])
x_batch, y_batch = generate_dataset(n,Beta_true)


#############################################################################
# Here you create you own LASSO algorithm for the linear regression model
# Hint: please check the slides of the gradient descent (1.5, the 7th to 14th page)
#############################################################################

def lasso_regression(parmsl, lamda, x_batch, y_batch):
    # Mean Squared Error Cost Function: cost
    cost = 0.0 
    weights = np.zeros(parmsl.shape[0])
    for i in range(x_batch.shape[1]):
        excludei_parmsl = np.delete(parmsl, i, 0)
        excludei_x_batch = np.delete(x_batch, i, 1)
        numerator = np.inner(x_batch[:, i].transpose(), (y_batch - np.matmul(excludei_x_batch, excludei_parmsl)))
        denominator = np.inner(x_batch[:, i].transpose(), x_batch[:, i])
        weights[i] = numerator/denominator

    # The learned coefficients: weight_next
    weights_next = weights

    for i in range(weights_next.shape[0]):
        xNormSq = np.linalg.norm(x_batch[:, i]) ** 2
        #Case 1: 𝛽𝑖 < 0 ⟹ 𝑔𝑖 − 𝜆 = 0
        if weights[i] < -(lamda / xNormSq):
            weights_next[i] = weights[i] + (lamda / xNormSq)
        #Case 2: 𝛽𝑖 > 0 ⟹ 𝑔𝑖 + 𝜆 = 0
        elif weights[i] > lamda / xNormSq:
            weights_next[i] = weights[i] - (lamda / xNormSq)
        #Case 3: 𝛽𝑖 = 0 ⟹ 0 ∈ 𝑔𝑖 − 𝜆, 𝑔𝑖 + 𝜆
        else:
            weights_next[i] = 0

    y_pred = np.matmul(x_batch, weights_next)

    cost = (np.linalg.norm(y_pred - y_batch)) ** 2

    return cost, weights_next


#############################################################################


#############################################################################
#                       Main Function
#############################################################################
# Hyper-parameters
training_epochs = 1000
weight_var = 1
#beta = np.random.randn(*Beta_true.shape)*weight_var
beta = np.zeros(*Beta_true.shape)
print(beta)

cost1 = 0
cost2 = 0
cost3 = 0

#lamda = 0.0001
lamda1 = 30
beta_show1 = np.zeros((training_epochs,Beta_true.shape[0]))
lamda2 = 3
beta_show2 = np.zeros((training_epochs,Beta_true.shape[0]))
lamda3 = 0.0001
beta_show3 = np.zeros((training_epochs,Beta_true.shape[0]))

#############################################################################
# Learning three lasso models given different lambda values

beta1 = beta
for epoch in range(training_epochs):   
    beta_show1[epoch] = beta1
    cost, beta_next = lasso_regression(beta1,lamda1,x_batch,y_batch)
    print('Epoch %3d, cost %.3f, beta[0] %.3f beta[1] is %.3f' % (epoch+1,cost,beta1[0],beta1[1]))
    beta1 = beta_next
cost1 = cost

beta2 = beta
print(beta)
for epoch in range(training_epochs):   
    beta_show2[epoch] = beta2
    cost, beta_next = lasso_regression(beta2,lamda2,x_batch,y_batch)
    print('Epoch %3d, cost %.3f, beta[0] %.3f beta[1] is %.3f' % (epoch+1,cost,beta1[0],beta1[1]))
    beta2 = beta_next
cost2 = cost

beta3 = beta
for epoch in range(training_epochs):   
    beta_show3[epoch] = beta3
    cost, beta_next = lasso_regression(beta3,lamda3,x_batch,y_batch)
    print('Epoch %3d, cost %.3f, beta[0] %.3f beta[1] is %.3f' % (epoch+1,cost,beta1[0],beta1[1]))
    beta3 = beta_next
cost3 = cost

#############################################################################
# Visualizing the dataset and the learned lasso model

fig, _axs = plt.subplots(nrows=1, ncols=3)
axs = _axs.flatten()

title_str = r'$\lambda$ %.2f, cost %.2f' % (lamda1,cost1)
axs[0].grid(True)
axs[0].set_title(title_str)
axs[0].semilogx(beta_show1)
axs[0].legend(('Beta0', 'Beta1', 'Beta2','Beta3', 'Beta4', 'Beta5','Beta6', 'Beta7'),
           loc='upper center')

title_str = r'$\lambda$ %.2f, cost %.2f' % (lamda2,cost2)
axs[1].grid(True)
axs[1].set_title(title_str)
axs[1].semilogx(beta_show2)

title_str = r'$\lambda$ %.4f, cost %.2f' % (lamda3,cost3)
axs[2].grid(True)
axs[2].set_title(title_str)
axs[2].semilogx(beta_show3)

plt.show()



