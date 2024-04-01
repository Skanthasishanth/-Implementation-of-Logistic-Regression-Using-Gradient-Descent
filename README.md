# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Use the standard libraries in python for finding linear regression.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Predict the values of array.
5. Calculate the accuracy, confusion and classification report b importing the required modules from sklearn.
6. Obtain the graph.

## Program:

```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: S Kantha Sishanth
RegisterNumber: 212222100020
```


```py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("ex2data1.txt",delimiter=",")
X = data[:,[0,1]]
Y = data[:,2]

X[:5]

Y[:5]

# VISUALIZING THE DATA
plt.figure()
plt.scatter(X[Y== 1][:, 0], X[Y==1][:,1],label="Admitted")
plt.scatter(X[Y==0][:,0],X[Y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction(theta, X, Y):
    h = sigmoid(np.dot(X, theta))
    J = -(np.dot(Y, np.log(h)) + np.dot(1-Y,np.log(1-h))) / X.shape[0]
    grad = np.dot(X.T, h-Y)/X.shape[0]
    return J,grad

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([0,0,0])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta = np.array([-24,0.2,0.2])
J,grad = costFunction(theta,X_train,Y)
print(J)
print(grad)

def cost(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  J=-(np.dot(Y,np.log(h))+np.dot(1-Y,np.log(1-h)))/X.shape[0]
  return J

def gradient(theta,X,Y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-Y)/X.shape[0]
  return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,Y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,Y):
    X_min , X_max = X[:, 0].min() - 1,X[:,0].max() + 1
    Y_min , Y_max = X[:, 1].min() - 1,X[:,1].max() + 1
    XX,YY = np.meshgrid(np.arange(X_min,X_max,0.1),
                        np.arange(Y_min,Y_max,0.1))
    X_plot = np.c_[XX.ravel(), YY.ravel()]
    X_plot = np.hsatck((np.ones((X_plot.shape[0],1)),X_plot))
    Y_plot = np.dot(X_plot, theta).reshape(XX.shape)
    plt.figure()
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Admitted')
    plt.scatter(X[Y==1][:,0],X[Y==1][:,1],label='Not admitted')
    plt.contour(XX,YY,Y_plot,levels=[0])
    plt.Xlabel("Exam 1 score")
    plt.Ylabel("Exam 2 score")
    plt.legend()
    plt.show()

print("Decision boundary-graph for exam score:")
plotDecisionBoundary(res.x,X,Y)


prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)

```

## Output:

### Array value of X:

![ML_1](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/2bcb820f-a9ea-47bf-856b-168d3abf1dd1)


### Array value of Y:

![ML_2](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/fdded343-81e5-4be1-80ae-ce201b3d6184)


### Exam 1-Score graph:

![ML_3](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/43c86198-49ca-428b-b8ef-64f4816a0cbf)


### Sigmoid function graph:

![ML_4](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/ac485ab7-903a-4799-a9b3-c08f54e670f2)


### X_Train_grad value:

![ML_5](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/192dc0ce-7a8b-4f3a-a45f-9e9fa32ad271)


### Y_Train_grad value:

![ML_6](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/b801d5bf-1140-4390-aa30-5108728181a2)


### Print res.X:

![ML_7](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/010e986b-eaf3-4b21-a368-4407e9f0f40d)


### Decision boundary-gragh for exam score:

![ML_8](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/7b28f3fe-1d35-4479-aa22-0ef52431693b)


### Probability value:

![ML_9](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/1b5d8826-ac8c-4e8d-a96a-de754f9ceebc)


### Prediction value of mean:

![ML_10](https://github.com/Skanthasishanth/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118298456/574abcfd-c247-4fd1-b874-9be5a9f60c30)


## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
