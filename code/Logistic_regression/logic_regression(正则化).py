import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn.preprocessing import PolynomialFeatures


def loadDataSet(fileName):
    dataSet = np.loadtxt(fileName, delimiter=',')
    one = np.ones((dataSet.shape[0], 1))
    dataSet = np.concatenate((one, dataSet), axis=1)
    X = dataSet[:, 0:3]
    Y = dataSet[:, 3].reshape(dataSet.shape[0], 1)
    return X, Y

# 构造多项式特征，如：X1*X2,X1^2等


def mapFeature(X1, X2):
    degree = 6
    out = np.ones((X1.shape[0], 1))
    for i in range(1, degree + 1, 1):
        for j in range(0, i + 1, 1):
            temp = X1**(i - j) * X2**(j)
            out = np.hstack((out, temp))
    return out


def sigmoid(X, theta):
    theta = np.reshape(theta, (X.shape[1], 1))
    return 1 / (1 + np.exp(-X.dot(theta)))


def costFunction(theta, X, Y):
    h = sigmoid(X, theta)
    loss = np.mean(-Y * np.log(h) + -(1 - Y) * np.log(1 - h)) + \
        0.5 * reg * theta.T.dot(theta) / X.shape[0]
    return loss


def gradient(theta, X, Y):
    grad = X.T.dot((sigmoid(X, theta) - Y)) / X.shape[0]
    theta[0] = 0
    theta = np.reshape(theta, (theta.shape[0], 1))
    grad = grad + reg * (1 / X.shape[0]) * theta
    return grad.ravel()


dataSet, Y = loadDataSet('ex2data2.txt')
X = mapFeature(dataSet[:, 1:2], dataSet[:, 2:3])
theta = np.zeros((X.shape[1], 1))
# print(costFunction(theta,X,Y))
# 定义正则化项的lambda值
reg = 1
# 最优化
theta = opt.fmin_cg(
    costFunction,
    x0=theta,
    args=(
        X,
        Y),
    maxiter=500,
    fprime=gradient)
print('theta:\n', theta)

# 绘图
flag = np.where(Y.ravel() == 0)
plt.scatter(dataSet[flag, 1], dataSet[flag, 2], c='r', marker='x', label=0)
flag = np.where(Y.ravel() == 1)
plt.scatter(dataSet[flag, 1], dataSet[flag, 2], c='b', marker='o', label=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(loc='upper left')
# plot the boundary
poly = PolynomialFeatures(6)
x1Min = dataSet[:, 1].min()
x1Max = dataSet[:, 1].max()
x2Min = dataSet[:, 2].min()
x2Max = dataSet[:, 2].max()
xx1, xx2 = np.meshgrid(np.linspace(x1Min, x1Max), np.linspace(x2Min, x2Max))
h1 = poly.fit_transform(np.c_[xx1.ravel(), xx2.ravel()]).dot(theta)
h1 = h1.reshape(xx1.shape)
plt.contour(xx1, xx2, h1, [0.5], colors='b', linewidth=.5)
plt.show()
