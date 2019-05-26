import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def loadDataSet(fileName):
    dataSet = np.loadtxt(fileName, delimiter=',')
    rows, cols = dataSet.shape
    one = np.ones((rows, 1))
    dataSet = np.concatenate((one, dataSet), axis=1)
    X = dataSet[:, 0:3]
    Y = dataSet[:, 3]
    Y = np.reshape(Y, (rows, 1))
    return X, Y


def sigmoid(X, theta):
    # 传进去的theta的shape为(3,)，在计算时要改为(3,1)
    theta = np.reshape(theta, (X.shape[1], 1))
    return (1 / (1 + np.exp(-np.dot(X, theta))))


def costFunction(theta, X, Y):
    h = sigmoid(X, theta)
    loss = np.mean(-Y * np.log(h) + -(1 - Y) * np.log(1 - h))
    return loss


def gradient(theta, X, Y):
    grad = np.zeros((X.shape[1], 1))
    grad = X.T.dot((sigmoid(X, theta) - Y)) / X.shape[0]
    return grad.ravel()


if __name__ == "__main":
    X, Y = loadDataSet('ex2data1.txt')
    theta = np.zeros((X.shape[1], 1))
    # 传进去的theta的shape为(3,)，在计算时要改为(3,1)
    theta = opt.fmin_cg(
        costFunction, x0=theta, args=(
            X, Y), maxiter=500, fprime=gradient)
    print(theta)
    # 可视化
    plt.figure()
    flag = np.where(Y.ravel() == 0)
    plt.scatter(X[flag, 1], X[flag, 2], c='r')
    flag = np.where(Y.ravel() == 1)
    plt.scatter(X[flag, 1], X[flag, 2], c='g')
    xl = np.arange(20, 100)
    yl = (-theta[0] - theta[1] * xl) / theta[2]
    plt.plot(xl, yl)
    plt.show()
