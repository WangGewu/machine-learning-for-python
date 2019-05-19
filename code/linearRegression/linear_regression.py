import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(filename):
    # delimiter表示分隔符
    # usecols表示选取的列
    dataSet = np.loadtxt(filename, delimiter=',')
    rows, cols = dataSet.shape
    # 为了将b和theta放在一个矩阵中，给X增加一维特征，值全为，将theta*x+b变为theta*x
    temp = np.ones((rows, 1))
    dataSet = np.concatenate((temp, dataSet), axis=1)
    Y = dataSet[:, cols]
    Y = np.reshape(Y, (rows, 1))
    X = dataSet[:, 0:cols]
    return dataSet, X, Y


def costFunction(X, Y, theta):
    nums = X.shape[0]
    # 计算损失函数
    cost = 0.5 * np.sum(np.power(X.dot(theta) - Y, 2)) / (nums)
    return cost


def gradientDescend(X, Y, theta, iters=1500, alpha=0.01):
    nums = X.shape[0]
    # 存储每次迭代后损失函数的值
    historyCost = []
    for i in range(iters):
        # 更新theta
        theta = theta - alpha * X.T.dot(np.dot(X, theta) - Y) / nums
        historyCost.append(costFunction(X, Y, theta))
    return theta, historyCost


if __name__ == "__main__":
    # X=n*2,Y=n*1
    dataSet, X, Y = loadDataSet('ex1data1.txt')
    # 初始化theta
    theta = np.array([[0.001], [0.001]])
    # 梯度下降迭代求解theta
    theta, historyCost = gradientDescend(X, Y, theta)
    print('theta:\n', theta)
    # 可视化
    plt.figure()
    xl = np.arange(5, 25)
    yl = theta[0, 0] + theta[1, 0] * xl
    x2 = dataSet[:, 1]
    y2 = dataSet[:, 2]
    plt.subplot(211)
    plt.plot(xl, yl)
    plt.scatter(x2, y2, c='red', s=10)
    plt.subplot(212)
    plt.plot(historyCost)
    plt.xlabel('iters')
    plt.ylabel('Cost')
    plt.show()
