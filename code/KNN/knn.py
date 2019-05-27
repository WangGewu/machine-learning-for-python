"""
数据集:mnist(mnist数据集是手写体数字的数据集，每幅图的大小是28×28，展开成向量就是1×784)
训练集:60000×784，第一列是标签，其后是特征
测试集:200×784个样本，第一列是标签，其后是特征，测试集全选的话跑的时间太长
"""
import numpy as np
from collections import Counter


def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter=',')
    data_feature = data[:, 1:]
    data_label = data[:, 0:1]
    return data_feature, data_label
# 找到列表中出现最多的元素


def findMostCommon(data_list):
    return (Counter(data_list).most_common(1))[0][0]


def knn(train_data, train_label, test_data, test_label, n):
    # 200个测试数据
    test_rows = 200
    train_rows = train_data.shape[0]
    # 存储预测标签
    predict_label = np.ones((test_rows, n))
    cnt = 0
    for i in range(test_rows):
        # 计算距离
        dist = np.sqrt(
            np.sum(np.power(train_data - test_data[i:i + 1, :], 2), axis=1))
        dist = dist.reshape((train_rows, 1))
        # 给距离矩阵排序
        sort_dist = np.argsort(dist, axis=0)
        sort_dist = sort_dist.reshape((train_rows, 1))
        # 将与测试数据距离最短的n个数据所对应的标签存为列表
        label = [train_label[sort_dist[j, 0], 0] for j in range(n)]
        # 找出出现最多的类别作为分类结果
        predict_label[i, 0] = findMostCommon(label)
    # 计算错误率
    for i in range(test_rows):
        if predict_label[i, 0] == test_label[i, 0]:
            cnt += 1
    accu = cnt / test_rows
    return predict_label, accu


if __name__ == "__main__":
    dataFeature, dataLabel = loadDataSet(r'D:\PY实战\ML\dataSet\mnist_train.txt')
    testFeature, testLabel = loadDataSet(r'D:\PY实战\ML\dataSet\mnist_test.txt')
    predict_label, accu = knn(dataFeature, dataLabel,
                              testFeature, testLabel, 25)
    print(accu)
