import numpy as np


def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter=',')
    # 提取特征
    data_feature = data[:, 1:]
    rows, cols = data_feature.shape
    # 归一化处理(可选)
    data_feature = data_feature / 255
    # 提取标签,data[:,0]返回的是向量(一维矩阵)，data[:,0:1]返回的是二维矩阵,这里用后者是为了格式统一
    data_label = data[:, 0:1]
    # 因为是二分类任务，因此将≥5的作为一类，＜5的作为一类
    data_label[data_label < 5] = -1
    data_label[data_label >= 5] = 1
    return data_feature, data_label


def perceptron(data_feature, data_label, iters=30):
    rows, cols = data_feature.shape
    # 使用向量化编更加简单，给特征向量加入1，变为[1,x1,x2,x3...]
    data_feature = np.concatenate((np.ones((rows, 1)), data_feature), axis=1)
    # 初始化
    w = np.zeros((cols, 1))
    bias = np.zeros((1, 1))
    # 将w与bias组合为一个向量
    w = np.concatenate((bias, w), axis=0)
    for i in range(iters):
        for j in range(rows):
            # 找出分类错误的样本(y(i)*x(i)*w<=0),更新w
            # data_label[j,0]*np.dot(data_feature[j:j+1,:]返回的是一个1×1的矩阵，矩阵不能直接和数字比较大小
            if (data_label[j, 0] *
                    np.dot(data_feature[j:j + 1, :], w))[0, 0] <= 0:
                # 更新公式:w=w+alpha*y(i)*x(i)
                w = w + 0.0001 * data_label[j, 0] * data_feature[j:j + 1, :].T
    return w


def predict(data_feature, data_label, w):
    rows, cols = data_feature.shape
    data_feature = np.concatenate((np.ones((rows, 1)), data_feature), axis=1)
    cnt = 0
    pre = np.dot(data_feature, w) * data_label
    for i in range(pre.shape[0]):
        # 找出分类正确的样本:y(i)*x(i)*w>0
        if pre[i, 0] > 0:
            cnt = cnt + 1
    # 返回正确率
    return cnt / pre.shape[0]


if __name__ == "__main__":
    train_feature, train_label = loadDataSet(
        r'D:\PY实战\ML\dataSet\mnist_train.txt')
    test_feature, test_label = loadDataSet(
        r'D:\PY实战\ML\dataSet\mnist_test.txt')
    w = perceptron(train_feature, train_label)
    accu = predict(test_feature, test_label, w)
    print(accu)