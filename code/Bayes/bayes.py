import numpy as np


def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter=',')
    sample = data[:, 1:]
    sample[sample <= 128] = 0
    sample[sample > 128] = 1
    label = data[:, 0]
    return sample, label


def getAllProb(train_sample, train_label):
    # 类别数
    class_num = 10
    # 训练样本数
    data_num = train_label.shape[0]
    # 特征数
    feature_num = train_sample.shape[1]
    # 累计每个类别出现的次数
    prob_class_num = np.zeros((class_num, 1))
    for i in range(data_num):
        prob_class_num[int(train_label[i])] += 1
    # 拉普拉斯平滑
    prob_class_laplacian = (prob_class_num + 1) / (data_num + class_num)
    # 开始计算条件概率
    prob_conditional_laplacian = np.zeros((class_num, feature_num, 2))
    # 累计每个类别中每个特征项对应特征值出现次数
    for i in range(data_num):
        for j in range(feature_num):
            prob_conditional_laplacian[int(
                train_label[i]), j, int(train_sample[i, j])] += 1
    # 拉普拉斯平滑
    for i in range(class_num):
        prob_conditional_laplacian[i, :, :] = (
            prob_conditional_laplacian[i, :, :] + 1) / (prob_class_num[i, 0] + 2)
    # 取log防止连乘后下溢
    return np.log(prob_class_laplacian), np.log(prob_conditional_laplacian)


def bayesTest(test_sample, test_label, prob_class, prob_conditional):
    print("Testing")
    test_num, feature_num = test_sample.shape
    # 类别数
    class_num = 10
    # 存储测试样本属于每个类别的概率
    pred_prob = np.zeros((test_num, class_num))
    for i in range(test_num):
        for j in range(class_num):
            pred_prob[i, j] = prob_class[j]
            for k in range(feature_num):
                # 取log后连乘变成了连加
                pred_prob[i, j] += prob_conditional[j,
                                                    k, int(test_sample[i, k])]
    pred_label = np.argmax(pred_prob, axis=1)
    # 计算准确率
    cnt = 0
    for i in range(test_num):
        if test_label[i] == pred_label[i]:
            cnt += 1
    accu = cnt / test_num
    return accu


if __name__ == "__main__":
    train_sample, train_label = loadDataSet(
        r'D:\py实战\ML\dataSet\mnist_train.txt')
    print(train_sample.shape)
    test_sample, test_label = loadDataSet(r'D:\py实战\ML\dataSet\mnist_test.txt')
    p_c, p_cond = getAllProb(train_sample, train_label)
    accu = bayesTest(test_sample, test_label, p_c, p_cond)
    print(accu)
