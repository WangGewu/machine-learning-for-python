"""
数据量多，跑的时间很久
"""
import numpy as np


def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter=',')
    sample = data[:, 1:]
    sample[sample <= 128] = 0
    sample[sample > 128] = 1
    label = data[:, 0]
    return sample, label

# 返回label中出现最多的类别


def findMaxCategory(label):
    dict = {}
    for i in range(label.shape[0]):
        if label[i] in dict.keys():
            dict[label[i]] += 1
        else:
            dict[label[i]] = 1
    max_category = -1
    for key in dict.keys():
        if dict[key] > max_category:
            max_category = dict[key]
            max_label = key
    return max_label

# 计算信息熵


def calcEnt(label):
    category_num = len(set(label))
    data_num = label.shape[0]
    dict = {}
    for i in range(label.shape[0]):
        if label[i] in dict.keys():
            dict[label[i]] += 1
        else:
            dict[label[i]] = 1
    Ent = 0
    for key in dict.keys():
        p = dict[key] / data_num
        Ent += p * np.log2(p)
    return -Ent

# 根据特征取值不同分割样本


def divSample(sample, label, feat_index, feat_value):
    index = []
    for i in range(sample.shape[0]):
        if sample[i, feat_index] == feat_value:
            index.append(i)
    return sample[index, :], label[index]

# 寻找最佳特征


def findBestFeature(sample, label, feature):
    max_gain = 0
    for i in feature:
        # 所有可能的取值
        all_value = set(sample[:, i])
        gain = calcEnt(label)
        for j in all_value:
            div_sample, div_label = divSample(sample, label, i, j)
            gain -= (div_label.shape[0] / label.shape[0]) * calcEnt(div_label)
        if gain > max_gain:
            best_feat = i
            max_gain = gain
    return best_feat, max_gain


def DT(sample, label, feature):
    # 如果sample中都属于同一类，贼返回树
    if len(set(label)) == 1:
        return label[0]
    # 如果feature为空，返回label中类别出现次数最多的作为树的类标
    if len(feature) == 0:
        return findMaxCategory(label)
    best_feat, max_gain = findBestFeature(sample, label, feature)
    threshold = 0.1
    if max_gain < threshold:
        return findMaxCategory(label)
    deci_tree = {best_feat: {}}
    all_value = set(sample[:, best_feat])
    for i in all_value:
        div_sample, div_label = divSample(sample, label, best_feat, i)
        div_feature = feature.copy()
        div_feature.remove(best_feat)
        deci_tree[best_feat][i] = DT(div_sample, div_label, div_feature)
    return deci_tree


if __name__ == "__main__":
    train_sample, train_label = loadDataSet(
        r'D:\py实战\ML\dataSet\mnist_train.txt')
    test_sample, test_label = loadDataSet(r'D:\py实战\ML\dataSet\mnist_test.txt')
    feature = []
    for i in range(train_sample.shape[1]):
        feature.append(i)
    tree = DT(train_sample, train_label, feature)
    print(tree)
