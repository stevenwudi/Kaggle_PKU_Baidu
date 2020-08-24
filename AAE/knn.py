import numpy as np
from numpy.linalg import norm
from collections import Counter


class KNNClassifier:

    def __init__(self, k):
        """初始化kNN分类器"""
        assert k >= 1, "k must be valid"
        self.k = k
        # 私有成员变量_
        self._X_train = None
        self._y_train = None

    def fit(self, X_train, y_train):
        """根据训练数据集X_train和y_train训练kNN分类器"""
        assert X_train.shape[0] == y_train.shape[0], \
            "the size of X_train must be equal to the size of y_train."
        assert self.k <= X_train.shape[0], \
            "the size of X_train must be at least k."

        self._X_train = X_train
        self._y_train = y_train
        return self

    def predict(self, X_predict):
        """给定待测预测数据集X_predict,返回表示X_predict的结果向量集"""
        assert self._X_train is not None and self._y_train is not None, \
            "must fit before predict!"
        assert X_predict.shape[1] == self._X_train.shape[1], \
            "the feature number of X_predict must be similar to X_train."

        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        """给定单个待测数据x,返回x的预测结果集"""
        assert x.shape[0] == self._X_train.shape[1], \
            "the feature number of x must be equal to X_train"

        # 欧拉距离
        # distances = [sqrt(np.sum((x_train-x)**2))
        #            for x_train in self._X_train]
        # 余弦距离

        distances = np.matmul(x[np.newaxis, :], self._X_train.transpose(1, 0)) / (norm(x[np.newaxis, :]) * norm(self._X_train, axis=1))
        distances = distances.squeeze()
        # 求出距离最小索引
        nearest = np.argsort(distances)[-self.k:]

        # 前k个距离最小的标签的点集
        topK_y = [self._y_train[i][0] for i in nearest[:self.k]]

        # 投票统计
        votes = Counter(topK_y)

        # 返回票数最多的标签
        return votes.most_common(1)[0][0]

    def _accuracy_score(self, y_true, y_predict):
        """计算 y_true 和 y_predict 之间的准确率"""

        assert y_true.shape[0] == y_predict.shape[0], \
            "the size of y_true must be equal to the size of y_predict"

        return sum(y_true == y_predict) / len(y_true)

    def score(self, X_test, y_test):
        """根据测算数据集 X_test 和 y_test 确定当前模型的准确度"""
        y_predict = self.predict(X_test)
        return self._accuracy_score(y_test, y_predict)

    def normalize(self, matrix):
        # max min norm -> (0,1), 按行归一化
        max_value = matrix.max(axis=1)
        min_value = matrix.min(axis=1)
        data_rows = matrix.shape[0]
        data_cols = matrix.shape[1]
        norm_matrix = np.empty((data_rows, data_cols))
        for r in range(data_rows):
            norm_matrix[r, :] = (matrix[r, :] - min_value[r]) / (max_value[r] - min_value[r])
        return norm_matrix

    def __repr__(self):
        return "KNN(k=%d)" % self.k
