#!/usr/bin/env python
# coding=utf-8


__author__ = 'Haizheng'
__date__ = ''

"""
特征选择的三种方法介绍：

    过滤型：
    选择与目标变量相关性较强的特征。缺点：忽略了特征之间的关联性。

    包裹型：
    基于线性模型相关系数以及模型结果AUC逐步剔除特征。如果剔除相关系数绝对值较小特征后，AUC无大的变化，或降低，则可剔除

    嵌入型：
    利用模型提取特征，一般基于线性模型与正则化（正则化取L1）,取权重非0的特征。（特征纬度特别高，特别稀疏，用svd,pca算不动）

https://blog.csdn.net/key_v/article/details/48008725
"""

"""1.过滤型"""
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
print(X_new.shape)

"""输出：
        (150L, 4L)
        (150L, 2L)"""

"""2.包裹型"""
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
X = boston["data"]
Y = boston["target"]
names = boston["feature_names"]

lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=1)  # 选择剔除1个
rfe.fit(X, Y)

print("features sorted by their rank:", rfe.ranking_)
print(sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))

"""输出：按剔除后AUC排名给出
features sorted by their rank:
[(1.0, 'NOX'), (2.0, 'RM'), (3.0, 'CHAS'), (4.0, 'PTRATIO'), (5.0, 'DIS'), (6.0, 'LSTAT'), (7.0, 'RAD'), (8.0, 'CRIM'), (9.0, 'INDUS'), (10.0, 'ZN'), (11.0, 'TAX')
, (12.0, 'B'), (13.0, 'AGE')]"""

"""3.嵌入型 ，老的版本没有SelectFromModel"""
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)

lsvc = LinearSVC(C=0.01, penalty='l1', dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)
print(X_new)

"""输出：
            (150,4)
            (150,3)
            """


"""
    先用 StandardScaler 对数据集每一列做标准化处理，（是 transformer）
    再用 PCA 将原始的 30 维度特征压缩的 2 维度，（是 transformer）
    最后再用模型 LogisticRegression。（是 Estimator）
"""
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.pipeline import Pipeline


# 列表，每个元组第一个值为变量名，元组第二个元素是 sklearn 中的 transformer 或 Estimator。
# 中间每一步是 transformer，即它们必须包含 fit 和 transform 方法，或者 fit_transform。
# 最后一步是一个 Estimator，即最后一步模型要有 fit 方法，可以没有 transform 方法。
pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))
                    ])

pipe_lr.fit(X, y)
print('Test accuracy: %.3f' % pipe_lr.score(X, y))


anova_filter = SelectKBest(f_regression, k=5)
clf = svm.SVC(kernel='linear')
anova_svm = Pipeline([('anova', anova_filter), ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
prediction = anova_svm.predict(X)
anova_svm.score(X, y)


