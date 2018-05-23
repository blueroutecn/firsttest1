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
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel,f_regression

iris = load_iris()
X, y = iris.data, iris.target
print(X.shape)
X_new = SelectKBest(chi2, k=3).fit_transform(X, y)
print(X_new.shape)

"""输出：
        (150L, 4L)
        (150L, 2L)"""

"""2.包裹型"""

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
anova_svm = Pipeline([('anova', anova_filter),
                      ('svc', clf)])
anova_svm.set_params(anova__k=10, svc__C=.1).fit(X, y)
prediction = anova_svm.predict(X)
anova_svm.score(X, y)


#
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 某些机器学习算法和模型只能接受定量特征的输入，那么需要将定性特征转换为定量特征。
from sklearn.preprocessing import OneHotEncoder

#
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel,f_regression

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
clf1 = LinearSVC(penalty="l2",C=.1)
clf = LogisticRegression(penalty='l2', C=.1,)



"""
管道的流水线
    1.数据预处理
    2.降维
    3.训练
    4.预测
    
在规则化参数的同时最小化误差
    规则化参数:防止我们的模型过分拟合我们的训练数据。
    最小化误差:模型拟合数据

问题背景：参数太多，会导致我们的模型复杂度上升，容易过拟合。
作用：
1、约束参数，降低模型复杂度。
2、规则项的使用还可以约束我们的模型的特性。这样就可以将人对这个模型的先验知识融入到模型的学习当中，
    强行地让学习到的模型具有人想要的特性，例如稀疏、低秩、平滑等等。

目标函数 =   误差平方和 + 规则化函数
第一项:Loss函数  L(yi,f(xi;w)) 就是误差平方和

如果是Square loss，那就是最小二乘了；
如果是Hinge Loss，那就是著名的SVM了；
如果是exp-Loss，那就是牛逼的 Boosting了；
如果是log-Loss，那就是Logistic Regression了

第二项-规则化函数Ω(w)
    L0范数是指向量中非0的元素的个数。如果我们用L0范数来规则化一个参数矩阵W的话，就是希望W的大部分元素都是0。都为稀疏。
    L1范数是指向量中各个元素绝对值之和，也有个美称叫“稀疏规则算子”（Lasso regularization）。
    
    参数稀疏的好处
        实现特征的自动选择
        可解释性
    
    L2范数是指向量各元素的平方和然后求平方根。 
    L2的作用=参数变小=模型变简单≈模型参数信息变少。
    L2范数不但可以防止过拟合，还可以让我们的优化求解变得稳定和快速。
    优化计算的角度。L2范数有助于处理 condition number不好的情况下矩阵求逆很困难的问题。
    
     L1在江湖上人称Lasso(直线)，L2人称Ridge(曲线)
     L1会趋向于产生少量的特征，而其他的特征都是0，
     而L2会选择更多的特征，这些特征都会接近于0。Lasso在特征选择时候非常有用，而Ridge就只是一种规则化而已。v
     
     L2比L1要好一些，因为L2之后，精度更好且较好适应、拟合。L1的效果在处理稀疏数据时候比较棒，且有利于稀疏数据的特征。
     L1+L2=Elastic Nets的办法，既可以处理稀疏问题，同时也可以保证精度。
     
     对于SVM和逻辑回归，参数C控制稀疏性：C越小，被选中的特征越少。对于Lasso，参数alpha越大，被选中的特征越少。
     SelectFromModel(LR(threshold=0.5, C=0.1)).fit_transform(iris.data, iris.target)
     
    
"""
pipe_lr = Pipeline([('sc', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))
                    ])

pipe_lr.fit(X, y)
print('Test accuracy: %.3f' % pipe_lr.score(X, y))

from numpy import log1p
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline,FeatureUnion

# 新建计算缺失值的对象
step1 = ('Imputer', Imputer())
# 新建将部分特征矩阵进行定性特征编码的对象
step2_1 = ('OneHotEncoder', OneHotEncoder(sparse=False))
# 新建将部分特征矩阵进行对数函数转换的对象
step2_2 = ('ToLog', FunctionTransformer(log1p))
# 新建将部分特征矩阵进行二值化类的对象
step2_3 = ('ToBinary', Binarizer())
# 新建部分并行处理对象，返回值为每个并行工作的输出的合并
step2 = (
'FeatureUnionExt', FeatureUnionExt(transformer_list=[step2_1, step2_2, step2_3], idx_list=[[0], [1, 2, 3], [4]]))
# 新建无量纲化对象
step3 = ('MinMaxScaler', MinMaxScaler())
# 新建卡方校验选择特征的对象
step4 = ('SelectKBest', SelectKBest(chi2, k=3))
# 新建PCA降维的对象
step5 = ('PCA', PCA(n_components=2))
# 新建逻辑回归的对象，其为待训练的模型作为流水线的最后一步
step6 = ('LogisticRegression', LogisticRegression(penalty='l2'))
# 新建流水线处理对象
# 参数steps为需要流水线处理的对象列表，该列表为二元组列表，第一元为对象的名称，第二元为对象
pipeline = Pipeline(steps=[step1, step2, step3, step4, step5, step6])
pipeline.fit(X, y)