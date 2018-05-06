#!/usr/bin/env python
# coding=utf-8


__author__ = 'Haizheng'
__date__ = ''

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split  # 训练集数据拆分
from sklearn.model_selection import GridSearchCV  # 参数调优
from sklearn.model_selection import StratifiedKFold


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics  # 模型评估
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib  # 模型输出

import xgboost as xgb
from xgboost import plot_importance  # 参数相关性

# https://blog.csdn.net/qq_31813549/article/details/79964973
from imblearn.over_sampling import SMOTE  # 样本不平衡

filename = "/home/johnhill_cn/JData/data/JData_Comment.csv"
datasize = 100000


def get_chunk(filename, datasize):
    # df = pd.read_csv(filename, sep=",", )
    return pd.read_csv(filename, sep=",", chunksize=datasize)


def loaddata():
    chunk = [data for data in get_chunk(filename, datasize)]
    df = pd.concat(chunk, ignore_index=True)
    return df


def train_data(data):
    train_y = data['has_bad_comment']

    train_x = data.drop(['has_bad_comment', 'dt'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=5)

    sm = SMOTE(kind='borderline1', random_state=42)
    x_train, y_train = sm.fit_sample(x_train, y_train)

    # from imblearn.over_sampling import SMOTE, ADASYN
    # X_resampled, y_resampled = SMOTE(kind='borderline1').fit_sample(X, y)

    model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=5000,
                              max_depth=5,
                              min_child_weight=3,
                              gamma=0.3,
                              subsample=0.85,
                              colsample_bytree=0.75,
                              objective='binary:logistic',
                              scale_pos_weight=1,
                              seed=27,
                              nthread=12,
                              reg_alpha=0.0005)
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    train_report = metrics.classification_report(y_train, y_train_pred)
    test_report = metrics.classification_report(y_test, y_test_pred)
    print(train_report)
    print(test_report)

def ceate_feature_map(features):
    outfile = open('xgb.fmap', 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def xgb_model():
    from sklearn import datasets
    from xgboost import plot_tree
    import matplotlib.pyplot as plt

    from collections import Counter

    iris = datasets.load_iris()
    data = iris.data[:100]
    label = iris.target[:100]

    train_x, test_x, train_y, test_y = train_test_split(data, label, random_state=0, test_size=.3)

    print(Counter(train_y))
    print(Counter(test_y))

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)

    # ceate_feature_map(dtrain.columns)

    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'eval_metric': 'auc',  # ROC曲线下的面积 真阳性率与假阳性率之间的关系
              'max_depth': 4,
              'lambda': 10,
              'subsample': 0.75,
              'colsample_bytree': 0.75,
              'min_child_weight': 2,
              'eta': 0.025,
              'seed': 0,
              'nthread': 8,
              'silent': 1,
              'verbose': True}

    watchlist = [(dtrain, 'train')]

    # 训练模型
    bst = xgb.train(params, dtrain, num_boost_round=50, evals=watchlist)

    # 预测
    ypred = bst.predict(dtest)

    # 设置阈值, 输出一些评价指标
    y_pred = (ypred > 0.5) * 1
    # predictions = [round(value) for value in y_pred]


    print('AUC: %.4f' % metrics.roc_auc_score(test_y,ypred))
    print('ACC: %.4f' % metrics.accuracy_score(test_y,y_pred))
    print('Recall: %.4f' % metrics.recall_score(test_y,y_pred))
    print('F1-score: %.4f' %metrics.f1_score(test_y,y_pred))
    print('Precesion: %.4f' %metrics.precision_score(test_y,y_pred))
    print(metrics.confusion_matrix(test_y,y_pred))

    ypred_leaf = bst.predict(dtest, pred_leaf=True)
    print(ypred_leaf)

    xgb.to_graphviz(bst, num_trees=0)
    # plot_tree(bst, fmap='xgb.fmap')
    # plt.show()

    plot_importance(bst)
    plt.show()

def opti_model(X, Y):
    model = xgb.XGBClassifier()
    learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
    param_grid = dict(learning_rate=learning_rate)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
    grid_result = grid_search.fit(X, Y)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    return model


if __name__ == '__main__':
    # df = loaddata()
    # print(df.has_bad_comment.value_counts())

    xgb_model()
