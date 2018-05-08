#!/usr/bin/env python
#coding=utf-8


__author__ = 'Haizheng'
__date__ = ''


import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn import linear_model, metrics
from sklearn.svm import SVC
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier,
                              BaggingClassifier, BaggingRegressor, RandomForestClassifier)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

params = {
    'learning_rate': 0.01,
    'n_estimators': 200,
    'max_depth': 5,
    'min_child_weight': 3,
    'gamma': 0,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'binary:logistic',
    # 在各类别样本十分不平衡时，把这个参数设定为正值，可以是算法更快收敛
    'scale_pos_weight': 1,
    'nthread': 4
}

def preprocess():
    train = pd.read_csv("/home/johnhill_cn/20170612/dataset/broadband_train.csv", encoding='gbk')

    train.fillna(train.mean(), inplace=True)
    train.dropna()
    train = pd.get_dummies(train)

    test = pd.read_csv("/home/johnhill_cn/20170612/dataset/broadband_test.csv", )
    test.fillna(test.mean(), inplace=True)


    test.rename(columns={'ARPU_3M': 'ARPB_3M'}, inplace=True)
    test = pd.get_dummies(test)


    return train, test

def lr_base(X, y):
    clf = linear_model.LogisticRegression(C=1.0, max_iter=100, class_weight='balanced', random_state=2017)
    clf.fit(X, y)
    predictions = clf.predict(X)

    print("1个lr分类器accuracy %.7g" % metrics.accuracy_score(y, predictions))
    print("1个lr分类器precision %.7g" % metrics.precision_score(y, predictions))
    print("1个lr分类器f1 %.7g" % metrics.f1_score(y, predictions))
    return clf

def ensemble1(X, y):
    clf1 = linear_model.LogisticRegression(penalty='l1', C=0.1, max_iter=200, class_weight='balanced', random_state=2017, )
    clf2 = RandomForestClassifier(random_state=20, n_estimators=500, max_features=6, max_depth=4,)
    clf3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, min_samples_split=4, min_samples_leaf=1,
                                      max_depth=3,
                                      max_features=None, subsample=1, random_state=2017)
    clf4 = XGBClassifier(**params)

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gbdt', clf3),
                                        ('xgb', clf4)
                                        ],
                            voting='soft', weights=[1, 1., 2, 2]
                            )
    for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['lr', 'Random Forest', 'gbdt', 'xgb', 'Ensemble']):
        scores = cross_val_score(clf, X, y, cv=5, scoring='f1')  # f1  accuracy roc_auc
        print("Accuracy: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), label))
    # print clf2
    return eclf

def xgb(X, y,):
    clf = XGBClassifier(**params)

    # modelfit(clf, X, y)
    clf.fit(X, y)
    ypred2 = clf.predict(X)

    print("分类器f1_score %.7g" % metrics.f1_score(y, ypred2))

    return clf

if __name__ == '__main__':
    train, test = preprocess()

    # print train.info()
    # scaler = Normalizer()
    X = train[[x for x in train.columns if x not in ["BROADBAND", "CUST_ID"]]]
    y = train["BROADBAND"]
    x_test = test[[x for x in test.columns if x not in ["CUST_ID"]]]

    # clf = lr_base(X, y)
    clf = ensemble1(X, y)
    # clf = xgb(X, y)
    clf.fit(X, y)

    predictions = clf.predict(x_test)
    res = pd.DataFrame(
        {'CUST_ID': test["CUST_ID"].as_matrix(), 'flag': predictions.astype(np.int32)})
    # res.to_csv("out1122.txt", sep=",", columns=["CUST_ID", "flag"], index=False, header=False)
    """
    Accuracy: 0.53212 (+/- 0.06974) [lr]
    Accuracy: 0.55029 (+/- 0.05256) [Random Forest]
    Accuracy: 0.52967 (+/- 0.07896) [gbdt]
    Accuracy: 0.37419 (+/- 0.11342) [knn]
    Accuracy: 0.56565 (+/- 0.04678) [Ensemble]
    
    Accuracy: 0.56579 (+/- 0.05448) [Ensemble]
    Accuracy: 0.57518 (+/- 0.04955) [Ensemble]
    Accuracy: 0.57636 (+/- 0.06106) [Ensemble]
    Accuracy: 0.58515 (+/- 0.07485) [Ensemble]
    
     F1 = 771929.8245614 得分 : 46.315789473684]
     F1 = 771929.8245614 得分 : 46.315789473684]
     F1 = 789473.68421053 得分 : 47.3684210526318]
     F1 = 798245.61403509 得分 : 47.8947368421054]

    """







