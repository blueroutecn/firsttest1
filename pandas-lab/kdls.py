#!/usr/bin/env python
# coding=utf-8

import time
import numpy as np
import pandas as pd
import xgboost as xgb
# import lightgbm as lgb
from xgboost import plot_importance  # 参数相关性
from xgboost.sklearn import XGBClassifier

from sklearn.model_selection import train_test_split  # 训练集数据拆分
from sklearn.model_selection import GridSearchCV  # 参数调优
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics, linear_model  # 模型评估

from sklearn import preprocessing
# X_scaled = preprocessing.scale(X)

# scaler = preprocessing.StandardScaler().fit(X)
# sscaler.transform(X)

# min_max_scaler = preprocessing.MinMaxScaler()
# X_train_minmax = min_max_scaler.fit_transform(X_train)


from sklearn.ensemble import (GradientBoostingClassifier, VotingClassifier,
                              BaggingClassifier, BaggingRegressor, RandomForestClassifier)

from imblearn.over_sampling import SMOTE
from sklearn.externals import joblib


# import matplotlib.pyplot as plt

def loaddata(filename):
    df = pd.read_csv(filename, sep="\t")

    names = ['Prd_Inst_Id', 'Gender_Id', 'Age',
             'Country_Flag', 'telecom_area_id', 'Innet_Dur', 'Line_Rate',
             'Ccust_Brd_Cnt', 'Ccust_Fix_Cnt', 'Ccust_CDMA_Cnt',
             'Ofr_Type_Id', 'Brd_Type_Flag', 'Use_Mons',
             'Stop_Cnt',
             'R3A_Net_Days', 'R3A_Wday_Net_Days', 'R3A_Hday_Net_Days', 'R3A_Net_Dur',
             'R3A_Wday_Net_Dur', 'R3A_Hday_Net_Dur', 'R3A_Day_Net_Dur',
             'R3A_Ngt_Net_Dur', 'R3A_Wday_Day_Net_Dur', 'R3A_Wday_Ngt_Net_Dur',
             'R3A_Hday_Day_Net_Dur', 'R3A_Hday_Ngt_Net_Dur', 'R3A_Net_Cnt',
             'R3A_Wday_Net_Cnt', 'R3A_Hday_Net_Cnt', 'R3A_Day_Net_Cnt',
             'R3A_Ngt_Net_Cnt', 'R3A_Wday_Day_Net_Cnt', 'R3A_Wday_Ngt_Net_Cnt',
             'R3A_Hday_Day_Net_Cnt', 'R3A_Hday_Ngt_Net_Cnt', 'R3A_Net_Kbyte',
             'R3A_Wday_Net_Kbyte', 'R3A_Hday_Net_Kbyte', 'R3A_Day_Net_Kbyte',
             'R3A_Ngt_Net_Kbyte', 'R3A_Wday_Day_Net_Kbyte', 'R3A_Wday_Ngt_Net_Kbyte',
             'R3A_Hday_Day_Net_Kbyte', 'R3A_Hday_Ngt_Net_Kbyte',
             'R3A_Day_Net_Dur_Rate', 'R3A_Ngt_Net_Dur_Rate',
             'R3A_Wday_Net_Days_Rate', 'R3A_Hday_Net_Days_Rate',
             'R3A_Avgday_Net_Dur', 'R3A_Avgday_Net_Kbyte', 'R3A_AvgWday_Net_Dur',
             'R3A_AvgHday_Net_Dur', 'R3A_Net_Kbyte_Dur_Speed',
             'Net_Kbyte_Dur_Speed_Trend', 'Net_Days_Trend', 'Net_Dur_Trend',
             'Net_Kbyte_Trend', 'Owe_Amt_Cnt',
             'Owe_Amt',
             'Inv_Amt', 'Acct_Bal_Amt',
             'Net_Use_Zero_Mc', 'Serv_Vnet_Flag', 'IPTV_Flag', 'Csum_Cs_Cpl_Cnt',
             'Csum_Cs_Conslt_Cnt', 'Csum_Cs_Oper_Cnt', 'Csum_Cs_Hnd_Cnt',
             'Csum_Cs_Other_Cnt', 'R3A_Fix_Inv_Amt', 'R3A_Fix_Call_Cnt',
             'R3A_Fix_Call_Dur', 'Fix_Call_Cnt_Trend', 'Fix_Call_Dur_Trend',
             'Fix_Inv_Amt_Trend', 'R3A_CDMA_Inv_Amt', 'R3A_CDMA_Call_Cnt',
             'R3A_CDMA_Call_Dur', 'CDMA_Call_Cnt_Trend', 'CDMA_Call_Dur_Trend',
             'CDMA_Inv_Amt_Trend', 'Csum_Fix_Comp_Cnt', 'Csum_CDMA_Comp_Cnt',
             'Churn_Flag', 'qf_flag',
             'ls_flag1']

    df = df[names]
    df['Line_Rate'] = df['Line_Rate'].map(daikuan_proc)
    df['Ccust_Brd_Cnt'] = df['Ccust_Brd_Cnt'].map(brdcnt_proc)
    df['Ccust_Fix_Cnt'] = df['Ccust_Fix_Cnt'].map(brdcnt_proc)
    df['Ccust_CDMA_Cnt'] = df['Ccust_CDMA_Cnt'].map(cdmacnt_proc)
    df['Age'] = df['Age'].map(age_proc)
    df['ls_flag1'] = df['ls_flag1'].map(flag_proc)
    # df.to_csv("data/test.txt")
    return df


def daikuan_proc(dk):
    if "K" in dk:
        return 5
    if "G" in dk or "未知速率" in dk:
        return 120

    dk = int(dk[:-1])

    big_cust = [1, 2, 4, 8, 10, 20, 50, 100]
    if dk in big_cust:
        return dk
    elif dk < 10:
        return 5
    elif dk < 20:
        return 15
    elif dk < 50:
        return 45
    elif dk < 100:
        return 80
    else:
        return 120


def brdcnt_proc(data, rate=3):
    try:
        cnt = int(data)
        if cnt > rate:
            return 10
        else:
            return cnt
    except:
        return .1


def cdmacnt_proc(data, rate=5):
    try:
        cnt = int(data)
        if cnt > rate:
            return 10
        else:
            return cnt
    except:
        return .1


def flag_proc(data):
    return data if data < 1 else 1


def age_proc(data):
    if data == 0:
        return 0
    if 18 <= data < 30:
        return 1
    elif 30 <= data < 50:
        return 2
    elif 50 <= data < 70:
        return 3
    else:
        return 4


id = "Prd_Inst_Id"
flag = "ls_flag1"


def weakclf(train):
    X = train[[x for x in train.columns if x not in [id, flag]]]
    Y = train[flag]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    sm = SMOTE(random_state=5)
    x_train, y_train = sm.fit_sample(x_train, y_train)

    model = ensemble1(x_train, y_train)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    train_report = metrics.classification_report(y_train, y_train_pred)
    print(train_report)


def train_data(train):
    X = train[[x for x in train.columns if x not in [id, flag]]]
    Y = train[flag]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    sm = SMOTE(random_state=5)
    x_train, y_train = sm.fit_sample(x_train, y_train)

    model = xgb.XGBClassifier(learning_rate=0.01,
                              n_estimators=100,
                              max_depth=5,
                              min_child_weight=3,
                              gamma=0.3,
                              subsample=0.85,
                              colsample_bytree=0.75,
                              objective='binary:logistic',
                              scale_pos_weight=1,
                              seed=5,
                              #                              nthread=12,
                              reg_alpha=0.001)
    model.fit(x_train, y_train)

    y_train_pred = model.predict(x_train)
    #    y_test_pred = model.predict(x_test)
    train_report = metrics.classification_report(y_train, y_train_pred)
    #    test_report = metrics.classification_report(y_test, y_test_pred)
    print(train_report)
    # plot_importance(model,max_num_features=20)
    # plt.show()

    joblib.dump(model, 'sample1.pkl')


#   clf=joblib.load('filename.pkl')
#    model.save_model('sample.model')

#    print(test_report)

def pred_test(train):
    X = train[[x for x in train.columns if x not in [id, flag,
                                                     # "Stop_Cnt", "Owe_Amt","Churn_Flag"
                                                     ]]]
    Y = train[flag]

    d = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16',
         'f17', 'f18', 'f19', 'f20', 'f21', 'f22', 'f23', 'f24', 'f25', 'f26', 'f27', 'f28', 'f29', 'f30', 'f31', 'f32',
         'f33', 'f34', 'f35', 'f36', 'f37', 'f38', 'f39', 'f40', 'f41', 'f42', 'f43', 'f44', 'f45', 'f46', 'f47', 'f48',
         'f49', 'f50', 'f51', 'f52', 'f53', 'f54', 'f55', 'f56', 'f57', 'f58', 'f59', 'f60', 'f61', 'f62', 'f63', 'f64',
         'f65', 'f66', 'f67', 'f68', 'f69', 'f70', 'f71', 'f72', 'f73', 'f74', 'f75', 'f76', 'f77', 'f78', 'f79', ''
                                                                                                                  'f80',
         'f81', 'f82', 'f83'
         ]
    e = [u'Gender_Id',
         u'Age',
         u'Country_Flag',
         u'telecom_area_id',
         u'Innet_Dur',
         u'Line_Rate', u'Ccust_Brd_Cnt',
         u'Ccust_Fix_Cnt', u'Ccust_CDMA_Cnt', u'Ofr_Type_Id', u'Brd_Type_Flag', u'Use_Mons',
         u'Stop_Cnt',
         u'R3A_Net_Days', u'R3A_Wday_Net_Days', u'R3A_Hday_Net_Days', u'R3A_Net_Dur', u'R3A_Wday_Net_Dur',
         u'R3A_Hday_Net_Dur', u'R3A_Day_Net_Dur', u'R3A_Ngt_Net_Dur', u'R3A_Wday_Day_Net_Dur', u'R3A_Wday_Ngt_Net_Dur',
         u'R3A_Hday_Day_Net_Dur', u'R3A_Hday_Ngt_Net_Dur', u'R3A_Net_Cnt', u'R3A_Wday_Net_Cnt', u'R3A_Hday_Net_Cnt',
         u'R3A_Day_Net_Cnt', u'R3A_Ngt_Net_Cnt', u'R3A_Wday_Day_Net_Cnt', u'R3A_Wday_Ngt_Net_Cnt',
         u'R3A_Hday_Day_Net_Cnt', u'R3A_Hday_Ngt_Net_Cnt', u'R3A_Net_Kbyte', u'R3A_Wday_Net_Kbyte',
         u'R3A_Hday_Net_Kbyte', u'R3A_Day_Net_Kbyte', u'R3A_Ngt_Net_Kbyte', u'R3A_Wday_Day_Net_Kbyte',
         u'R3A_Wday_Ngt_Net_Kbyte', u'R3A_Hday_Day_Net_Kbyte', u'R3A_Hday_Ngt_Net_Kbyte', u'R3A_Day_Net_Dur_Rate',
         u'R3A_Ngt_Net_Dur_Rate', u'R3A_Wday_Net_Days_Rate', u'R3A_Hday_Net_Days_Rate', u'R3A_Avgday_Net_Dur',
         u'R3A_Avgday_Net_Kbyte', u'R3A_AvgWday_Net_Dur', u'R3A_AvgHday_Net_Dur', u'R3A_Net_Kbyte_Dur_Speed',
         u'Net_Kbyte_Dur_Speed_Trend', u'Net_Days_Trend', u'Net_Dur_Trend', u'Net_Kbyte_Trend',
         u'Owe_Amt_Cnt',
         u'Owe_Amt',
         u'Inv_Amt', u'Acct_Bal_Amt', u'Net_Use_Zero_Mc', u'Serv_Vnet_Flag', u'IPTV_Flag',
         u'Csum_Cs_Cpl_Cnt', u'Csum_Cs_Conslt_Cnt', u'Csum_Cs_Oper_Cnt', u'Csum_Cs_Hnd_Cnt', u'Csum_Cs_Other_Cnt',
         u'R3A_Fix_Inv_Amt', u'R3A_Fix_Call_Cnt', u'R3A_Fix_Call_Dur', u'Fix_Call_Cnt_Trend', u'Fix_Call_Dur_Trend',
         u'Fix_Inv_Amt_Trend', u'R3A_CDMA_Inv_Amt', u'R3A_CDMA_Call_Cnt', u'R3A_CDMA_Call_Dur', u'CDMA_Call_Cnt_Trend',
         u'CDMA_Call_Dur_Trend', u'CDMA_Inv_Amt_Trend',
         u'Csum_Fix_Comp_Cnt', u'Csum_CDMA_Comp_Cnt',
         u'Churn_Flag', 'qf_flag'
         ]

    X.rename(columns=dict(zip(e, d)), inplace=True)
    clf = joblib.load('sample0508.pkl')
    # plot_importance(clf, max_num_features=20)
    # plt.show()

    y_test_pred = clf.predict(X)
    y_test_pred1 = clf.predict_proba(X)
    test_report = metrics.classification_report(Y, y_test_pred)
    print(test_report)

    out = pd.DataFrame(
        {'Prd_Inst_Id': train["Prd_Inst_Id"].as_matrix(), "t": train["ls_flag1"].as_matrix(),
         'flag': y_test_pred.astype(np.int32), 'flag1': y_test_pred1[:, 1].astype(np.float)})
    out.to_csv("out1125.txt", sep=",", columns=["Prd_Inst_Id", "t", "flag", "flag1"], index=False, header=False)

    # submission = pd.DataFrame({
    #     'Prd_Inst_Id': train["Prd_Inst_Id"].as_matrix(),
    #     'flag': y_test_pred.astype(np.int32)
    # })
    # submission.to_csv('titanic.csv', index=False)


# def single_model_stacking(clf):
#     """用python参加Kaggle的些许经验总结"""
#     skf = list(StratifiedKFold(y, 10))
#     dataset_blend_train = np.zeros((Xtrain.shape[0], len(set(y.tolist()))))
#     # dataset_blend_test = np.zeros((Xtest.shape[0],len(set(y.tolist()))))
#     dataset_blend_test_list=[]
#     loglossList=[]
#     for i, (train, test) in enumerate(skf):
#     # dataset_blend_test_j = []
#         X_train = Xtrain[train]
#         y_train =dummy_y[train]
#         X_val = Xtrain[test]
#         y_val = dummy_y[test]
#         if clf=='NN_fit':
#             fold_pred,pred=NN_fit(X_train, y_train,X_val,y_val)
#         if clf=='xgb_fit':
#             fold_pred,pred=xgb_fit(X_train, y_train,X_val,y_val)
#         if clf=='lr_fit':
#             fold_pred,pred=lr_fit(X_train, y_train,X_val,y_val)
#         print('Fold %d, logloss:%f '%(i,log_loss(y_val,fold_pred)))
#         dataset_blend_train[test, :] = fold_pred dataset_blend_test_list.append( pred )
#         loglossList.append(log_loss(y_val,fold_pred))
#     dataset_blend_test = np.mean(dataset_blend_test_list,axis=0)
#     print('average log loss is :',np.mean(log_loss(y_val,fold_pred)))
#     print ("Blending.")
#     clf = LogisticRegression(multi_class='multinomial',solver='lbfgs')
#     clf.fit(dataset_blend_train, np.argmax(dummy_y,axis=1))
#     pred = clf.predict_proba(dataset_blend_test)
#     return pred


def data_explore(df):
    print(df['ls_flag1'].value_counts())
    #
    df['ls_flag1'].plot(kind='hist')
    # plt.show()
    print(df.info())


def opti_clf(train):
    # XGBoost调试参数

    # 加载测试集，训练集
    X = train[[x for x in train.columns if x not in [id, flag]]]
    Y = train[flag]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=5)

    sm = SMOTE(random_state=5)
    x_train, y_train = sm.fit_sample(x_train, y_train)

    # cv_params = {'n_estimators': [1100, 1200, 1300, 1500, 2000]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

    cv_params = {'max_depth': [3, 4, 5, 6, 7, 8, 9, 10], 'min_child_weight': [1, 2, 3, 4, 5, 6]}
    other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'subsample': [0.6, 0.7, 0.8, 0.9], 'colsample_bytree': [0.6, 0.7, 0.8, 0.9]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'reg_alpha': [0.05, 0.1, 1, 2, 3], 'reg_lambda': [0.05, 0.1, 1, 2, 3]}
    # other_params = {'learning_rate': 0.1, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}
    #
    # cv_params = {'learning_rate': [0.01, 0.05, 0.07, 0.1, 0.2]}
    # other_params = {'learning_rate': 0.05, 'n_estimators': 2000, 'max_depth': 9, 'min_child_weight': 5, 'seed': 0,
    #                 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.2, 'reg_alpha': 0, 'reg_lambda': 1}

    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='neg_mean_absolute_error', cv=5,
                                 verbose=1, n_jobs=4)
    optimized_GBM.fit(x_train, y_train)
    evalute_result = optimized_GBM.grid_scores_
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))


def ensemble1(X, y):
    clf1 = linear_model.LogisticRegression(penalty='l1', C=0.1, max_iter=200, class_weight='balanced',
                                           random_state=2017, )
    clf2 = RandomForestClassifier(random_state=20, n_estimators=500, max_features=6, max_depth=4, )
    clf3 = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, min_samples_split=4, min_samples_leaf=1,
                                      max_depth=3,
                                      max_features=None, subsample=1, random_state=2017)

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


if __name__ == '__main__':
    stime = time.time()
    print('start time:{}'.format(stime))
    df = loaddata("chen_hskd2_03.txt")

    opti_clf(df)
    # weakclf(df)
    # train_data(df)
    # pred_test(df)

    etime = time.time()
    print("end time:{}  duration:{}".format(etime, etime - stime))
