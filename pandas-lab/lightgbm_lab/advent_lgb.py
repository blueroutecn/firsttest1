#!/usr/bin/env python
#coding=utf-8


__author__ = 'Haizheng'
__date__ = ''


import json
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

try:
    import cPickle as pickle
except BaseException:
    import pickle

"""
    设置特征名称
    直接使用类别特征，不用one-hot
    保存模型 json ，加载模型 pickle
    特征重要性，特征名称
    加载模型，继续训练
    调参
    回调
    
"""
# load or create your dataset
print('加载数据...')
df_train = pd.read_csv('/home/johnhill_cn/PycharmProjects/firsttest/data/binary.train', header=None, sep='\t')
df_test = pd.read_csv('/home/johnhill_cn/PycharmProjects/firsttest/data/binary.test', header=None, sep='\t')
W_train = pd.read_csv('/home/johnhill_cn/PycharmProjects/firsttest/data/binary.train.weight', header=None)[0]
W_test = pd.read_csv('/home/johnhill_cn/PycharmProjects/firsttest/data/binary.test.weight', header=None)[0]

# 将dataframe转换成nparray
y_train = df_train[0].values
y_test = df_test[0].values
X_train = df_train.drop(0, axis=1).values
X_test = df_test.drop(0, axis=1).values

# 获取行数和维度信息
num_train, num_feature = X_train.shape


# 创建lgb的数据集，如果复用，free_raw_data=False
# 稀疏特征(Sparse Features)
lgb_train = lgb.Dataset(X_train,
                        y_train, # label
                        weight=W_train, free_raw_data=False)

# dataset类中执行流程
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train,
                       weight=W_test, free_raw_data=False)


# 设置clf的参数dict boost参数
params = {
    'boosting_type': 'gbdt',  # 因为是
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# 设置特征名称 feature_name List,哪里调用？
feature_name = ['feature_' + str(col) for col in range(num_feature)]

print('开始训练...')
print(lgb_train)
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,  # 用自己验证模型效果
                feature_name=feature_name,
                categorical_feature=[21]) # list 使用columns或则index

# check feature name
print('完成前10轮...')
# 将对象 x 转换为表达式字符串 repr
print('7th feature name is:', gbm.best_iteration)
print('8th feature name is:', repr(lgb_train.feature_name[7]))

# 保存模型
gbm.save_model('model.txt')

# dump json 理解为序列化模型
print('Dump model to JSON...')
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)

# 特征名陈
print('Feature names:', gbm.feature_name())


# 特征重要读
l = list(gbm.feature_importance())
# print('最佳:', gbm.best_iteration, gbm.best_score)
print(sorted(dict(zip(feature_name,l)).items(), key=lambda d: d[1], reverse=True))

# 加载模型
print('加载模型')
bst = lgb.Booster(model_file='model.txt') # 对应clf.save_model

# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test, num_iteration=bst.best_iteration)
# eval with loaded model
print('The rmse of loaded model\'s prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# dump model with pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
y_pred = pkl_bst.predict(X_test,
                         num_iteration=8
                         )
# 评估模型
"""
objective [ default=reg:linear ] 这个参数定义需要被最小化的损失函数。最常用的值有
    "reg:linear" --线性回归
    "reg:logistic" --逻辑回归
    "binary:logistic" --二分类的逻辑回归，返回预测的概率(不是类别)
    "binary:logitraw" --输出归一化前的得分
    "count:poisson" --poisson regression for count data, output mean of poisson distribution
        max_delta_step is set to 0.7 by default in poisson regression (used to safeguard optimization)
    "multi:softmax" --设定XGBoost做多分类，你需要同时设定num_class(类别数)的值
    "multi:softprob" --输出维度为ndata * nclass的概率矩阵
    "rank:pairwise" --设定XGBoost去完成排序问题(最小化pairwise loss)
    "reg:gamma" --gamma regression with log-link. Output is a mean of gamma distribution. It might be useful, e.g., for modeling insurance claims severity, or for any outcome that might be gamma-distributed
    "reg:tweedie" --Tweedie regression with log-link. It might be useful, e.g., for modeling total loss in insurance, or for any outcome that might be Tweedie-distributed.

metric
    "rmse": 均方误差
    "mae": 绝对平均误差
    "logloss": negative log损失
    "error": 二分类的错误率
    "error@t": 通过提供t为阈值(而不是0.5)，计算错误率
    "merror": 多分类的错误类，计算公式为#(wrong cases)/#(all cases).
    "mlogloss": 多类log损失
    "auc": ROC曲线下方的面积 for ranking evaluation.
    "ndcg":Normalized Discounted Cumulative Gain
    "map":平均准确率
    "ndcg@n","map@n": n can be assigned as an integer to cut off the top positions in the lists for evaluation.
    "ndcg-","map-","ndcg@n-","map@n-": In XGBoost, NDCG and MAP will evaluate the score of a list without any positive samples as 1. By adding "-" in the evaluation metric XGBoost will evaluate these score as 0 to be consistent under some conditions. training repeatedly


"""
print('The rmse of pickled model\'s prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)

# continue training
# init_model accepts:
# 1. model file name
# 2. Booster()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model='model.txt',
                valid_sets=lgb_eval)

print('Finish 10 - 20 rounds with model file...')

# decay learning rates
# learning_rates accepts:
# 1. list/tuple with length = num_boost_round
# 2. function(curr_iter)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,   # str表示读外部模型， boost()
                learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                valid_sets=lgb_eval)

print('Finish 20 - 30 rounds with decay learning rates...')

# 修改其他参数
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

print('Finish 30 - 40 rounds with changing bagging_fraction...')


# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, value: array, is_higher_better: bool
# binary error
def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                fobj=loglikelood,
                feval=binary_error,
                valid_sets=lgb_eval)

print('Finish 40 - 50 rounds with self-defined objective function and eval metric...')

print('Start a new training job...')


# callback 先传给它一个函数，好在合适的时候调用，以完成目标任务。
def reset_metrics():
    def callback(env):
        lgb_eval_new = lgb.Dataset(X_test, y_test, reference=lgb_train)
        if env.iteration - env.begin_iteration == 5:
            print('Add a new valid dataset at iteration 5...')
            env.model.add_valid(lgb_eval_new, 'new valid')
    callback.before_iteration = True
    callback.order = 0
    return callback


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_train,
                callbacks=[reset_metrics()])

print('Finish first 10 rounds with callback function...')


"""
train_goalkeeper_data = data[~data['gk'].isnull()].copy()
"""