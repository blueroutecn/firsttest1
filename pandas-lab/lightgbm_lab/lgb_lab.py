#!/usr/bin/env python
# coding=utf-8


__author__ = 'Haizheng'
__date__ = ''

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from pathlib import Path

path = Path("/home/johnhill_cn/PycharmProjects/firsttest/data")


def main():
    # load or create your dataset
    print('Load data...')
    df_train = pd.read_csv(path / "regression.train", header=None, sep='\t')
    df_test = pd.read_csv(path / "regression.test", header=None, sep='\t')

    print(df_train.shape, df_test.shape)
    y_train = df_train[0].values
    y_test = df_test[0].values
    X_train = df_train.drop(0, axis=1).values
    X_test = df_test.drop(0, axis=1).values

    print(X_train.shape, X_test.shape)
    print('Start training...')

    # train
    # 回归 评估使用mse
    gbm = lgb.LGBMRegressor(objective='regression',
                            num_leaves=31,
                            learning_rate=0.1,
                            n_estimators=40)

    gbm.fit(X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='l1',
            early_stopping_rounds=5)
    print('Start predicting...')

    # predict
    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
    # eval
    print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
    # feature importances
    print('Feature importances:', list(gbm.feature_importances_))

    # other scikit-learn modules
    estimator = lgb.LGBMRegressor(num_leaves=31)
    param_grid = {
        'learning_rate': [0.01, 0.1, 1],
        'n_estimators': [20, 40]
    }
    gbm = GridSearchCV(estimator, param_grid)
    gbm.fit(X_train, y_train)
    print('Best parameters found by grid search are:', gbm.best_params_)

def sample(dtrain, dtest):
    predictors =[]
    dummies = []
    target = None
    # 01.    train    set and test    set
    train_data = lgb.Dataset(dtrain[predictors], label=dtrain[target], feature_name=list(dtrain[predictors].columns),
                             categorical_feature=dummies)

    test_data = lgb.Dataset(dtest[predictors], label=dtest[target], feature_name=list(dtest[predictors].columns),
                            categorical_feature=dummies)

    # // 02.    parameters
    param = {
        # num_leaves = 2^(max_depth) 2 **(max_depth)
        'max_depth': 6,
        'num_leaves': 64,
        'learning_rate': 0.03,
        'scale_pos_weight': 1,
        'num_threads': 40,
        'objective': 'binary',
        # Bagging参数：bagging_fraction+bagging_freq（必须同时设置）、feature_fraction
        'bagging_fraction': 0.7,
        'bagging_freq': 1,
        'min_sum_hessian_in_leaf': 100
    }

    # 样本分布非平衡数据集
    param['is_unbalance'] = 'true'
    param['metric'] = 'auc'

    # // 03.    cv and train
    bst = lgb.cv(param, train_data, num_boost_round=1000, nfold=3, early_stopping_rounds=30)

    estimators = lgb.train(param, train_data, num_boost_round=len(bst['auc-mean']))

    # // 04.    test    predict
    ypred = estimators.predict(dtest[predictors])

if __name__ == '__main__':
    main()

    # found_images = pathlib.Path('/path/').glob('**/*.jpg')
    for fl in path.glob("*.weight"):
        print(fl)

    a, b , *rest = range(10)
    print(a, b)
    print(rest)


