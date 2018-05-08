import pandas as pd
import lightgbm as lgb
import gc

train_df = pd.read_pickle("./data/training.pkl.gz")
val_df = pd.read_pickle("./data/validation.pkl.gz")
y_train = train_df["is_attributed"]
y_val = val_df["is_attributed"]
len_train = len(train_df)
train_df = train_df.append(val_df)

train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

set_ = [['app', 'os', 'device'], ['os', 'device', 'channel'], ['os', 'day', 'hour'], ['channel', 'day', 'hour']]
for selcols in set_:
    print(selcols)
    pr = "X1"
    gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
        rename(index=str, columns={selcols[len(selcols)-1]: pr})
    train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')

    del gp
    gc.collect()

    x_val = train_df[len_train:]
    x_train = train_df[:len_train]

    #del train_df
    #gc.collect()

    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    predictors = ['app','device','os', 'channel', 'hour', 'day', pr]
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':"auc",
        'learning_rate': 0.2,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 1,
        'verbose': 0
    }

    params = {
        'learning_rate': 0.20,
        #'is_unbalance': 'true', # replaced with scale_pos_weight argument
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'scale_pos_weight':200 # because training data is extremely unbalanced 
    }
    lgb_params.update(params)

    xgtrain = lgb.Dataset(x_train[predictors].values, label=y_train,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )

    xgvalid = lgb.Dataset(x_val[predictors].values, label=y_val,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )

    del x_train, x_val
    gc.collect

    evals_results = {}

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'],
                     evals_result=evals_results, 
                     num_boost_round=10,
                     verbose_eval=10, 
                     feval=None)

    train_df = train_df.drop(pr, axis=1)