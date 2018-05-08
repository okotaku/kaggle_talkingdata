import gc
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from make_dfv8 import prep_data
from lgb_model import lgb_modelfit_nocv
from load_stackcsv import load_csv

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

cvfiles = ['./val/wordbatch_fm_ftrl_val.csv', "val/callum-lgbval.csv.gz", "val/simple_xgb_val.csv.gz", "val/lgbv5val.csv.gz", "val/mdxgb_val.csv.gz"]
subfiles = ['./pred/wordbatch_fm_ftrl.csv', "./pred/callum-lgbsub.csv.gz", "pred/simple_xgb.csv.gz", "pred/lgbv5sub.csv.gz", "pred/mdxgb_sub.csv.gz"]
"""dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

train_col = ['ip','app','device','os', 'channel', 'click_time', 'is_attributed']
test_col = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
print("load data")
train_df = pd.read_pickle("./data/validation.pkl.gz")
test_df = pd.read_csv("./data/test.csv", dtype=dtypes, usecols=test_col)
len_test = len(test_df)

train_df = train_df.append(test_df)
del test_df
gc.collect()

print("prep data")
train_df, predictors = prep_data(train_df)
test_df = train_df[-len_test:]
train_df = train_df[:-len_test]
test_df.to_csv("test.csv", index=False)
train_df.to_csv("train.csv", index=False)"""

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")
train_df = train_df.fillna(-1)
test_df = test_df.fillna(-1)

predictors = ['ip_next', 'ip_app_next', 'app_next', 'ip_os_next', 'ip_os_device_app_next', 'nextClick',# 'app', 'device', 'os', 'channel', 'hour',
              'ip_tcount', 'ip_tchan_count', 'ip_app_count', 'ip_app_os_var', 'ip_app_var', 'ip_day_channel_var_hour', 'ip_app_channel_var_day', 
              'ip_app_channel_var_hour', 'nip_day_test_hh', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10', 'X11', 'X12', 'X13', 'X14']

train_df, _ = load_csv(train_df, cvfiles)#, predictors=predictors)
test_df, _ = load_csv(test_df, subfiles)
predictors2 = ['stack0_rank', 'stack1_rank', 'stack2_rank', 'stack3_rank', 'stack4_rank']

len_val = int(len(train_df)*0.33)
val_df = train_df[-len_val:]
train_df = train_df[:-len_val]

y_train = train_df["is_attributed"]
y_val = val_df["is_attributed"]

gc.collect()

target = 'is_attributed'
categorical = ['app', 'device', 'os', 'channel', 'hour']

print(predictors)
"""bst, best_iteration = lgb_modelfit_nocv(train_df,
                        val_df,
                        predictors, 
                        y_train,
                        y_val,  
                        metrics='auc',
                        categorical_features=categorical)"""

sc = MinMaxScaler()
train_df[predictors] = sc.fit_transform(train_df[predictors].values)
val_df[predictors] = sc.transform(val_df[predictors].values)

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
test_df[predictors] = sc.transform(test_df[predictors].values)
del sc
gc.collect()

print(train_df.head())

predictors = predictors + predictors2
train_df = train_df[predictors].values
val_df = val_df[predictors].values
test_df = test_df[predictors].values

bst = LogisticRegression(C=1e6)
bst.fit(train_df, y_train)
del train_df
gc.collect()

val_pred = bst.predict_proba(val_df)[:,1]
print( "val roc", roc_auc_score(y_val, val_pred ))

print("Predicting...")
#sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
sub['is_attributed'] = bst.predict_proba(test_df)[:,1]
print("writing...")
sub.to_csv('stack2ndsub.csv.gz',index=False,compression='gzip')

joblib.dump(bst, 'lgbm.pkl') 

del bst
gc.collect()

print("done!")