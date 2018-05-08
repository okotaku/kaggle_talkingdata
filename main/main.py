import gc
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from make_dfv8 import prep_data
from lgb_model import lgb_modelfit_nocv


dtypes = {
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
train_df = pd.read_pickle("./data/training.pkl.gz")
val_df = pd.read_pickle("./data/validation.pkl.gz")
test_df = pd.read_csv("./data/test.csv", dtype=dtypes, usecols=test_col)


len_train = len(train_df)
len_test = len(test_df)
y_train = train_df["is_attributed"]
y_val = val_df["is_attributed"]
val_df = val_df.drop("is_attributed", axis=1)
train_df = pd.concat((train_df, val_df), axis=0).reset_index(drop=True)
train_df = train_df.append(test_df)
del val_df
gc.collect()

print("prep data")
train_df, predictors = prep_data(train_df)
val_df = train_df[len_train:-len_test]
test_df = train_df[-len_test:]
train_df = train_df[:len_train]
gc.collect()

target = 'is_attributed'
categorical = ['app', 'device', 'os', 'channel', 'hour']

print(predictors)
bst, best_iteration = lgb_modelfit_nocv(train_df,
                        val_df,
                        predictors, 
                        y_train,
                        y_val,  
                        metrics='auc',
                        categorical_features=categorical)

#bst = xgb_modelfit_nocv(train_df, val_df, predictors, target)

val = pd.DataFrame()
print("Predicting...")
val['is_attributed'] = bst.predict(val_df[predictors],num_iteration=best_iteration)
#val['is_attributed'] = bst.predict(val_df[predictors].values)
print("writing...")
val.to_csv('lgbv8val.csv.gz',index=False,compression='gzip')


sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors],num_iteration=best_iteration)
#sub['is_attributed'] = bst.predict(test_df[predictors].values)
print("writing...")
sub.to_csv('lgbv8sub.csv.gz',index=False,compression='gzip')

joblib.dump(bst, 'lgbm.pkl') 

print("ok")
import matplotlib.pyplot as plt
ax = lgb.plot_importance(bst, max_num_features=10)
plt.savefig("fig.png")

del bst
gc.collect()

print("done!")
