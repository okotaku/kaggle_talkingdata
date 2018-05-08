#https://www.kaggle.com/panjianning/talkingdata-simple-lightgbm-0-9772/code
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from contextlib import contextmanager

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')
    
def df_add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0) + 1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+'_count'] = counts[unqtags]
    
def make_count_features(df):
    with timer("add count features"):
        df['click_time']= pd.to_datetime(df['click_time'])
        dt= df['click_time'].dt
        df['day'] = dt.day.astype('uint8')
        df['hour'] = dt.hour.astype('uint8')
        df['minute'] = dt.minute.astype('uint8')
        del(dt)
        
        df_add_counts(df, ['ip'])
        df_add_counts(df, ['ip','day','hour','minute'])
        df_add_counts(df, ['os','device'])
        df_add_counts(df, ['os','app','channel'])
        
        df_add_counts(df, ['ip', 'day', 'hour'])
        df_add_counts(df, ['ip', 'app'])
        df_add_counts(df, ['ip', 'app', 'os'])
        df_add_counts(df, ['ip', 'device'])
        df_add_counts(df, ['app', 'channel'])

def make_next_click_feature(df):
    with timer("Adding next click times"):
        df['click_time'] = (df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
        df['next_click'] = (df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1).fillna(3000000000) - df.click_time).astype(np.float32)
        
path = './data/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

with timer("load training data"):
    train_df = pd.read_pickle("./data/training.pkl.gz")
    
with timer("load val data"):
    val_df = pd.read_pickle("./data/validation.pkl.gz")
    
with timer("load test data"):
    test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    
    
num_train = len(train_df)
num_test = len(test_df)
y_train = train_df.is_attributed.values
y_val = val_df.is_attributed.values

sub = pd.DataFrame()
sub['click_id'] = test_df.click_id.values

common_column = ['ip','app','device','os','channel','click_time']
concat_df = train_df[common_column].append(val_df[common_column])
concat_df = concat_df.append(test_df[common_column])

del train_df, val_df, test_df
gc.collect()

make_count_features(concat_df)
make_next_click_feature(concat_df)

gc.collect()

target = "is_attributed"
categorical_features = ['ip','app','os','channel','device']
predictors = list(set(concat_df.columns)-set([target])-set(['click_time']))

xgb_params = {
    'colsample_bylevel': 0.1,
    'colsample_bytree': 1.0,
    'gamma': 5.103973694670875e-08,
    'learning_rate': 0.140626707498132,
    'max_delta_step': 20,
    'max_depth': 6,
    'min_child_weight': 4,
    'n_estimators': 100,
    'reg_alpha': 1e-09,
    'reg_lambda': 1000.0,
    'scale_pos_weight': 499.99999999999994,
    'subsample': 1.0
}

model = xgb.XGBClassifier(**xgb_params)
model.fit(concat_df.iloc[:num_train][predictors].values, y_train)

 
preditions=model.predict_proba(concat_df.iloc[-num_test:][predictors].values)[:, 1]

sub['is_attributed']  = preditions

sub.to_csv('simple_xgb.csv.gz', float_format='%.8f', index=False, compression='gzip')
del sub
gc.collect()

pred_val = model.predict_proba(concat_df.iloc[num_train:-num_test][predictors].values)[:, 1]
pred_df = pd.DataFrame()
pred_df['is_attributed']  = pred_val

pred_df.to_csv('simple_xgb_val.csv.gz', float_format='%.8f', index=False,compression='gzip')
print("done")