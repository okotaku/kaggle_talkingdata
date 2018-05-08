#https://www.kaggle.com/cgundlach/lgbm-single-model-lb-9791

from sklearn.model_selection import train_test_split
import pandas as pd
import time
import gc
import numpy as np
import xgboost as xgb
import os
os.environ['OMP_NUM_THREADS'] = '4'

max_rounds = 1000
early_stop = 50
opt_rounds = 680

output_file = 'callum-lgbsub.csv'

path = "./data/"

dtypes = {
    'ip'		:'uint32',
    'app'		:'uint16',
	'device'	:'uint16',
	'os'		:'uint16',
	'channel'	:'uint16',
	'is_attributed'	:'uint8',
	'click_id'	:'uint32',
	}

print('Loading train.csv...')
train_df = pd.read_pickle("./data/training.pkl.gz")
val_df = pd.read_pickle("./data/validation.pkl.gz")

print('Load test.csv...')
test_cols = ['ip', 'app', 'device', 'os', 'click_time', 'channel', 'click_id']
test_df = pd.read_csv(path + "test.csv", dtype=dtypes, usecols=test_cols)

print('Preprocessing...')

most_freq_hours_in_test_data = [4,5,9,10,13,14]
least_freq_hours_in_test_data = [6, 11, 15]

def add_counts(df, cols):
    arr_slice = df[cols].values
    unq, unqtags, counts = np.unique(np.ravel_multi_index(arr_slice.T, arr_slice.max(axis=0)+1),
                                     return_inverse=True, return_counts=True)
    df["_".join(cols)+"_count"] = counts[unqtags]

def add_next_click(df):
    D = 2**26
    df['category'] = (df['ip'].astype(str) + "_" + df['app'].astype(str) + "_" + df['device'].astype(str) \
                      + "_" + df['os'].astype(str)).apply(hash) % D
    click_buffer = np.full(D, 3000000000, dtype=np.uint32)
    df['epochtime'] = df['click_time'].astype(np.int64) // 10 ** 9
    next_clicks = []
    for category, time in zip(reversed(df['category'].values), reversed(df['epochtime'].values)):
        next_clicks.append(click_buffer[category] - time)
        click_buffer[category] = time
    del click_buffer
    gc.collect()
    df['next_click'] = list(reversed(next_clicks))
    df.drop(['category', 'epochtime'], axis=1, inplace=True)
    

def preproc_data(df):
    
    #Extrace date info
    df['click_time']= pd.to_datetime(df['click_time'])
    df['hour'] = df['click_time'].dt.hour.astype('uint8')
    df['day'] = df['click_time'].dt.day.astype('uint8')
    df['wday'] = df['click_time'].dt.dayofweek.astype('uint8')
    gc.collect()

    #Groups
    df['in_test_hh'] = ( 3
	    		 - 2 * df['hour'].isin( most_freq_hours_in_test_data )
			 - 1 * df['hour'].isin( least_freq_hours_in_test_data )).astype('uint8')

    print('Adding next_click...')
    add_next_click(df)

    print('Grouping...')
    
    add_counts(df, ['ip'])
    add_counts(df, ['os', 'device'])
    add_counts(df, ['os', 'app', 'channel'])

    add_counts(df, ['ip', 'device'])
    add_counts(df, ['app', 'channel'])

    add_counts(df, ['ip', 'wday', 'in_test_hh'])
    add_counts(df, ['ip', 'wday', 'hour'])
    add_counts(df, ['ip', 'os', 'wday', 'hour'])
    add_counts(df, ['ip', 'app', 'wday', 'hour'])
    add_counts(df, ['ip', 'device', 'wday', 'hour'])
    add_counts(df, ['ip', 'app', 'os'])
    add_counts(df, ['wday', 'hour', 'app'])
    

    df.drop(['ip', 'day', 'click_time'], axis=1, inplace=True )
    gc.collect()

    print( df.info() )

    return df

y_train = train_df.is_attributed.values
y_val = val_df.is_attributed.values

submit = pd.DataFrame()
submit['click_id'] = test_df['click_id']

len_train = len(train_df)
len_test = len(test_df)
common_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
train_df = train_df[common_cols].append(val_df[common_cols])
train_df = train_df.append(test_df[common_cols])

train_df = preproc_data(train_df)

test_df = train_df.iloc[-len_test:]
val_df = train_df.iloc[len_train:-len_test]
train_df = train_df.iloc[:len_train]

gc.collect()

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

target = 'is_attributed'

inputs = list(set(train_df.columns) - set([target]))  
cat_vars = ['app', 'device', 'os', 'channel', 'hour', 'wday']

print('Train size:', len(train_df))
print('Valid size:', len(val_df))

gc.collect()

print('Training...')
model = xgb.XGBClassifier(**xgb_params)
model.fit(train_df[inputs].values, y_train)


print('Predicting...')
submit['is_attributed'] = model.predict_proba(test_df[inputs].values)[:, 1]

print('Creating:', output_file)
submit.to_csv('callum-xgbsub.csv.gz', float_format='%.8f', index=False, compression='gzip')


pred_val = model.predict_proba(val_df[inputs].values)[:, 1]
pred_df = pd.DataFrame()
pred_df['is_attributed']  = pred_val
pred_df.to_csv('callum-xgbval.csv.gz', float_format='%.8f', index=False,compression='gzip')
print('Done!')