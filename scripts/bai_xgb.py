import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    # print('the Id of train_df while function before merge: ',id(df)) # the same with train_df
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    # print('the Id of train_df while function after merge: ',id(df)) # id changes
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left', copy=False)
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )

def lgb_modelfit_nocv(dtrain, dvalid, predictors, target='target', feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, categorical_features=None,metrics='auc'):
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
    model.fit(dtrain[predictors].values, dtrain[target].values)
    
    del dtrain
    
    gc.collect()

    return bst1

# --------------------------------------------------------------------------------------------------------------
def DO(frm,to,fileno):
    dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint16',
            'os'            : 'uint16',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8', # 【consider bool?need test】
            'click_id'      : 'uint32', # 【consider 'attributed_time'?】
            }
    
    print('loading train data...',frm,to)
    # usecols:Using this parameter results in much faster parsing time and lower memory usage.
    train_df = pd.read_pickle("./data/training.pkl.gz")
    val_df = pd.read_pickle("./data/validation.pkl.gz")

    print('loading test data...')
    test_df = pd.read_csv("./data/test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

    len_train = len(train_df)
    len_test = len(test_df)
    train_df=train_df.append(val_df)
    train_df=train_df.append(test_df) # Shouldn't process individually,because of lots of count,mean,var variables
    # train_df['is_attributed'] = train_df['is_attributed'].fillna(-1)
    train_df['is_attributed'].fillna(-1,inplace=True)
    train_df['is_attributed'] = train_df['is_attributed'].astype('uint8',copy=False)
    # train_df['click_id'] = train_df['click_id'].fillna(-1)
    train_df['click_id'].fillna(-1,inplace=True)
    train_df['click_id'] = train_df['click_id'].astype('uint32',copy=False)
    
    del test_df, val_df
    gc.collect()
    
    print('Extracting new features...')
    train_df['click_time']= pd.to_datetime(train_df['click_time'])
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    gc.collect()
    
    train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint16', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['app'], 'channel', 'X6','uint8', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8','uint8', show_max=False ); gc.collect()
    train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=False ); gc.collect()
    train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=False ); gc.collect()
    train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'channel', 'A0', show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'channel'], 'A1', show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'device', 'os','app'], 'A2', show_max=False ); gc.collect()
    # ip-device-hour?

    train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount','uint16',show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count','uint32', show_max=False ); gc.collect()
    train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=False ); gc.collect()
    train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=False ); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=False ); gc.collect()
    train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=False ); gc.collect()
    train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=False ); gc.collect()


# nextclick----------------------------------------------------------------------------------------------------------
    # print('doing nextClick')
    # predictors=[]
    # new_feature = 'nextClick'
    # filename='nextClick_%d_%d.csv'%(frm,to)

    # if os.path.exists(filename):
    #     print('loading from save file')
    #     QQ=pd.read_csv(filename).values
    # else:
    #     D=2**26
    #     train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
    #         + "_" + train_df['os'].astype(str)).apply(hash) % D
    #     # from 1970/1/1, 50year*365day*24*60*60=1,576,800,000 seconds, so 2,000,000,000 is enough
    #     click_buffer= np.full(D, 3000000000, dtype=np.uint32) # Return a new array of given shape and type, filled with fill_value.
        
    #     train_df['epochtime']= train_df['click_time'].astype(np.int64,copy=False) // 10 ** 9
    #     next_clicks= []
    #     # After reverse, the time becomes future to past, make next_clicks positive
    #     for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
    #         next_clicks.append(click_buffer[category]-t)
    #         click_buffer[category]= t
    #     del(click_buffer)
    #     QQ= list(reversed(next_clicks))

    #     if not debug:
    #         print('saving')
    #         pd.DataFrame(QQ).to_csv(filename,index=False)
            
    # train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)

    # train_df[new_feature] = pd.Series(QQ).astype('float32',copy=False)
    # predictors.append(new_feature)
    # train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
    # predictors.append(new_feature+'_shift')
    
    # del QQ
    # gc.collect()
    
#=====================================================================================================
    print('doing nextClick 2...')
    predictors=[]
    
    train_df['click_time'] = (train_df['click_time'].astype(np.int64,copy=False) // 10 ** 9).astype(np.int32,copy=False)
    train_df['nextClick'] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - train_df.click_time).astype(np.float32,copy=False)
    print(train_df['nextClick'].head(30))
    train_df.drop(['click_time','day'], axis=1, inplace=True)
    predictors.append('nextClick')
    gc.collect()
    
#----------------------------------------------------------------------------------------------------------------
    print("vars and data type: ")
    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour',
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
    categorical = ['app', 'device', 'os', 'channel', 'hour',]
    print('predictors',predictors)

    test_df = train_df.iloc[-len_test:]
    val_df = train_df.iloc[len_train:-len_test]
    train_df = train_df.iloc[:len_train]
    test_df.drop(columns='is_attributed',inplace=True)
    train_df.drop(columns='click_id',inplace=True)
    
    print("train size: ", len(train_df))
    print("valid size: ", len(val_df))
    print("test size : ", len(test_df))
    train_df.info()

    sub = pd.DataFrame()
    sub['click_id'] = test_df['click_id']
    gc.collect()

    print("Training...")
    start_time = time.time()

    bst = lgb_modelfit_nocv(
                            train_df, 
                            val_df, 
                            predictors, 
                            target, 
                            early_stopping_rounds=50, 
                            verbose_eval=True, 
                            num_boost_round=2000, 
                            categorical_features=categorical)
    del train_df
    gc.collect()
    print('[{}]: model training time'.format(time.time() - start_time))

    print("Predicting...")
    sub['is_attributed'] = bst.predict_proba(test_df[predictors].values)
    del test_df
    sub.to_csv('bai-xgbsub.csv.gz', float_format='%.8f', index=False, compression='gzip')
    del sub
    gc.collect()
    
    pred_val = bst.predict_proba(val_df[predictors].values)
    del val_df
    gc.collect()
    pred_df = pd.DataFrame()
    pred_df['is_attributed']  = pred_val
    pred_df.to_csv('bai-xgbval.csv.gz', float_format='%.8f', index=False,compression='gzip')
    print("All done...")
    

# Main function-------------------------------------------------------------------------------------
if __name__ == '__main__':
    inpath = '../input/'
    
    #【In order to get 0.9798, you have to change nchunk to all and frm to 0 to use entire dataset】
    nrows=184903891-1 # the first line is columns' name
    nchunk=25000000 # 【The more the better】
    val_size=2500000
    frm=nrows-75000000
    
    debug=False
    # debug=True
    if debug:
        print('*** Debug: this is a test run for debugging purposes ***')
        frm=0
        nchunk=100000
        val_size=10000
    
    to=frm+nchunk
    
    DO(frm,to,6) # fileno start from 0