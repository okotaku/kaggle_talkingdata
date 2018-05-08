import numpy as np
import pandas as pd
import gc
import os


def prep_data(train_df):
    predictors=[]
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]
    debug=1
    frm,to = 1, 1
    print('Extracting new features...')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['click_time']= pd.to_datetime(train_df['click_time'])
    
    gc.collect()
    
    train_df['in_test_hh'] = (   3 
                         - 2*train_df['hour'].isin(  most_freq_hours_in_test_data ) 
                         - 1*train_df['hour'].isin( least_freq_hours_in_test_data ) ).astype('uint8')

    print('group by : ip_day_test_hh')
    gp = train_df[['ip', 'day', 'in_test_hh', 'channel']].groupby(by=['ip', 'day',
             'in_test_hh'])[['channel']].count().reset_index().rename(index=str, 
             columns={'channel': 'nip_day_test_hh'})
    train_df = train_df.merge(gp, on=['ip','day','in_test_hh'], how='left')
    del gp
    train_df.drop(['in_test_hh'], axis=1, inplace=True)
    train_df['nip_day_test_hh'] = train_df['nip_day_test_hh'].astype('uint32')
    gc.collect()
    
    GROUP_BY_NEXT_CLICKS = [
    {'groupby': ['ip']},
    {'groupby': ['ip', 'app']},
    {'groupby': ['app']},
    {'groupby': ['ip', 'os']},
    {'groupby': ['ip', 'os', 'device', 'app']}
    ]

    # Calculate the time to next click for each group
    for spec in GROUP_BY_NEXT_CLICKS:
    
       # Name of new feature
        new_feature = '{}_{}'.format('_'.join(spec['groupby']),"next")    
    
        # Unique list of features to select
        all_features = spec['groupby'] + ['click_time']
        train_df[new_feature] = (train_df[all_features].groupby(spec[
            'groupby']).click_time.shift(-1) - train_df.click_time).dt.seconds.astype('float32')
        predictors.append(new_feature)
        gc.collect()
    
    naddfeat=15
    for i in range(0,naddfeat):
        if i==0: selcols=['ip', 'channel']; QQ=4;
        if i==1: selcols=['ip', 'device', 'os', 'app']; QQ=5;
        if i==2: selcols=['ip', 'day', 'hour']; QQ=4;
        if i==3: selcols=['ip', 'app']; QQ=4;
        if i==4: selcols=['ip', 'app', 'os']; QQ=4;
        if i==5: selcols=['ip', 'device']; QQ=4;
        if i==6: selcols=['ip', 'os']; QQ=5;
        if i==7: selcols=['ip', 'device', 'os', 'app']; QQ=4;
        if i==8: selcols=['ip', 'app', 'channel']; QQ=4;
        if i==9: selcols=['app', 'os', 'channel']; QQ=4;
        if i==10: selcols=['ip', 'app']; QQ=5;
        if i==11: selcols=['ip', 'device']; QQ=5;
        if i==12: selcols=['ip', 'app', 'device']; QQ=5;
        if i==13: selcols=['ip', 'app']; QQ=6;
        if i==14: selcols=['ip', 'app', 'device']; QQ=6;
        print('selcols',selcols,'QQ',QQ)
        
        filename='X%d_%d_%d.csv'%(i,frm,to)
        
        if os.path.exists(filename):
            if QQ==5: 
                gp=pd.read_csv(filename,header=None)
                train_df['X'+str(i)]=gp
            else: 
                gp=pd.read_csv(filename)
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
        else:
            if QQ==0:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].count().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==1:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].mean().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==2:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].var().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==3:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].skew().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
            if QQ==4:
                gp = train_df[selcols].groupby(by=selcols[0:len(selcols)-1])[selcols[len(selcols)-1]].nunique().reset_index().\
                    rename(index=str, columns={selcols[len(selcols)-1]: 'X'+str(i)})
                train_df = train_df.merge(gp, on=selcols[0:len(selcols)-1], how='left')
                train_df['X'+str(i)] = train_df['X'+str(i)].astype('uint16')
            if QQ==5:
                gp = train_df[selcols].groupby(by=selcols).cumcount()
                train_df['X'+str(i)]=gp.values.astype('uint16')
            if QQ==6:
                gp = train_df.iloc[::-1][selcols].groupby(by=selcols).cumcount()
                train_df['X'+str(i)]=gp.values.astype('uint16')
            
            if not debug:
                 gp.to_csv(filename,index=False)
            
        del gp
        gc.collect()    

    print('doing nextClick')
    
    new_feature = 'nextClick'
    train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    train_df[new_feature] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1).fillna(3000000000) - train_df.click_time).astype(np.float32)
    predictors.append(new_feature)
    
    #train_df[new_feature+'_shift'] = pd.DataFrame(QQ).shift(+1).values
    #predictors.append(new_feature+'_shift')
    
    del QQ
    gc.collect()

    print('grouping by ip-day-hour combination...')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_tcount'})
    train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
    train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
    del gp
    gc.collect()

    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
    del gp
    gc.collect()
    
    print('grouping by ip-app combination...')
    gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].var().reset_index().rename(index=str, columns={'channel': 'ip_app_var'})
    train_df = train_df.merge(gp, on=['ip','app'], how='left')
    train_df['ip_app_var'] = train_df['ip_app_var'].astype('float32')
    del gp
    gc.collect()

    # Adding features with var and mean hour (inspired from nuhsikander's script)
    print('grouping by : ip_day_chl_var_hour')
    gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
    train_df['ip_tchan_count'] = train_df['ip_tchan_count'].astype('float32')
    del gp
    gc.collect()

    print('grouping by : ip_app_os_var_hour')
    gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
    train_df['ip_app_os_var'] = train_df['ip_app_os_var'].astype('float32')
    del gp
    gc.collect()

    print('grouping by : ip_app_channel_var_day')
    gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    train_df['ip_app_channel_var_day'] = train_df['ip_app_channel_var_day'].astype('float32')
    del gp
    gc.collect()

    print('grouping by : ip_app_chl_mean_hour')
    gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_var_hour'})
    train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
    train_df['ip_app_channel_var_hour'] = train_df['ip_app_channel_var_hour'].astype('float32')
    del gp
    gc.collect()
    
    print('grouping by : ip_day_chl_mean_hour')
    gp = train_df[['ip','day', 'channel','hour']].groupby(by=['ip', 'day', 'channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_day_channel_var_hour'})
    train_df = train_df.merge(gp, on=['ip','day', 'channel'], how='left')
    train_df['ip_day_channel_var_hour'] = train_df['ip_day_channel_var_hour'].astype('float32')
    del gp
    gc.collect()

    print("vars and data type: ")
    train_df.info()

    predictors.extend(['app','device','os', 'channel', 'hour', 
                  'ip_tcount', 'ip_tchan_count', 'ip_app_count',
                  'ip_app_os_var', 'ip_app_var', 'ip_day_channel_var_hour',
                  'ip_app_channel_var_day','ip_app_channel_var_hour', 'nip_day_test_hh'])
    for i in range(0,naddfeat):
        predictors.append('X'+str(i))
        
    print('predictors',predictors)
    
    return(train_df, predictors)

