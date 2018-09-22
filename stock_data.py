#/usr/bin/python
# -*- coding: utf-8 -*-

from qntstock.local import FS_PATH, FS_PATH_OL
import tushare as ts
import pandas as pd
import os
import time
import datetime
import random
from urllib.error import HTTPError


def get_stock_list(way_path=None, autype='D'):
    if way_path is None:
        way_path = FS_PATH
    if not os.path.exists(way_path+'/'+autype):
        raise OSError('path ', way_path+'/'+autype, 'does not exist.')
    l = os.listdir(way_path+'/'+autype)
    return [i.split('.')[0] for i in l]


def get_stock_data(code, start_date=None, end_date=None, way='fs', way_path=None, autype='D'):
    '''
    get stock data df

    TODO: add choice about autype: hfq, qfq, zc
    ---
    way:    'csv':  do not use code. way_path should be given an absolute path of stock data csv file. if default, return an empty
            pd.DataFrame()

            'fs':   way_path should be given a path that saving all pre-download stock data files. if default,
            using FS_PATH in local.py. if no stock code in the path, return an empty pd.DataFrame()

            'web': way_path should be given a conn handle for ts.bar(...,conn,...) param, if default, using a
            temp ts.get_apis()

            # 'db':   way_path should be given a series of using databases. if default, using 'stock' series. if no
            # dataframe returned, return an empty pd.DataFrame()
    ---
    return dataframe
    '''

    if way not in ['csv', 'fs', 'web']:
        raise OSError('"way" accept for "csv", "fs", "web".')

    if way =='fs':
        if way_path is None:
            way_path = FS_PATH
        way_path = way_path+'/'+autype+'/'+code+'.csv'

    if way in ['csv', 'fs'] and not os.path.exists(way_path):
        raise OSError('file ', way_path, 'does not exist.')

    if way == 'web':
        if way_path is None:
            conn = ts.get_apis()
        else:
            conn = way_path
        df = ts.bar(code, conn=conn, start_date=start_date, end_date=end_date, adj='hfq', freq=autype, factors=['tor'])
        if df is not None and len(df) > 0:
            df = df.reset_index().sort_values('datetime')
            df = df.rename(columns={'datetime':'date', 'vol':'volume'})
            df = df[['date','open','close','high','low','volume','amount','tor']]
        else:
            df = pd.DataFrame(None, columns=['date','open','close','high','low','volume','amount','tor'])
    else:   #'csv' or 'fs'
        df = pd.read_csv(way_path)

        if way == 'csv' and not set(['date','open','close','high','low','volume','amount','tor']).issubset(set(df.columns)):
            raise OSError('the csv file does not contain the necessary columns:\n[date,open,close,high,low,volume,amount,tor]')

        if autype in ['1MIN','5MIN','15MIN','30MIN','60MIN']:
            end_date = end_date+' 24:00:00'
            start_date = start_date+' 00:00:00'
        if start_date is not None and end_date is not None:
            df = df[(df['date']>=start_date) & (df['date']<=end_date)]
        elif start_date is not None:
            df = df[df['date']>=start_date]
        elif end_date is not None:
            df = df[df['date']<=end_date]

    df = df[df['volume']>10]
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
    return df


def update_stock_data(code, new_data_df, way='csv', way_path=None, autype='D'):
    '''
    update stock data using new_data_df, if one day exist in new_data_df, using values in new_data_df.

    ---
    way:    'csv':  way_path should be given an absolute path of stock data csv file. if default, return an empty
            pd.DataFrame()

            'fs':   way_path should be given a path that saving all pre-download stock data files. if default,
            using FS_PATH in local.py. if no stock code in the path, return an empty pd.DataFrame()

            # 'db':   way_path should be given a series of using databases. if default, using 'stock' series. if no
            # dataframe returned, return an empty pd.DataFrame()

    ---
    return  None
    '''
    if new_data_df is None or len(new_data_df) == 0:
        print('Warning: new_data_df is None or empty dataframe(length = 0), do nothing and return')
        return

    if way not in ['csv', 'fs']:
        raise OSError('"way" accept for "csv", "fs".')

    if way =='fs':
        if way_path is None:
            way_path = FS_PATH
        way_path = way_path+'/'+autype+'/'+code+'.csv'

    if not os.path.exists(way_path):
        print('Warning: file ', way_path, 'does not exist.')
        if way =='csv':
            print('writing new_data_df to path:', way_path, '. columns of new dataframe are:', new_data_df.columns)
            new_data_df.to_csv(way_path, index=False)
        elif way == 'fs':
            standard_columns = ['date','open','close','high','low','volume','amount','tor']
            if set(standard_columns).issubset(set(new_data_df.columns)):
                print('writing new_data_df to path:', way_path, '. columns of new dataframe are:', standard_columns)
                df = new_data_df[standard_columns]
                df.to_csv(way_path, index=False)
            else:
                raise OSError('could not write new_data_df, the columns are not standard. the standard columns are:', standard_columns)
    else:   # old_df and new_data_df both have data
        old_df = pd.read_csv(way_path)

        if not set(old_df.columns) == set(new_data_df.columns):
            raise OSError('could not combine and update, columns of two dataframe are not match.')

        if 'date' not in old_df.columns:
            raise OSError('"date" not in columns of dataframe. please change the relative column name to "date".')

        new_data_df['date'] = new_data_df['date'].astype('str')
        overlap_date_set = set(new_data_df['date']) & set(old_df['date'])
        old_df = old_df[True-old_df['date'].isin(overlap_date_set)]

        df = pd.concat([old_df,new_data_df])
        df = df[df['volume']>10]
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        df.to_csv(way_path, index=False)


def update_all_stock_data(start_date, end_date, fs_path=None, code_set=None, autype='D'):
    '''
    update all stock data

    ---
    return None
    '''
    if fs_path is None:
        fs_path = FS_PATH
    if not os.path.exists(fs_path):
        raise OSError('path ', fs_path, 'does not exist.')
    stock_basics = ts.get_stock_basics()
    new_code_set = set(stock_basics.index) if code_set is None else set(code_set)
    done_code_set = set()
    tmp_fail_code_set = set()
    fail_code_set = set()
    cnt = 0
    print('updating all stock data...')
    while(len(new_code_set)>0):
        try:
            conn = ts.get_apis()
            for code in new_code_set:
                print('now downloading new data ', code)
                new_data_df = get_stock_data(code, start_date, end_date, 'web', conn, autype)
                update_stock_data(code, new_data_df, 'fs', fs_path, autype)
                done_code_set.add(code)
                t = random.choice([0.1,0.3,0.5])
                time.sleep(t)
                cnt += 1
        except OSError as err:
            print('Error: could not update ', code, '. err msg:', err,'adding to list and will try again later.')
            new_code_set = new_code_set - done_code_set
            done_code_set.clear()
            tmp_fail_code_set.add(code)
            new_code_set = new_code_set - tmp_fail_code_set
            t = random.choice([20,30,25])
            time.sleep(t)

    print('now updated file number is :', cnt, 'failed and re-trying code number is:', len(tmp_fail_code_set))
    print('==========')
    print('==========')
    print('now checking for failed stock code again...')
    done_code_set.clear()
    while(len(tmp_fail_code_set)>0):
        try:
            conn = ts.get_apis()
            for code in tmp_fail_code_set:
                print('now re-downloading new data ', code)
                new_data_df = get_stock_data(code, start_date, end_date, 'web', conn, autype)
                update_stock_data(code, new_data_df, 'fs', fs_path, autype)
                done_code_set.add(code)
                t = random.choice([2,1.7,1.9,1.5])
                time.sleep(t)
        except OSError as err:
            print('Error: could not update ', code, '. err msg:', err)
            tmp_fail_code_set = tmp_fail_code_set - done_code_set
            done_code_set.clear()
            fail_code_set.add(code)
            tmp_fail_code_set = tmp_fail_code_set - fail_code_set
            t = random.choice([10,20,13,17])
            time.sleep(t)
    print('==========')
    print('==========')
    print('Failed code set:', fail_code_set)


def update_all_today_online(last_date=None, code_set=None, autype='D'):
    '''
    update all stock data today, using ts.get_day_all(date).

    NOTE:   1. not hfq data, only use during 9:30am - 15:00pm for speed.
            must use update_stock_data() to overwrite.
            2. if one stock ting2 pai2 in last_date, this function will not conclude this stock.
            3. the reference day to compute adj is *before* and *the nearest to* param "last_date"
            details see the building of code_set
    ---
    return  None
    '''
    today_date = datetime.datetime.now().strftime('%Y-%m-%d')
    today_all_df = ts.get_day_all(today_date)
    lastday_date = last_date
    if lastday_date > today_date:
        raise OSError('"last_date" is later than today. please check again.')
    if lastday_date is  None:
        lastday_date = today_date
    lastday_date, lastday_all_df = _get_nearest_trade_date_and_df_before(lastday_date)

    # 根据昨天的数据和本地csv数据，计算出后复权的权重比例
    # 再计算今天对应的后复权股价，写进csv文件里
    today_all_df = today_all_df[today_all_df['volume']>10]
    lastday_all_df = lastday_all_df[lastday_all_df['volume']>10]
    new_code_set = (set(today_all_df['code']) & set(lastday_all_df['code'])) if code_set is None else set(code_set)

    for code in new_code_set:
        lastday_code_df = lastday_all_df[lastday_all_df['code']==code]
        old_df = get_stock_data(code, way='fs')
        if lastday_date not in set(old_df['date']):
            print('Warning: ', code, ' the last trade date in filesystem is not match with it in web.')
            print('    jump to next stock code. if all code makes warning, change the param "last_date".')
        else:
            old_last_series = old_df[old_df['date']==lastday_date]
            adj = _comput_adj_with(old_last_series, lastday_code_df)

            new_df = today_all_df[today_all_df['code'] == code]
            new_df['date'] = today_date
            new_df = new_df[['date','open','price','high','low','volume','amount','turnover']]
            new_df.columns = ['date','open','close','high','low','volume','amount','tor']
            new_df['open'] = round(new_df['open'] * adj, 2)
            new_df['close'] = round(new_df['close'] * adj, 2)
            new_df['high'] = round(new_df['high'] * adj, 2)
            new_df['low'] = round(new_df['low'] * adj, 2)

            old_df = old_df[old_df['date']!=today_date]

            df = pd.concat([old_df, new_df])
            df.to_csv(FS_PATH_OL+'/'+code+'.csv', index=False)

def _get_nearest_trade_date_and_df_before(lastday_date):
    lastday = datetime.datetime.strptime(lastday_date, '%Y-%m-%d')
    lastday_all_df = None
    e = None
    for _ in range(18):
        lastday = lastday - datetime.timedelta(days=1)
        try_count = 0
        while try_count < 3:
            try:
                lastday_all_df = ts.get_day_all(lastday.strftime('%Y-%m-%d'))
            except HTTPError as err:
                if err.msg == 'Not Found':
                    try_count += 1
                    e = err
                else:
                    raise err
            if lastday_all_df is not None:
                break
        if lastday_all_df is not None:
            break
    if lastday_all_df is None:
        raise e
    return lastday.strftime('%Y-%m-%d'), lastday_all_df


def _comput_adj_with(old_last_series, lastday_code_df):
    adj1 = old_last_series['open'].values[0] / lastday_code_df['open'].values[0]
    adj2 = old_last_series['high'].values[0] / lastday_code_df['high'].values[0]
    adj3 = old_last_series['low'].values[0] / lastday_code_df['low'].values[0]
    adj4 = old_last_series['close'].values[0] / lastday_code_df['price'].values[0]
    return (adj1 + adj2 + adj3 + adj4) / 4


if __name__ == "__main__":
    #print(get_stock_data('002478',
    #    start_date='2017-01-01',way='fs',way_path='/home/wyn/data/stock_data/'))

    #update_all_stock_data('2017-12-29','2018-02-10')
    #update_all_stock_data('2014-01-01','2018-02-10', autype='60MIN')
    #update_all_today_online('2017-12-25')
    # update_all_stock_data('2018-04-06','2018-09-08', autype='D')
    # update_all_stock_data('2018-04-06','2018-09-08', autype='W')
    update_all_stock_data('2018-04-06','2018-09-08', autype='60MIN')
    #print(get_stock_data('603696','2018-02-10','2018-03-24','web', autype='60MIN'))
