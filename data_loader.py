#/usr/bin/python
# -*- coding: utf-8 -*-

from qntstock.local import FS_PATH, FS_PATH_OL, token
from qntstock.utils import Utils
from multiprocessing.pool import ThreadPool, Pool
import tushare as ts
import pandas as pd
import os
import time
import datetime
import random
from urllib.error import HTTPError

class DataLoader():
    def __init__(self, fs_path=None):
        self.fs_path = fs_path if fs_path is not None else FS_PATH
        if not os.path.exists(self.fs_path):
            raise OSError('path %s does not exist.'%self.fs_path)
        self.standard_cols = ['date','open','close','high','low','volume','amount','tor']
        self.utils = Utils()

    def get_stock_data(self, code, start_date=None, end_date=None, way='fs', way_path=None, autype='D'):
        '''
        get stock data df

        TODO: add choice about autype: hfq, qfq, zc
        ---
        way:    'csv':  do not use code. way_path should be given an absolute path of stock data csv file. if default, return an empty
                pd.DataFrame()

                'fs':   way_path should be given a path that saving all pre-download stock data files. if default,
                using FS_PATH in local.py. if no stock code in the path, return an empty pd.DataFrame()

                'web_bar': way_path should be given a conn handle for ts.bar(...,conn,...) param, should using a ts.get_apis()

                'web_k': do not need way_path

                # 'db':   way_path should be given a series of using databases. if default, using 'stock' series. if no
                # dataframe returned, return an empty pd.DataFrame()
        ---
        return dataframe
        '''

        if way not in ['csv', 'fs', 'web_bar', 'web_k']:
            raise OSError('"way" accept for "csv", "fs", "web".')

        if way =='fs':
            if way_path is None:
                way_path = self.fs_path
            way_path = os.path.join(way_path, autype, '%s.csv'%code)

        if way in ['csv', 'fs'] and not os.path.exists(way_path):
            raise OSError('file ', way_path, 'does not exist.')

        if way == 'web_bar':
            conn = way_path
            df = ts.bar(code, conn=conn, start_date=start_date, end_date=end_date, adj='hfq', freq=autype, factors=['tor'])
            if df is not None and len(df) > 0:
                df = df.reset_index().sort_values('datetime')
                df = df.rename(columns={'datetime':'date', 'vol':'volume'})
                df = df[self.standard_cols]
            else:
                df = pd.DataFrame(None, columns=self.standard_cols)
        
        elif way == 'web_k':
            df = ts.get_k_data(code, start_date, end_date, autype=None)
        else:   #'csv' or 'fs'
            df = pd.read_csv(way_path)

            if way == 'csv' and not set(self.standard_cols).issubset(set(df.columns)):
                raise OSError('the csv file does not contain the necessary columns:\n%s'%self.standard_cols)

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

    def update_stock_data(self, code, new_data_df, way='csv', way_path=None, autype='D'):
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
            self.utils.logger.info('Warning: %s new_data_df is None or length = 0, do nothing and return'%code)
            return

        if way not in ['csv', 'fs']:
            raise OSError('"way" accept for "csv", "fs".')

        if way =='fs':
            if way_path is None:
                way_path = self.fs_path
            way_path = os.path.join(way_path, autype, '%s.csv'%code)

        if not os.path.exists(way_path):
            self.utils.logger.info('Warning: file %s does not exist.'%way_path)
            if way =='csv':
                self.utils.logger.info('writing new_data_df to path: %s. columns of new dataframe are: %s'%(way_path, new_data_df.columns))
                new_data_df.to_csv(way_path, index=False)
            elif way == 'fs':
                if set(self.standard_cols).issubset(set(new_data_df.columns)):
                    self.utils.logger.info('writing new_data_df to path: %s. columns of new dataframe are: %s'%(way_path, self.standard_cols))
                    df = new_data_df[self.standard_cols]
                    df.to_csv(way_path, index=False)
                else:
                    raise OSError('could not write new_data_df, the columns are not standard. the standard columns are:', self.standard_cols)
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

    def update_all_stock_data(self, start_date, end_date, fs_path=None, code_set=None, autype='D', way='web_k'):
        '''
        update all stock data

        ---
        return None
        '''
        if fs_path is None:
            fs_path = self.fs_path
        if not os.path.exists(fs_path):
            raise OSError('path %s does not exist.'%fs_path)
        stock_basics = ts.get_stock_basics()
        new_code_set = set(stock_basics.index) if code_set is None else set(code_set)
        code_list = list(new_code_set)
        tmp_fail_code_list = []
        fail_code_list = []

        def update_one_code(code, i, processors, utils, get_stock_data, update_stock_data):
            try:
                utils.logger.info('start downloading new data %s, i: %s, processor: %s'%(code, i, processors))
                conn = ts.get_apis()
                for i in range(3):
                    new_data_df = get_stock_data(code, start_date, end_date, 'web_bar', conn, autype)
                    if new_data_df is not None and len(new_data_df)>0:
                        break
                    else:
                        t = random.choice([1.1,1.3,1.5])
                        time.sleep(t)

                ts.close_apis(conn)
                update_stock_data(code, new_data_df, 'fs', fs_path, autype)
                t = random.choice([1.1,1.3,1.5])
                time.sleep(t)
                utils.logger.info('finish process new data %s, i: %s, processor: %s'%(code, i, processors))
                return None if new_data_df is not None and len(new_data_df) > 0 else code
            except OSError as err:
                self.utils.logger.info('Error: could not update %s. err msg: %s \nadding to list and will try again later.'%(code, err))
                fail_code_list.append(code)
                ts.close_apis(conn)
                t = random.choice([20,30,25])
                time.sleep(t)
                return code


        self.utils.logger.info('updating all stock data...')
        processors = 12
        pool = ThreadPool(processors)
        results = []

        for i, code in enumerate(code_list):
            # update_one_code(code, i, self.utils, self.get_stock_data, self.update_stock_data)
            r = pool.apply_async(update_one_code,
                        args=(code, i, i%processors, self.utils, self.get_stock_data, self.update_stock_data,))
            results.append(r)
        pool.close()
        pool.join()

        self.utils.logger.info('here')
        
        for r in results:
            tmp_fail_code_list.append(r.get())

        tmp_fail_code_list = [code for code in tmp_fail_code_list if code is not None]

        fail_cnt = len(tmp_fail_code_list)
        success_cnt = len(code_list) - fail_cnt
        self.utils.logger.info('updated file number : %s. failed and re-trying number : %s'%(success_cnt, fail_cnt))
        self.utils.logger.info('==========\n==========\nnow checking for failed stock code again...')

        pool = ThreadPool(processors)
        results = []
        for i, code in enumerate(tmp_fail_code_list):
            cur_conn = conn[i%processors]
            r = pool.apply_async(update_one_code,
                        args=(code, i, i%processor, self.utils, self.get_stock_data, self.update_stock_data,))
            results.append(r)
        pool.close()
        pool.join()

        for r in results:
            fail_code_list.append(r.get())

        fail_code_list = [code for code in fail_code_list if code is not None]
        self.utils.logger.info('==========\n==========\nFailed code set: %s'%fail_code_list)


if __name__ == "__main__":
    dl = DataLoader()
    # dl.update_all_stock_data('2018-09-09','2018-09-15', autype='D')
    dl.update_all_stock_data('2017-12-19','2018-09-15', autype='D')
    dl.update_all_stock_data('2017-12-19','2018-09-15', autype='W')
    dl.update_all_stock_data('2017-12-19','2018-09-15', autype='60MIN')
