#/usr/bin/python
# -*- coding: utf-8 -*-

from qntstock.local import FS_PATH, FS_PATH_OL, token
from qntstock.utils import Utils
from multiprocessing.pool import ThreadPool
from pymongo import DESCENDING, ASCENDING
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
        self.standard_cols = ['date','open','close','high','low','volume','tor']
        self.utils = Utils()

    def get_stock_data(self, code, start_date=None, end_date=None, way='db', way_path=None, autype='hfq', ktype='D', ex_cols=None):
        '''
        get stock data df

        TODO: add choice about autype: hfq, qfq, zc
        ---
        way:    'csv':  do not use code. way_path should be given an absolute path of stock data csv file. if default, return an empty
                pd.DataFrame()

                'fs':   way_path should be given a path that saving all pre-download stock data files. if default,
                using FS_PATH in local.py. if no stock code in the path, return an empty pd.DataFrame()

                'web_k': do not need way_path

                # 'db':   way_path should be given a series of using databases. if default, using 'stock' series. if no
                # dataframe returned, return an empty pd.DataFrame()
        ---
        return dataframe
        '''

        if way not in ['csv', 'fs', 'web_bar', 'web_k', 'db']:
            raise OSError('"way" accept for "csv", "fs", "web".')

        if way =='fs':
            if way_path is None:
                way_path = self.fs_path
            way_path = os.path.join(way_path, ktype, '%s.csv'%code)

        if way in ['csv', 'fs'] and not os.path.exists(way_path):
            raise OSError('file ', way_path, 'does not exist.')

        if way == 'db':
            if way_path is None:
                way_path = 'kdata'
            conn = self.utils.get_conn()
            db = conn[way_path]
            coll = db[ktype]
            query = {'code': code, }
            if start_date is not None or end_date is not None:
                query['date'] = {}
                if start_date is not None:
                    query['date']['$gte'] = start_date
                if end_date is not None:
                    query['date']['$lte'] = end_date

            df = pd.DataFrame(list(coll.find(query)))
            cols = self.standard_cols + ex_cols if ex_cols is not None else self.standard_cols
            df = df[cols]

        elif way == 'web_k':
            df = ts.get_k_data(code, start_date, end_date, autype=autype, ktype=ktype)
        
        # deprecated
        elif way == 'web_bar':
            conn = way_path
            df = ts.bar(code, conn=conn, start_date=start_date, end_date=end_date, adj=autype, freq=ktype, factors=['tor'])
            if df is not None and len(df) > 0:
                df = df.reset_index().sort_values('datetime')
                df = df.rename(columns={'datetime':'date', 'vol':'volume'})
                df = df[self.standard_cols]
            else:
                df = pd.DataFrame(None, columns=self.standard_cols)
        
        else:   #'csv' or 'fs'
            df = pd.read_csv(way_path)

            if way == 'csv' and not set(self.standard_cols).issubset(set(df.columns)):
                raise OSError('the csv file does not contain the necessary columns:\n%s'%self.standard_cols)

            if ktype in ['1MIN','5MIN','15MIN','30MIN','60MIN']:
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

    def update_all(self, start_date, end_date=None, code_set=None, processors=10):
        if end_date is None:
            end_date = self.utils.get_logdate('today')
        # 每天的code基本面数据，写入基本面表

        date_list = self.utils.get_date_list(start_date, end_date)
        for date in date_list:
            try:
                cur_code_df = ts.get_day_all(date)
                if cur_code_df is not None and len(cur_code_df)>0:
                    self.utils.update_daily_df(date, cur_code_df, 'basic', 'dayall')
            except HTTPError as err:
                    if err.msg == 'Not Found':
                        self.utils.logger.warning('date: %s, ts.get_day_all return no data.'%date)
                    else:
                        raise err
        self.utils.logger.info("基本面数据写入完成...")

        # 更新天级k线hfq数据，写入周级表
        def process_one_code_d(code, start_date, end_date):
            utils = Utils()
            utils.logger.info("处理%s日k线数据..."%code)
            df = ts.get_k_data(code, start_date, end_date, autype='hfq', ktype='D')
            # 计算复权因子，写入基本面表
            records = utils.find_code_date( 'basic', 'dayall', code, start_date, end_date)
            if len(df) <= 0:
                return
            df['tor'] = df['date'].apply(lambda x: records[x]['turnover'] if x in records.keys() else None)
            utils.update_kdata_df(code, df, 'kdata', 'D')
            for date in df.date:
                if date not in records.keys():
                    utils.logger.warning("股票代码%s 在%s 没有基本面数据，且存在日k线数据，待人工确认..."%(code, date))
                    return
                nfq_open = records[date]['open']
                hfq_open = df[df.date==date].open.values[0]
                aufactor = hfq_open / nfq_open
                docs.append({'code': code, 'date': date, 'aufactor': aufactor})
            utils.update_doc(docs, 'basic', 'dayall')

        if code_set is None:
            code_set = self.utils.get_all_code_list(start_date, end_date)    
        docs = []
        pool = ThreadPool(processors)
        for code in code_set:
            pool.apply_async(process_one_code_d, args=(code, start_date, end_date))
        pool.close()
        pool.join()
        self.utils.logger.info("日k线数据写入完成，后复权因子写入基本面表完成...")

        # 更新周级k线hfq数据，写入周级表
        def process_one_code_w(code, start_date, end_date):
            utils = Utils()
            utils.logger.info("处理%s周k线数据..."%code)
            df = ts.get_k_data(code, start_date, end_date, autype='hfq', ktype='W')
            if len(df) <= 0:
                return
            records = utils.find_code_date( 'basic', 'dayall', code, start_date, end_date)
            daily_date_list = sorted(records.keys())
            weekly_date_set = set(df['date'])
            if len(weekly_date_set-set(daily_date_list))>0:
                utils.logger.warning('在股票代码%s中，以下日期存在周k线数据但不存在基本面对应数据，待人工确认...%s'%(code, weekly_date_set-set(daily_date_list)))
            tor_dic = {}
            cur_tor = 0
            for date in daily_date_list:
                cur_tor += records[date]['turnover']
                if date in weekly_date_set:
                    tor_dic[date] = cur_tor
                    cur_tor = 0
            df['tor'] = df['date'].apply(lambda x: tor_dic[x] if x in tor_dic.keys() else 0)
            df = df[['date','open','close','high','low','volume','tor','code']]
            utils.update_kdata_df(code, df, 'kdata', 'W')

        if code_set is None:
            code_set = self.utils.get_all_code_list(start_date, end_date)    
        pool = ThreadPool(processors)
        for code in code_set:
            pool.apply_async(process_one_code_w, args=(code, start_date, end_date))
        pool.close()
        pool.join()
        self.utils.logger.info("周k线数据写入完成...")

        # 更新分钟k线数据 （可能是多个分钟级k线list）
        def process_one_code_Mn(code, start_date, end_date, ktype):
            utils = Utils()
            utils.logger.info("处理%s %s分钟k线数据..."%(code, ktype))
            df = ts.get_k_data(code, start_date, end_date, autype='hfq', ktype=ktype)
            if len(df) == 0:
                return
            df = df[(df['date']>=start_date) & (df['date']<=(end_date+'-'))]
            if len(df) == 0:
                return
            #   读每天的复权因子字段，使用复权因子更新价格
            records = utils.find_code_date('basic', 'dayall', code, start_date, end_date)
            #TODO: 如果没有，应该保持和前一天一样的aufactor
            cur_date_set = set(df['date'].apply(lambda x: x[0:10]))
            basic_date_set = set(records.keys())
            if len(cur_date_set-basic_date_set)>0:
                utils.logger.warning('在股票代码%s中，以下日期存在k线数据但不存在基本面对应数据，待人工确认... %s'%(code, cur_date_set-basic_date_set))
            df['aufactor'] = df['date'].apply(lambda x: records[x[0:10]]['aufactor'] if x[0:10] in records.keys() else 1)
            for col in ('open','close','high','low'):
                df[col] = round(df[col] * df['aufactor'], 3)
            #   使用全天交易量和换手率更新换手率
            for _, r in records.items():
                if r['volume'] > 0:
                    r['tor_p_v'] = r['turnover'] / r['volume']
                else:
                    r['tor_p_v'] = int(ktype) / 240.0
            df['tor'] = df.apply(lambda x: records[x['date'][0:10]]['tor_p_v']*x['volume'] if x['date'][0:10] in records.keys() else 0, axis=1)
            df = df[['date','open','close','high','low','volume','tor','code']]
            utils.update_kdata_df(code, df, 'kdata', 'M'+ktype)
        
        if code_set is None:
            code_set = self.utils.get_all_code_list(start_date, end_date)
        for ktype in ('60',):
            pool = ThreadPool(processors)
            for code in code_set:
                pool.apply_async(process_one_code_Mn, args=(code, start_date, end_date, ktype))
            pool.close()
            pool.join()
            self.utils.logger.info("%s分钟k线数据写入完成..."%ktype)
        
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

        def update_one_code(code, idx, processors, utils, get_stock_data, update_stock_data):
            try:
                conn = ts.get_apis()
                new_data_df = None
                utils.logger.info('start downloading new data %s, i: %s, processor: %s'%(code, idx, processors))
                new_data_df = get_stock_data(code, start_date, end_date, 'web_bar', conn, autype)
                ts.close_apis(conn)
                utils.logger.info('close apis for %s, i: %s, processor: %s'%(code, idx, processors))
                t = random.choice([1.1,1.3,1.5])
                time.sleep(t)
                if new_data_df is not None:
                    if len(new_data_df)>0:
                        utils.logger.info('get new data %s, i: %s, processor: %s'%(code, idx, processors))
                    else:
                        utils.logger.info('length of data %s is 0, i: %s, processor: %s'%(code, idx, processors))
                    update_stock_data(code, new_data_df, 'fs', fs_path, autype)
                    utils.logger.info('finish process new data %s, i: %s, processor: %s'%(code, idx, processors))
                return None if new_data_df is not None and len(new_data_df) > 0 else code
            except OSError as err:
                ts.close_apis(conn)
                utils.logger.info('close apis for %s, i: %s, processor: %s'%(code, idx, processors))
                self.utils.logger.info('Error: could not update %s. err msg: %s \nadding to list and will try again later.'%(code, err))
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
            r = pool.apply_async(update_one_code,
                        args=(code, i, i%processors, self.utils, self.get_stock_data, self.update_stock_data,))
            results.append(r)
        pool.close()
        pool.join()

        for r in results:
            fail_code_list.append(r.get())

        fail_code_list = [code for code in fail_code_list if code is not None]
        self.utils.logger.info('==========\n==========\nFailed code set: %s'%fail_code_list)

    def update_all_today_online(self, last_date=None, code_set=None, autype='D'):
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
        lastday_date, lastday_all_df = self._get_nearest_trade_date_and_df_before(lastday_date)

        # 根据昨天的数据和本地csv数据，计算出后复权的权重比例
        # 再计算今天对应的后复权股价，写进csv文件里
        today_all_df = today_all_df[today_all_df['volume']>10]
        lastday_all_df = lastday_all_df[lastday_all_df['volume']>10]
        new_code_set = (set(today_all_df['code']) & set(lastday_all_df['code'])) if code_set is None else set(code_set)

        for code in new_code_set:
            lastday_code_df = lastday_all_df[lastday_all_df['code']==code]
            old_df = self.get_stock_data(code, way='fs')
            if lastday_date not in set(old_df['date']):
                print('Warning: ', code, ' the last trade date in filesystem is not match with it in web.')
                print('    jump to next stock code. if all code makes warning, change the param "last_date".')
            else:
                old_last_series = old_df[old_df['date']==lastday_date]
                adj = self._comput_adj_with(old_last_series, lastday_code_df)

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

    def _get_nearest_trade_date_and_df_before(self, lastday_date):
        lastday = datetime.datetime.strptime(lastday_date, '%Y-%m-%d')
        lastday_all_df = None
        e = None
        for _ in range(30):
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
            pass
            # raise e
        return lastday.strftime('%Y-%m-%d'), lastday_all_df


    def _comput_adj_with(self, old_last_series, lastday_code_df):
        adj1 = old_last_series['open'].values[0] / lastday_code_df['open'].values[0]
        adj2 = old_last_series['high'].values[0] / lastday_code_df['high'].values[0]
        adj3 = old_last_series['low'].values[0] / lastday_code_df['low'].values[0]
        adj4 = old_last_series['close'].values[0] / lastday_code_df['price'].values[0]
        return (adj1 + adj2 + adj3 + adj4) / 4


if __name__ == "__main__":
    dl = DataLoader()
    # dl.update_all_stock_data('2018-09-09','2018-09-15', autype='D')
    dl.update_all('2018-09-09')
