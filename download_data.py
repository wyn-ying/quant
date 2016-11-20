#!/usr/bin/python
# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
from qntstock.database import get_connection
from qntstock.utils import PATH
from time import sleep
from datetime import date
import numpy as np
import os


def download_data_append_hfq(start_date, end_date=None, \
                            from_code=None, update_list=False):
    record = open(PATH+'/log/log.txt', 'a')
    conn6 = get_connection(stock_pool='6')
    conn3 = get_connection(stock_pool='3')
    conn0 = get_connection(stock_pool='0')
    if update_list:
        codelist = ts.get_stock_basics()
        try:
            os.system('rm '+PATH+'/data/code_list.npy')
        except Exception as err:
            pass
        # os.system('rm /home/wyn/stock/code_list_time.npy')
        np.save(PATH+'/data/code_list.npy', codelist.index)
        # np.save('/home/wyn/stock/code_list_time.npy', codelist['timeToMarket'].values)
    log_dir = PATH+'/log/'+str(date.today())
    try:
        os.system('mkdir '+log_dir)
    except Exception as err:
        pass
    stock_code_list = list(np.load(PATH+'/data/code_list.npy'))

    start_idx=0
    if from_code is not None:
        for stock_code in stock_code_list:
            start_idx += 1
            if stock_code == from_code:
                break

    #start_idx += 1

    cnt = start_idx
    for stock_code in stock_code_list[start_idx:]:
        stock_code_sql = ('sh' if stock_code[0] == '6' else 'sz') + stock_code
        print('\n'+str(cnt)+'start:'+stock_code_sql)
        record.writelines(str(cnt)+' start:'+stock_code_sql+'-----')
        data = ts.get_h_data(stock_code, start=start_date, end=end_date,\
                            retry_count=5, pause=0.1, autype='hfq', drop_factor=False)
        if data is None:
            errorlog = open(log_dir+'/errorlog_'+stock_code_sql+'.txt', 'a')
            errorlog.writelines('\nERROR:::data of'+stock_code_sql \
                        +'in 07-16~07-22 may be missed. \n')
            errorlog.close()
        else:
            data = data.reindex(index=data.index[::-1])
            if stock_code[0] == '6':
                data.to_sql(name=stock_code_sql, con=conn6, if_exists='append')
            elif stock_code[0] == '3':
                data.to_sql(name=stock_code_sql, con=conn3, if_exists='append')
            else:
                data.to_sql(name=stock_code_sql, con=conn0, if_exists='append')

        record.writelines(str(cnt)+':'+stock_code_sql+'has finished. \n')
        print(str(cnt)+':'+stock_code_sql+'has finished. \n')
        cnt += 1
    record.close()


def fix_data(code, start_date, end_date):
    stock_pool = cdoe[2]
    conn = get_connection(stock_pool=stock_pool)
    data = ts.get_h_data(code[2:8], start = start_date, end = end_date,\
                retry_count = 5, pause=0.1,autype = 'hfq', drop_factor=False)
    if data is None:
        print('error.\n')
    else:
        data = data[::-1]
        data.to_sql(name=code, con=conn,if_exists='append')


if __name__ == '__main__':
    download_data_append_hfq(start_date='2016-10-13', end_date='2016-11-14',\
                            from_code=None, update_list=True)
    # fix_data(code='sz000028', start_date='2016-10-28', end_date='2016-11-14')
