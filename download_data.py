#!/usr/bin/python
# -*- coding: utf-8 -*-

import tushare as ts
import pandas as pd
from qntstock.database import get_connection
from qntstock.utils import PATH, ProgressBar
from time import sleep
from datetime import date
import numpy as np
import os
import sys


def download_data_append_hfq(start_date, end_date=None, from_code=None, \
                        update_list=False, to_series='stock'):
    conn6 = get_connection(series=to_series, stock_pool='6')
    conn3 = get_connection(series=to_series, stock_pool='3')
    conn0 = get_connection(series=to_series, stock_pool='0')
    if update_list:
        codelist = ts.get_stock_basics()
        try:
            os.system('rm '+PATH+'/data/code_list.npy')
        except Exception as err:
            pass
        # os.system('rm /home/wyn/stock/code_list_time.npy')
        np.save(PATH+'/data/code_list.npy', codelist.index)
        # np.save('/home/wyn/stock/code_list_time.npy', codelist['timeToMarket'].values)
    log_dir = PATH+'/data/log/'+str(date.today())
    record = open(log_dir + '.txt', 'a')

    stock_code_list = list(np.load(PATH+'/data/code_list.npy'))

    start_idx=0
    if from_code is not None:
        for stock_code in stock_code_list:
            start_idx += 1
            if stock_code == from_code:
                break

    #start_idx += 1
    turnover_list = []
    cnt = start_idx
    bar = ProgressBar(count=start_idx, total=len(stock_code_list[start_idx:]))

    for stock_code in stock_code_list[start_idx:]:
        stock_code_sql = ('sh' if stock_code[0] == '6' else 'sz') + stock_code
        bar.log(stock_code_sql)

        data = ts.get_h_data(stock_code, start=start_date, end=end_date,\
                            retry_count=5, pause=0.1, autype='hfq', drop_factor=False)
        data_t = ts.get_hist_data(stock_code, start=start_date, end=end_date,retry_count=5, pause=0.1)

        if data is None or len(data) == 0:
            record.writelines('\nERROR:::data of '+stock_code_sql \
                        +' may be missed. \n')
        else:
            if data_t is None or len(data_t) == 0:
                data['turnover'] = np.NAN
                print('No turnover data')
                turnover_list.append(stock_code)
            else:
                data['turnover'] = list(data_t['turnover'] / 100)

            data = data.reindex(index=data.index[::-1])
            if stock_code[0] == '6':
                data.to_sql(name=stock_code_sql, con=conn6, if_exists='append')
            elif stock_code[0] == '3':
                data.to_sql(name=stock_code_sql, con=conn3, if_exists='append')
            else:
                data.to_sql(name=stock_code_sql, con=conn0, if_exists='append')

        bar.move()
        cnt += 1

    if len(turnover_list) > 0:
        print('缺少换手率数据的有：')
        record.writelines('\n缺少换手率数据的有：\n')
        for code in turnover_list:
            print(code)
            record.writelines(code + ',')
        print('共', len(turnover_list), '只')

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
    download_data_append_hfq(start_date='2016-12-21', end_date='2016-12-22',\
                            from_code=None, update_list=True)
    # fix_data(code='sz000028', start_date='2016-10-28', end_date='2016-11-14')
