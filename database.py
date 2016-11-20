#!/usr/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
import pandas as pd
from utils import PATH, DB_PATH, _sort



def get_connection(stock_pool='0'):
    """
    get_connection(stock_pool='0'):

    Get connection with database using sqlalchemy

    Input:
        stock_pool: (string): '0' or '3' or '6', stock pool to connect

    Return:
        conn: a handle of connection
    """
    conn = create_engine(DB_PATH+'/stock_' + stock_pool)
    return conn


def get_stock_list(stock_pool='0'):
    """
    get_stock_list(stock_pool='0'):

    Get the list of stock codes in database pools

    Input:
        stock_pool: (string): '0' or '3' or '6', stock pool to connect

    Return:
        Array: list of stock codes, , like ['sh600001', 'sh600002', ...]
    """
    conn = get_connection(stock_pool=stock_pool)
    tdf = pd.read_sql("select table_name from information_schema.tables \
                            where table_schema='stock_" + stock_pool \
                            + "' and table_type='base table';", conn)
    test_list = tdf['table_name'].values
    return test_list


def get_df(code, conn=None):
    """
    get_df(code, conn=None):

    Get the sorted dataframe of stock

    Input:
        code: (string): stock code with or without pool code, like 'sh603588', '300121'
        conn: connection handle of the code, if None it will be slow to create handle locally based on parameter 'code'.

    Return:
        Array: list of stock codes, , like ['sh600001', 'sh600002', ...]
    """
    if len(code) == 6:
        code = ('sh' if code[0]=='6' else 'sz') + code
    if conn is None:
        conn = get_connection(stock_pool=code[2])
    sql = "select distinct * from "+code_name+";"
    df = pd.read_sql(sql, conn)
    df = _sort(df)
    return df


def remove_duplication(code_list):
    """
    remove_duplication(code_list):

    Remove duplicate records in database

    Input:
        code_list: (list of string): list of codes to process, like ['300019', ...]

    Return:
        None
    """
    conn6 = get_connection(stock_pool='6')
    conn3 = get_connection(stock_pool='3')
    conn0 = get_connection(stock_pool='0')

    conn={'0':conn0, '3':conn3, '6':conn6}
    for code in code_list:
        print(code)
        code_name = ('sh' if code[0]=='6' else 'sz') + code
        sql = "select distinct * from "+code_name+";"
        df = pd.read_sql(sql, conn[code[0]])
        #df.drop('index', axis=1,inplace=True)
        connn = conn[code[0]]
        df.to_sql(name=code_name, con=connn, if_exists='replace', index=False)

if __name__ == '__main__':
    #
    # Use remove_duplication
    #
    code_list = ['300019', '600634', '300548', '300287', '300551', '603887']
    remove_duplication(code_list)
