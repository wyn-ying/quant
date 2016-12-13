#!/usr/bin/python
# -*- coding: utf-8 -*-

from sqlalchemy import create_engine
import pandas as pd
from qntstock.utils import PATH, DB_PATH, _sort



def get_connection(series='stock', stock_pool='0'):
    """
    get_connection(stock_pool='0'):

    Get connection with database using sqlalchemy

    Input:
        stock_pool: (string): '0' or '3' or '6', stock pool to connect

    Return:
        conn: a handle of connection
    """
    conn = create_engine(DB_PATH+'/' + series + '_' + stock_pool)
    return conn


def get_stock_list(series='stock', stock_pool='0'):
    """
    get_stock_list(stock_pool='0'):

    Get the list of stock codes in database pools

    Input:
        stock_pool: (string): '0' or '3' or '6', stock pool to connect

    Return:
        Array: list of stock codes, like ['sh600001', 'sh600002', ...]
    """
    conn = get_connection(series=series, stock_pool=stock_pool)
    tdf = pd.read_sql("select table_name from information_schema.tables \
                            where table_schema='" + series + "_" + stock_pool \
                            + "' and table_type='base table';", conn)
    test_list = tdf['table_name'].values
    return test_list


def get_df(code, series='stock', conn=None):
    """
    get_df(code, conn=None):

    Get the sorted dataframe of stock

    Input:
        code: (string): stock code with or without pool code, like 'sh603588', '300121'
        conn: connection handle of the code, if None it will be slow to create handle locally based on parameter 'code'.

    Return:
        DataFrame: data of the stock
    """
    if len(code) == 6:
        code_name = ('sh' if code[0] == '6' else 'sz') + code
    if conn is None:
        conn = get_connection(series=series, stock_pool=code[2])
    sql = "select distinct * from " + code_name + ";"
    df = pd.read_sql(sql, conn)
    df = _sort(df)
    return df


def remove_duplication(code_list, series='stock'):
    """
    remove_duplication(code_list):

    Remove duplicate records in database

    Input:
        code_list: (list of string): list of codes to process, like ['300019', ...]

    Return:
        None
    """
    conn6 = get_connection(series=series, stock_pool='6')
    conn3 = get_connection(series=series, stock_pool='3')
    conn0 = get_connection(series=series, stock_pool='0')

    conn={'0':conn0, '3':conn3, '6':conn6}
    for code in code_list:
        print(code)
        code_name = ('sh' if code[0]=='6' else 'sz') + code
        sql = "select distinct * from " + code_name + ";"
        df = pd.read_sql(sql, conn[code[0]])
        connn = conn[code[0]]
        df.to_sql(name=code_name, con=connn, if_exists='replace', index=False)


def copy_mysql(from_series='stock', to_series='stock_backup'):
    """
    copy_mysql(from_series='stock', to_series='stock_backup'):

    Backup or recover data

    Input:
        from_series: (string): series of database copy from

        to_series: (string): series of database copy to

    Return:
        None
    """
    l = []
    for stock_pool in ['0', '3', '6']:
        test_list = get_stock_list(series=from_series, stock_pool=stock_pool)
        l.extend(test_list)

    conf6 = get_connection(series=from_series, stock_pool='6')
    conf3 = get_connection(series=from_series, stock_pool='3')
    conf0 = get_connection(series=from_series, stock_pool='0')
    conf={'0':conf0, '3':conf3, '6':conf6}

    cont6 = get_connection(series=to_series, stock_pool='6')
    cont3 = get_connection(series=to_series, stock_pool='3')
    cont0 = get_connection(series=to_series, stock_pool='0')
    cont={'0':cont0, '3':cont3, '6':cont6}

    for code in l:
        sql = 'select distinct * from '+code + ';';
        df = pd.read_sql(sql, conf[code[2]])
        df.to_sql(name=code, con=cont[code[2]],if_exists='replace', index=False)


def backup_csv(from_series='stock', to_path=PATH+'/data/backup'):
    """
    backup_csv(from_series='stock', to_path=PATH+'/data/backup'):

    Backup data periodly

    Input:
        from_series: (string): series of database backup from

        to_path: (string): path to backup to

    Return:
        None
    """
    l = []
    for stock_pool in ['0', '3', '6']:
        test_list = get_stock_list(series=from_series, stock_pool=stock_pool)
        l.extend(test_list)

    conf6 = get_connection(series=from_series, stock_pool='6')
    conf3 = get_connection(series=from_series, stock_pool='3')
    conf0 = get_connection(series=from_series, stock_pool='0')
    conf={'0':conf0, '3':conf3, '6':conf6}

    for code in l:
        sql = 'select distinct * from '+code + ';';
        df = pd.read_sql(sql, conf[code[2]])
        df.to_csv(path_or_buf=to_path + '/' + code + '.csv', index=False)


def recover(from_path=PATH+'/data/backup', to_series='stock'):
    """
    recover(from_path=PATH+'/data/backup', to_series='stock'):

    Recover data from csv

    Input:
        from_from: (string): path of csv backup file

        to_series: (string): series of database to recover

    Return:
        None
    """
    l = _getlist(from_path, [])
    cont6 = get_connection(series=to_series, stock_pool='6')
    cont3 = get_connection(series=to_series, stock_pool='3')
    cont0 = get_connection(series=to_series, stock_pool='0')
    cont={'0':cont0, '3':cont3, '6':cont6}

    for i in l:
        tmp=i.split('/')[-1]
        code = tmp.split('.')[0]
        print(code)
        df = pd.read_csv(i)
        df.to_sql(name=code, con=cont[code[2]], if_exists='replace', index=False)


def _getlist(dir, filelist):
    newdir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir=os.path.join(dir,s)
            getlist(newdir, filelist)
    return filelist


if __name__ == '__main__':
    #
    # Use remove_duplication
    #
    #code_list = ['300019', '600634', '300548', '300287', '300551', '603887']
    #remove_duplication(code_list)
    backup_csv()
