#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import pandas as pd
from deco import concurrent, synchronized
from qntstock.utils import PATH, DB_PATH, _sort, ProgressBar



def get_connection(series='stock', stock_pool='0'):
    """
    get_connection(stock_pool='0'):

    Get connection with database using sqlalchemy

    Input:
        stock_pool: (string): '0' or '3' or '6', stock pool to connect

    Return:
        conn: a handle of connection
    """
    conn = create_engine(DB_PATH+'/' + series + '_' + stock_pool, poolclass=NullPool)
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
    else:
        code_name = code
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

    bar = ProgressBar(total=len(l))
    for code in l:
        bar.log(code)
        sql = 'select distinct * from '+code + ';';
        df = pd.read_sql(sql, conf[code[2]])
        df.to_csv(path_or_buf=to_path + '/' + code + '.csv', index=False)
        bar.move()


def backup_csv_paral(from_series='stock', to_path=PATH+'/data/backup'):
    """
    backup_csv_paral(from_series='stock', to_path=PATH+'/data/backup'):

    Backup data periodly with concurrent tech

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

    _backup_csv_paral(l,from_series, to_path)


@concurrent
def _backup_csv_paral_s(code, from_series, to_path):
    sql = 'select distinct * from '+code + ';';
    con = get_connection(series=from_series, stock_pool=code[2])
    df = pd.read_sql(sql, con)
    df.to_csv(path_or_buf=to_path + '/' + code + '.csv', index=False)

@synchronized
def _backup_csv_paral(l,from_series, to_path):
    for code in l:
        _backup_csv_paral_s(code, from_series, to_path)



def restore(from_path=PATH+'/data/backup', to_series='stock'):
    """
    restore(from_path=PATH+'/data/backup', to_series='stock'):

    Restore data from csv

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

    bar = ProgressBar(total=len(l))
    for i in l:
        bar.log(code)
        tmp=i.split('/')[-1]
        code = tmp.split('.')[0]
        df = pd.read_csv(i)
        df['date'] = df['date'].apply(lambda date: pd.Timestamp(date))
        df.to_sql(name=code, con=cont[code[2]], if_exists='replace', index=False)
        bar.move()


@synchronized
def restore_paral(from_path=PATH+'/data/backup', to_series='stock'):
    """
    restore_paral(from_path=PATH+'/data/backup', to_series='stock'):

    Restore data from csv with concurrent tech

    Input:
        from_from: (string): path of csv backup file

        to_series: (string): series of database to recover

    Return:
        None
    """
    l = _getlist(from_path, [])
    for i in l:
        _restore_paral_s(i, to_series)


@concurrent
def _restore_paral_s(i, series):
    tmp=i.split('/')[-1]
    code = tmp.split('.')[0]
    df = pd.read_csv(i)
    df['date'] = df['date'].apply(lambda date: pd.Timestamp(date))
    try:
        print(code)
        con = get_connection(series=series, stock_pool=code[2])
        df.to_sql(name=code, con=con, if_exists='replace', index=False)
    except Exception as e:
        print("\n------------------------------------\n")

def _getlist(dir, filelist):
    newdir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir=os.path.join(dir,s)
            _getlist(newdir, filelist)
    return filelist


if __name__ == '__main__':
    #
    # Use remove_duplication
    #
    #code_list = ['300019', '600634', '300548', '300287', '300551', '603887']
    #remove_duplication(code_list)
    backup_csv_paral()
    #backup_csv(to_path='/home/wyn/data/test')
    restore_paral()
