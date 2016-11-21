#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import talib


def hhv(sdf, name, num=(10,30,60)):
    """
    hhv(sdf, name, num=(10,30,60)):

    计算一组移动最高值线

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括[name]

        name: (string): 待求值的列名

        n: (int): 求值周期

    Output:
        (DataFrame): 移动最高值线数据，包含[name+'_HHV_'+num, ...]
    """
    tdf=sdf.copy()
    name_list = list()
    for n in num:
        hhv_tmp = tdf[name].rolling(center=False, window=n).mean()
        new_name = name + '_HHV_' + str(n)
        name_list.append(new_name)
        tdf[new_name] = hhv_tmp
    rdf = tdf[name_list]
    return rdf


def llv(sdf, name, num=(10,30,60)):
    """
    llv(sdf, name, num=(10,30,60)):

    计算一组移动最低值线

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括[name]

        name: (string): 待求值的列名

        n: (int): 求值周期

    Output:
        (DataFrame): 移动最低值线数据，包含[name+'_LLV_'+num, ...]
    """
    tdf=sdf.copy()
    name_list = list()
    for n in num:
        llv_tmp = tdf[name].rolling(center=False, window=n).mean()
        new_name = name + '_LLV_' + str(n)
        name_list.append(new_name)
        tdf[new_name] = llv_tmp
    rdf = tdf[name_list]
    return rdf


def ma(sdf, ma_num=(5, 10, 20, 30, 60)):
    """
    ma(sdf, ma_num=[5, 10, 20, 30, 60]):

    计算一组移动平均线

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        ma_num: (list of int): 要计算的移动平均线的时间窗，如[5,10,...]

    Output:
        (DataFrame): 移动平均线数据，形如['MA_5', ...]
    """
    tdf=sdf.copy()
    name_list = list()
    for n in ma_num:
        ma_n = tdf['close'].rolling(center=False, window=n).mean()
        name = 'MA_' + str(n)
        name_list.append(name)
        tdf[name] = round(ma_n, 2)
    rdf = tdf[name_list]
    return rdf


def expma(sdf, expma_num=(5, 10, 20, 30, 60)):
    """
    expma(sdf, expma_num=[5, 10, 20, 30, 60]):

    计算一组指数平均线

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        expma_num: (list of int): 要计算的移动平均线的时间窗，如[5,10,...]

    Output:
        (DataFrame): 指数移动平均线数据，形如['EXPMA_5', ...]
    """
    tdf=sdf.copy()
    name_list = list()
    for n in expma_num:
        expma_n = tdf['close'].ewm(span=n, adjust=False).mean()
        name = 'EXPMA_' + str(n)
        name_list.append(name)
        tdf[name] = round(expma_n, 2)
    rdf = tdf[name_list]
    return rdf


def bbi(sdf):
    """
    bbi(sdf):

    计算BBI

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

    Output:
        (DataFrame): BBI指标数据，包含['BBI']
    """
    tdf=sdf.copy()
    name_list = list()
    rdf=pd.DataFrame()
    for n in (3, 6, 12, 24):
        ma_n = tdf['close'].rolling(center=False, window=n).mean()
        name = 'MA_' + str(n)
        name_list.append(name)
        tdf[name] = round(ma_n, 2)
    tmp = tdf[name_list].apply(sum, axis=1)/4
    dic = {'BBI': tmp}
    rdf = pd.DataFrame(dic)
    return rdf


def macd(sdf, fast=12, slow=26, avg=9):
    """
    macd(sdf, fast=12, slow=26, avg=9):

    计算MACD

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

    Output:
        (DataFrame): MACD指标，包含['MACD_DIFF', 'MACD_DEA', 'MACD_BAR']
    """
    tdf=sdf.copy()
    ema_fast = round(tdf['close'].ewm(span=fast, adjust=False).mean(), 2)
    ema_slow = round(tdf['close'].ewm(span=slow, adjust=False).mean(), 2)
    diff = ema_fast - ema_slow
    dea = round(diff.ewm(span=avg, adjust=False).mean(), 2)
    bar = 2 * (diff - dea)
    dic = {'MACD_BAR': bar, 'MACD_DIFF': diff, 'MACD_DEA': dea}
    rdf = pd.DataFrame(dic)
    return rdf


def kdj(sdf, n=9, m=3):
    """
    kdj(sdf, n=9, m=3):

    计算KDJ指标

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['high', 'low', 'close']

        n: (int): RSV的时间窗

        m: (int): K线和D线的指数平均线的时间窗

    Output:
        (DataFrame): KDJ指标，包含['KDJ_K', 'KDJ_D', 'KDJ_J']
    """
    tdf=sdf.copy()
    low_list = tdf['low'].rolling(window=n, center=False).min()
    low_list = low_list.fillna(value=tdf['low'].expanding(min_periods=1).min())
    high_list = tdf['high'].rolling(window=n, center=False).max()
    high_list=high_list.fillna(value=tdf['high'].expanding(min_periods=1).max())
    rsv = round((tdf['close'] - low_list) / (high_list - low_list) * 100, 1)
    if high_list[len(high_list) - 1] - low_list[len(low_list) - 1] == 0:
        rsv[len(rsv) - 1] = 0
    else:
        rsv[len(rsv) - 1] = 33.3
    kdj_k = round(rsv.ewm(com=m - 1, adjust=False).mean(), 1)
    if not high_list[len(high_list) - 1] - low_list[len(low_list) - 1] == 0:
        kdj_k[len(kdj_k) - 1] = 11.1
    kdj_d = round(kdj_k.ewm(com=m - 1, adjust=False).mean(), 1)
    kdj_j = 3 * kdj_k - 2 * kdj_d
    dic = {'KDJ_K': kdj_k, 'KJD_D': kdj_d, 'KDJ_J': kdj_j}
    rdf = pd.DataFrame(dic)
    return rdf


def rsi(sdf, n=(6, 12, 24)):
    """
    rsi(sdf, n=[6, 12, 24]):

    计算一组RSI指标

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list or tuple): 要计算的RSI的时间窗，如[6,12,...]

    Output:
        (DataFrame): RSI数据，形如['RSI_6', ...]
    """
    tdf=sdf.copy()
    name_list = list()
    for num in n:
        rsi_n = talib.RSI(tdf['close'].values, timeperiod=num)
        name = 'RSI_' + str(num)
        name_list.append(name)
        tdf[name] = round(pd.Series(rsi_n).fillna(0), 1)

    rdf = tdf[name_list]
    return rdf


def wr(sdf, period=14):
    """
    wr(sdf, period=14):

    计算WILLR指标

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['high', 'low', 'close']

        period: (int): WILLR指标的时间窗

    Output:
        (DataFrame): WR数据，包含['WR']
    """
    tdf=sdf.copy()
    high = tdf['high'].values
    low = tdf['low'].values
    close = tdf['close'].values
    tmp = talib.WILLR(high , low, close, period) * -1
    dic = {'WR': tmp}
    rdf = pd.DataFrame(dic)
    return rdf


def boll(sdf,n=20, std=2):
    """
    boll(sdf,n=20, std=2):

    计算BOLL线

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括[close']

        n: (int): 计算标准差的时间窗

        std: (int): 标准差的倍数

    Output:
        (DataFrame): BOLL数据，包含['BOLL_MD', 'BOLL_UPPER', 'BOLL_LOWER']
    """
    tdf=sdf.copy()
    mid = tdf['close'].rolling(center=False, window=n).mean()
    std_line = tdf['close'].rolling(center=False, window=n).std()
    ub = mid + std * std_line
    lb = mid - std * std_line
    dic = {'BOLL_MD': mid, 'BOLL_UPPER': ub, 'BOLL_LOWER': lb}
    rdf = pd.DataFrame(dic)
    return rdf
