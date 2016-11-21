#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from qntstock.index import *
from qntstock.factor_base_func import *


def factor_macd_cross(sdf, fast=12, slow=26, avg=9):
    """
    factor_macd_cross(sdf, fast=12, slow=26, avg=9):

    计算MACD交叉信号，MACD出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

    Output:
        (Series): MACD交叉信号
    """
    rdf = macd(sdf, fast, slow, avg)
    signal = cross_c(rdf['MACD_BAR'])
    return signal


def factor_macd_fast_inflection(sdf, fast=12, slow=26, avg=9):
    """
    factor_macd_fast_inflection(sdf, fast=12, slow=26, avg=9):

    计算MACD快线拐头信号，向上拐头当天1,向下拐头当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

    Output:
        (Series): MACD快线拐头信号
    """
    rdf = macd(sdf, fast, slow, avg)
    signal = inflection_c(rdf['MACD_DIFF'])
    return signal


def factor_macd_slow_inflection(sdf, fast=12, slow=26, avg=9):
    """
    factor_macd_slow_inflection(sdf, fast=12, slow=26, avg=9):

    计算MACD慢线拐头信号，向上拐头当天1,向下拐头当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

        fast: (int): DIFF中的快线时间窗

        slow: (int): DIFF中的慢线时间窗

        avg: (int): DEA中的平滑时间窗

    Output:
        (Series): MACD慢线拐头信号
    """
    rdf = macd(sdf, fast, slow, avg)
    signal = inflection_c(rdf['MACD_DIFF'])
    return signal


def factor_bbi_cross(sdf):
    """
    factor_bbi_cross_1(sdf):

    计算BBI交叉信号，BBI出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 股票数据，至少包括['close']

    Output:
        (Series): BBI交叉信号
    """
    rdf = bbi(sdf)
    rdf['BAR'] = rdf['close']-sdf['BBI']
    signal = cross_c(rdf['BAR'])
    return signal


def factor_rsi_cross(sdf, n=(6, 12)):
    """
    factor_rsi_cross(sdf, n=[6, 12]):

    计算给定两条RSI交叉信号，MACD出现金叉当天1,出现死叉当天为-1,其他为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 参与计算的两条RSI的时间窗，如[6,12]，以第一条线上穿第二条线为金叉

    Output:
        (Series): RSI交叉信号
    """
    rdf = rsi(sdf, n)
    rdf['BAR'] = rdf['RSI_'+str(n[0])]-rdf['RSI_'+str(n[1])]
    signal = cross_c(rdf['BAR'])
    return signal


def factor_ma_raise(sdf, n=(5,10,20,30,60), strict=True):
    """
    factor_ma_raise(sdf, n=[5,10,20,30,60], strict=True):

    计算给定一组天数到均线是否都呈上升趋势，若都上升，则为1，反之则为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 要计算的移动平均线的时间窗，如[5,10,...]

        strict: (boolean): True为严格上升，走平为0；Flase为宽松条件，上升途中走平也认为在上升

    Output:
        (series): MA线组上升信号
    """
    rdf = ma(sdf, n)
    name_list = []
    for ma_n in n:
        col_name = 'MA_'+str(ma_n)+'RAISE'
        rdf[col_name] = wave(rdf['MA_'+str(ma_n)], strict)
        rdf[col_name] = rdf[col_name].map(lambda x: True if x > 0 else False)
        name_list.append(col_name)
    signal = rdf[name_list].all(axis=1).map(lambda x: 1 if x is True else 0)
    return signal


def factor_ma_long_position(sdf, n=(5,10,20,30,60), strict=True):
    """
    factor_ma_long_position(sdf, n=[5,10,20,30,60]):

    计算给定一组天数到均线是否多头排列，若完全多头（不包含走平），则为1，反之则为0

    Input:
        sdf: (DataFrame): 按时间升序排好到股票数据，至少包括['close']

        n: (list of int): 严格按升序排列的移动平均线的时间窗，如[5,10,...]

        strict: (boolean): True为严格多头，两条线相等为0；Flase为宽松条件，两条线相等时看之前是不是多头

    Output:
        (series): MA线组多头信号
    """
    #def func(x, n):
    #    r = 1
    #    for i in range(1,len(n)):
    #        if x['MA_'+str(n[i-1])] <= x['MA_'+str(n[i])]:
    #            r = 0
    #            break
    #    return r
    num = sorted(n)
    ma_name=map(lambda x: 'MA_' + str(x), num)
    rdf = ma(sdf, n)
    name_list=[]
    for i in range(1,len(num)):
        name = 'MA_LONG_POSITION_'+str(i)
        rdf[name] = upper_than(rdf[ma_name[i-1]], rdf[ma_name[i]], strict)
        name_list.append(name)
    signal = rdf[name_list].apply(axis=1)   #TODO: all emplement are 1
    # signal = rdf.apply(func, axis=1, args=(n,))
    return signal
